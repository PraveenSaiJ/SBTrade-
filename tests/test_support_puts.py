from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.config import PutConfig, load_all
from tool.indicators import hv_rank, hv_series
from tool.ivhistory import make_rank_fn
from tool.levels import NY, compute_levels
from tool.puts import PutRec
from tool.store import Store
from tool.support_puts import (
    collateral_check, pick_support_put, recommend_support_puts, visible_rows, window_expiries,
)
from tool.supports import nearest_support, support_levels

CFG = PutConfig()
FRIDAY = datetime(2026, 10, 23, 11, 0, tzinfo=NY)
MONDAY = datetime(2026, 10, 5, 11, 0, tzinfo=NY)


# ---------------------------------------------------------------- expiry window
def test_expiries_are_fridays_7_to_45_days_out_capped_at_four():
    got = window_expiries(FRIDAY, CFG)
    # Fridays from Oct 30 to Dec 7: Oct 30, Nov 6, 13, 20, 27, Dec 4 -> first two, the monthly (Nov 20), the last (Dec 4)
    assert got == [date(2026, 10, 30), date(2026, 11, 6), date(2026, 11, 20), date(2026, 12, 4)]
    assert all(d.weekday() == 4 and 7 <= (d - FRIDAY.date()).days <= 45 for d in got)


def test_the_expiry_cap_and_window_are_configurable():
    assert window_expiries(FRIDAY, PutConfig(max_expiries=2)) == [date(2026, 10, 30), date(2026, 11, 20)]  # nearest + monthly
    assert window_expiries(MONDAY, PutConfig(min_dte=7, max_dte=11)) == [date(2026, 10, 16)]  # only one Friday fits
    assert window_expiries(MONDAY, PutConfig(min_dte=1, max_dte=3)) == []


# ---------------------------------------------------------------- support levels
def uptrend_bars() -> pd.DataFrame:
    close = np.concatenate([100 + 0.25 * np.arange(250), 162.25 + np.array([1, -1.5, -3.0, -4.2, -3.4])])
    idx = pd.bdate_range(end="2026-10-02", periods=len(close))
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6}, index=idx)


def test_support_levels_are_under_the_price_highest_first_with_labels():
    bars = uptrend_bars()
    spot = float(bars["close"].iloc[-1])
    levels = compute_levels(bars, now=datetime(2026, 10, 2, 17, 0, tzinfo=NY), spot=spot)
    found = support_levels(bars, spot, levels, CFG, now=datetime(2026, 10, 2, 17, 0, tzinfo=NY))
    prices = [p for p, _ in found]
    assert prices == sorted(prices, reverse=True) and all(p < spot for p in prices)
    assert all(label.startswith("zone ") and ":" in label for _, label in found)  # "zone 157.0-159.0: SMA50 + swing low"
    assert nearest_support(found) == found[0] and nearest_support([]) is None


# ---------------------------------------------------------------- strike choice
def puts(rows: dict[float, tuple[float, float]], expiry: date = date(2026, 11, 6)) -> pd.DataFrame:
    """strike -> (bid, delta); deltas are negative like Schwab's put deltas."""
    return pd.DataFrame([
        {"expiry": expiry, "strike": k, "bid": b, "ask": b + 0.05, "delta": d} for k, (b, d) in rows.items()
    ])


def test_the_strike_is_at_or_under_the_support_and_the_stock_price():
    chain = puts({100.0: (3.0, -0.45), 98.0: (2.2, -0.35), 96.0: (1.6, -0.28), 94.0: (1.1, -0.22), 90.0: (0.4, -0.10)})
    row, in_band = pick_support_put(chain, support=97.0, spot=100.0, dte=14, cfg=CFG)
    assert in_band and row["strike"] <= 97.0 and 0.20 <= abs(row["delta"]) <= 0.30
    assert row["strike"] == 96.0  # of 96 (28 delta) and 94 (22 delta) the better annualized return on collateral


def test_the_stock_price_caps_the_strike_when_support_is_above_it():
    chain = puts({100.0: (3.0, -0.50), 96.0: (1.6, -0.27)})
    row, _ = pick_support_put(chain, support=105.0, spot=99.0, dte=14, cfg=CFG)
    assert row["strike"] == 96.0


def test_with_nothing_in_the_band_the_strike_nearest_the_band_is_used_and_flagged():
    chain = puts({96.0: (3.0, -0.40), 94.0: (2.4, -0.36), 90.0: (0.2, -0.05)})
    row, in_band = pick_support_put(chain, support=97.0, spot=100.0, dte=14, cfg=CFG)
    assert not in_band and row["strike"] == 94.0  # |delta| 0.36 is the nearest to the 0.20-0.30 band


def test_no_bid_or_no_eligible_strike_or_no_days_left_gives_none():
    assert pick_support_put(puts({96.0: (0.0, -0.2)}), 97.0, 100.0, 14, CFG) is None
    assert pick_support_put(puts({99.0: (2.0, -0.4)}), 97.0, 100.0, 14, CFG) is None  # above the support
    assert pick_support_put(puts({96.0: (1.0, -0.2)}), 97.0, 100.0, 0, CFG) is None


# ---------------------------------------------------------------- the engine
class Source:
    def __init__(self, bars, spot, chain_for):
        self._bars, self._spot, self._chain_for = bars, spot, chain_for

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars

    def spot(self, symbol):
        return self._spot

    def option_chain(self, symbol, expiry=None, right="C"):
        return self._chain_for(expiry)


def _chain_for(expiry):
    # 11 days to 28 days out: same shape each time, premium grows with time
    extra = 0.1 * (expiry - date(2026, 10, 5)).days
    return puts({160.0: (1.2 + extra, -0.40), 157.0: (0.9 + extra, -0.27), 154.0: (0.6 + extra, -0.21), 150.0: (0.2, -0.10)}, expiry)


def test_engine_rows_have_the_support_fields_and_the_arithmetic():
    bars = uptrend_bars()
    spot = 160.0
    recs = recommend_support_puts(["TEST"], Source(bars, spot, _chain_for), now=MONDAY, cfg=CFG)
    assert recs and all(r.mode == "csp_support" for r in recs)
    good = [r for r in recs if r.status == "recommend"]
    assert good, [r.reason for r in recs]
    r = good[0]
    assert r.strike <= r.support and r.strike < spot and 7 <= r.dte <= 45
    assert r.break_even == pytest.approx(r.strike - r.bid) and r.effective_cost == pytest.approx(r.strike - r.bid)
    assert r.annualized_return == pytest.approx(r.bid / r.strike * 365 / r.dte)
    assert r.prob_otm == pytest.approx(1 - abs(r.delta))
    assert r.collateral_per_contract == pytest.approx(r.strike * 100) and r.support_label
    assert r.rank_source == "HV"  # no IV history was supplied


def test_an_earnings_date_before_expiry_gates_the_row_instead_of_dropping_it():
    from tool.data.earnings import EarningsEvent

    class Events:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, date(2026, 10, 20), "amc")] if end >= date(2026, 10, 20) else []

    recs = recommend_support_puts(["TEST"], Source(uptrend_bars(), 160.0, _chain_for), now=MONDAY, events=Events(), cfg=CFG)
    gated = [r for r in recs if r.status == "earnings_gated"]
    assert gated and all("earnings 2026-10-20" in r.reason and r.strike for r in gated)
    assert any("earnings play" in n for n in gated[0].notes)


def test_downtrends_and_symbols_without_support_are_skipped_with_a_reason():
    src = Source(uptrend_bars(), 160.0, _chain_for)
    [down] = recommend_support_puts(["TEST"], src, now=MONDAY, setup_kinds={"TEST": "downtrend"}, cfg=CFG)
    assert down.status == "skip" and "downtrend" in down.reason

    bars = uptrend_bars()
    [none] = recommend_support_puts(["TEST"], Source(bars, 90.0, _chain_for), now=MONDAY, cfg=CFG)  # price under every level
    assert none.status == "skip" and "no support" in none.reason


def test_a_data_error_becomes_a_skip_row():
    class Broken(Source):
        def spot(self, symbol):
            raise RuntimeError("no quote")

    [rec] = recommend_support_puts(["TEST"], Broken(uptrend_bars(), 160.0, _chain_for), now=MONDAY, cfg=CFG)
    assert rec.status == "skip" and "no quote" in rec.reason


# ---------------------------------------------------------------- earnings gate in the tab
def test_earnings_rows_show_only_with_the_switch_and_a_rank_above_60():
    rows = pd.DataFrame({
        "symbol": ["OK", "HIGH", "LOW", "NONE"],
        "status": ["recommend", "earnings_gated", "earnings_gated", "earnings_gated"],
        "rank": [10.0, 75.0, 40.0, None],
    })
    assert list(visible_rows(rows, earnings_play=False)["symbol"]) == ["OK"]
    assert list(visible_rows(rows, earnings_play=True)["symbol"]) == ["OK", "HIGH"]  # low and unknown ranks never pass
    assert list(visible_rows(rows, earnings_play=True, rank_min=30)["symbol"]) == ["OK", "HIGH", "LOW"]


# ---------------------------------------------------------------- collateral check
def test_collateral_check_under_over_and_unset():
    assert collateral_check(5_000, 20_000)[0] == "ok" and "15,000 left" in collateral_check(5_000, 20_000)[1]
    state, message = collateral_check(25_000, 20_000)
    assert state == "over" and "short $5,000" in message
    state, message = collateral_check(5_000, None)
    assert state == "unset" and "puts: cash" in message
    assert collateral_check(20_000, 20_000)[0] == "ok"  # exactly the cash is allowed


def test_cash_comes_from_config(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("puts:\n  cash: {rh_taxable: 40000, ibkr: 5000}\n  max_expiries: 3\n  min_dte: 10\n", encoding="utf-8")
    cfg = load_all(path).puts
    assert cfg.cash == {"rh_taxable": 40000, "ibkr": 5000} and cfg.max_expiries == 3 and cfg.min_dte == 10
    assert load_all(tmp_path / "missing.yaml").puts.cash == {}  # unset by default: nothing is guessed


# ---------------------------------------------------------------- HV rank and the IV switch
def _closes(returns) -> pd.Series:
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2025-01-01", periods=len(returns)))


def test_hv_rank_is_high_after_a_volatility_spike_and_low_after_a_calm():
    calm, wild = np.tile([0.005, -0.005], 150), np.tile([0.03, -0.03], 10)
    spike = _closes(np.concatenate([calm, wild]))
    assert hv_rank(spike) == pytest.approx(100.0)
    quiet = _closes(np.concatenate([np.tile([0.03, -0.03], 150), np.tile([0.005, -0.005], 10)]))
    assert hv_rank(quiet) == pytest.approx(0.0)
    assert hv_series(spike).dropna().iloc[-1] > hv_series(spike).dropna().iloc[0]


def test_hv_rank_is_none_for_short_or_flat_history():
    assert hv_rank(_closes(np.tile([0.01, -0.01], 20))) is None  # not enough samples
    assert hv_rank(_closes(np.zeros(300))) is None  # perfectly flat: no range


def test_the_rank_switches_from_hv_to_iv_once_sixty_days_exist(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    closes = _closes(np.concatenate([np.tile([0.005, -0.005], 150), np.tile([0.03, -0.03], 10)]))
    rank = make_rank_fn(store)
    assert rank("AAA", closes)[1] == "HV"  # no IV recorded
    for i in range(59):
        store.save_iv(date.fromordinal(date(2026, 7, 1).toordinal() + i), {"AAA": 0.30 + 0.002 * i})
    assert rank("AAA", closes)[1] == "HV"  # 59 days: still not enough
    store.save_iv(date(2026, 9, 29), {"AAA": 0.50})
    value, source = rank("AAA", closes)
    assert source == "IV" and value == pytest.approx(100.0)  # today's 0.50 is the top of its recorded range
    store.close()


# ---------------------------------------------------------------- storage
def test_support_rows_round_trip_through_the_options_table(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    rec = PutRec(
        "AAA", "csp_support", "recommend", "ok", "support", date(2026, 11, 6), 96.0, 1.6, -0.27, 100.0,
        premium_per_contract=160.0, collateral_per_contract=9600.0, return_on_collateral=0.0167, annualized_return=0.43,
        effective_cost=94.4, dte=14, support=97.0, support_label="SMA50", rank=33.0, rank_source="HV", break_even=94.4,
        prob_otm=0.73, notes=["cash-secured: keep $9,600 in cash"],
    )
    store.save_option_recs([("csp_support", [rec])])
    row = store.latest_option_recs("csp_support").iloc[0]
    assert (row["dte"], row["support"], row["support_label"], row["rank_source"]) == (14, 97.0, "SMA50", "HV")
    assert row["break_even"] == 94.4 and row["prob_otm"] == pytest.approx(0.73) and row["rank"] == 33.0
    assert row["collateral"] == 9600.0 and row["source"] == "schwab"
    store.close()


def test_every_expiry_of_a_symbol_survives_saving(tmp_path):
    # the options table is keyed by (run, family, symbol, expiry_type): all four expiries must be kept, not just the last
    recs = recommend_support_puts(["TEST"], Source(uptrend_bars(), 160.0, _chain_for), now=MONDAY, cfg=CFG)
    expiries = {r.expiry for r in recs if r.status == "recommend"}
    assert len(expiries) >= 3
    store = Store(tmp_path / "t.sqlite")
    store.save_option_recs([("csp_support", recs)])
    saved = store.latest_option_recs("csp_support")
    assert len(saved) == len(recs) and set(saved[saved["status"] == "recommend"]["expiry"]) == {e.isoformat() for e in expiries}
    store.close()

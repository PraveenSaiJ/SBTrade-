from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from tool.cc_flags import call_flags, ex_dividend
from tool.cc_monitor import find_rolls, monitor_covered_calls
from tool.config import CoveredCallConfig, load_all
from tool.covered_calls import recommend_covered_calls
from tool.holdings import Holdings, Lot, ShortOption
from tool.income_calls import (
    is_uncapped, net_return, pick_income_strike, recommend_income_calls, tier_of, tier_target,
)
from tool.levels import NY

NOW = datetime(2026, 10, 7, 11, 0, tzinfo=NY)  # Wednesday: the next weekly expiry is Fri Oct 9
EXPIRY = date(2026, 10, 9)
CFG = CoveredCallConfig()


def chain(rows: dict[float, tuple[float, float, float]], expiry: date = EXPIRY) -> pd.DataFrame:
    """strike -> (bid, ask, delta)."""
    return pd.DataFrame([
        {"expiry": expiry, "strike": k, "bid": b, "ask": a, "mid": (b + a) / 2, "delta": d} for k, (b, a, d) in rows.items()
    ])


# ---------------------------------------------------------------- net return and strike choice
def test_net_return_subtracts_the_fee_per_share():
    assert net_return(2.00, 100.0) == pytest.approx(0.02)
    assert net_return(2.00, 100.0, fee_per_contract=0.65) == pytest.approx((2.00 - 0.0065) / 100)


def test_the_income_strike_is_the_one_nearest_the_money_that_meets_the_target():
    ch = chain({99.0: (3.0, 3.1, 0.6), 100.0: (2.0, 2.1, 0.5), 101.0: (1.4, 1.5, 0.4), 103.0: (0.5, 0.6, 0.2)})
    row, net, meets = pick_income_strike(ch, spot=100.0, floor=0.0, target=0.015)
    assert row["strike"] == 100.0 and meets and net == pytest.approx(0.02)  # 99 is in the money: never offered


def test_a_target_nothing_meets_gives_the_best_available_marked_below_target():
    ch = chain({100.0: (1.2, 1.3, 0.5), 101.0: (0.8, 0.9, 0.4), 103.0: (0.3, 0.4, 0.2)})
    row, net, meets = pick_income_strike(ch, spot=100.0, floor=0.0, target=0.015)
    assert row["strike"] == 100.0 and not meets and net == pytest.approx(0.012)  # best available, flagged


def test_the_cost_basis_is_a_floor_even_above_the_money():
    ch = chain({100.0: (2.0, 2.1, 0.5), 104.0: (0.9, 1.0, 0.25), 106.0: (0.5, 0.6, 0.15)})
    row, net, meets = pick_income_strike(ch, spot=100.0, floor=105.0, target=0.004)
    assert row["strike"] == 106.0  # never sell a call under the cost basis


def test_no_eligible_strike_returns_none():
    assert pick_income_strike(chain({95.0: (6.0, 6.1, 0.9)}), spot=100.0, floor=0.0, target=0.015) is None
    assert pick_income_strike(chain({105.0: (0.0, 0.05, 0.1)}), spot=100.0, floor=0.0, target=0.015) is None  # no bid


def test_a_fee_can_push_a_strike_below_target():
    ch = chain({100.0: (1.51, 1.6, 0.5)})
    assert pick_income_strike(ch, 100.0, 0.0, 0.015, fee_per_contract=0.0)[2] is True
    assert pick_income_strike(ch, 100.0, 0.0, 0.015, fee_per_contract=5.0)[2] is False  # $0.05/share fee


# ---------------------------------------------------------------- tiers, uncapped, config
def test_tiers_targets_and_uncapped_defaults():
    assert tier_of("NVDA") == "safe" and tier_of("ZS") == "proven" and tier_of("IREN") == "aggressive"
    assert tier_of("PLTR") == "other"
    assert tier_target("safe") == 0.015 and tier_target("nonexistent") == 0.015
    assert is_uncapped("CRWV", "rh_taxable") and is_uncapped("CRWV", "rh_ira")  # whole symbol
    assert is_uncapped("SNOW", "rh_ira") and not is_uncapped("SNOW", "rh_taxable")  # symbol:account only
    assert not is_uncapped("NVDA", "rh_taxable")


def test_covered_call_config_overrides(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "covered_calls:\n  fee_per_contract: 0.65\n  roll_weeks: 2\n  tier_targets: {safe: 0.01, other: 0.02}\n"
        "  uncapped: [CRWV]\n  events:\n    - {symbol: NVDA, date: 2026-10-08, label: GTC keynote}\n",
        encoding="utf-8",
    )
    cc = load_all(path).covered_calls
    assert cc.fee_per_contract == 0.65 and cc.roll_weeks == 2 and cc.uncapped == ("CRWV",)
    assert cc.tier_targets["safe"] == 0.01 and cc.events[0]["label"] == "GTC keynote"


# ---------------------------------------------------------------- flags
FUND = {"MSFT": {"dividendAmount": 3.92, "dividendDate": "2026-10-08 00:00:00.0"}, "AMZN": {"dividendAmount": 0.0}}


def test_ex_dividend_is_read_from_schwab_fundamentals_as_a_quarterly_amount():
    assert ex_dividend("MSFT", FUND, CFG) == (date(2026, 10, 8), pytest.approx(0.98))
    assert ex_dividend("AMZN", FUND, CFG) == (None, 0.0) and ex_dividend("XYZ", FUND, CFG) == (None, 0.0)


def test_ex_dividend_before_expiry_is_flagged_with_early_assignment_risk_when_time_value_is_thin():
    flags = call_flags("MSFT", NOW.date(), EXPIRY, FUND, spot=520.0, strike=515.0, mid=5.40, cfg=CFG)
    # intrinsic 5.00, time value 0.40 < the 0.98 dividend
    assert any("ex-dividend 2026-10-08" in f for f in flags) and any("early-assignment risk" in f for f in flags)
    rich = call_flags("MSFT", NOW.date(), EXPIRY, FUND, spot=520.0, strike=515.0, mid=8.0, cfg=CFG)
    assert any("ex-dividend" in f for f in rich) and not any("early-assignment" in f for f in rich)
    far = call_flags("MSFT", NOW.date(), EXPIRY, FUND, spot=520.0, strike=560.0, mid=0.4, cfg=CFG)
    assert not any("early-assignment" in f for f in far)  # a deep out-of-the-money call is not exercised for the dividend
    assert call_flags("MSFT", NOW.date(), date(2026, 10, 7), FUND, 520.0, 515.0, 5.4, CFG) == []  # ex-date after expiry


def test_manual_events_inside_the_window_are_flagged():
    cfg = CoveredCallConfig(events=({"symbol": "NVDA", "date": "2026-10-08", "label": "GTC keynote"},
                                    {"symbol": "NVDA", "date": "2026-11-20", "label": "Investor day"}))
    flags = call_flags("NVDA", NOW.date(), EXPIRY, None, cfg=cfg)
    assert flags == ["event: GTC keynote 2026-10-08"]


# ---------------------------------------------------------------- the engine
class Source:
    def __init__(self, spots, chains, bars=None):
        self.spots, self.chains, self._bars = spots, chains, bars

    def spot(self, symbol):
        return self.spots[symbol]

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars

    def option_chain(self, symbol, expiry=None, right="C"):
        return self.chains[(symbol, expiry)].copy()


def holdings(**lots):
    return Holdings(lots=[
        Lot(symbol=sym, account=acct, shares=shares, cost=cost, no_covered_calls=flag)
        for sym, (acct, shares, cost, flag) in lots.items()
    ])


def test_income_rows_skip_uncapped_and_use_tiers_and_flags():
    h = holdings(NVDA=("rh_taxable", 200, 100.0, False), CRWV=("rh_taxable", 100, 50.0, True),
                 SNOW=("rh_ira", 100, 100.0, False), MSFT=("rh_taxable", 100, 300.0, False))
    src = Source(
        {"NVDA": 100.0, "SNOW": 150.0, "MSFT": 520.0},
        {("NVDA", EXPIRY): chain({100.0: (2.0, 2.1, 0.5), 102.0: (1.0, 1.1, 0.3)}),
         ("SNOW", EXPIRY): chain({150.0: (3.0, 3.1, 0.5)}),
         ("MSFT", EXPIRY): chain({520.0: (9.0, 9.2, 0.5)})},
    )
    out = recommend_income_calls(h, src, NOW, fundamentals=FUND)
    by = {r.symbol: r for r in out}
    assert set(by) == {"NVDA", "MSFT"}  # CRWV flagged, SNOW:rh_ira uncapped by config
    nvda = by["NVDA"]
    assert nvda.mode == "income" and nvda.tier == "safe" and nvda.strike == 100.0 and nvda.free_lots == 2
    assert nvda.net_pct == pytest.approx(0.02) and nvda.meets_target and nvda.prob_itm == pytest.approx(0.5)
    assert any("ex-dividend" in f for f in by["MSFT"].flags)


def test_income_rows_flag_earnings_instead_of_skipping_them():
    from tool.data.earnings import EarningsEvent

    class Events:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, date(2026, 10, 8), "amc")]

    h = holdings(NVDA=("rh_taxable", 100, 90.0, False))
    src = Source({"NVDA": 100.0}, {("NVDA", EXPIRY): chain({100.0: (2.0, 2.1, 0.5)})})
    [rec] = recommend_income_calls(h, src, NOW, events=Events())
    assert rec.status == "recommend" and any("earnings 2026-10-08" in f for f in rec.flags)


def test_a_position_with_no_eligible_strike_gets_a_skip_row():
    h = holdings(NVDA=("rh_taxable", 100, 90.0, False))
    src = Source({"NVDA": 100.0}, {("NVDA", EXPIRY): chain({95.0: (6.0, 6.1, 0.9)})})
    [rec] = recommend_income_calls(h, src, NOW)
    assert rec.status == "skip" and rec.mode == "income" and "no strike" in rec.reason


def test_the_guardrail_engine_marks_uncapped_positions_and_sets_tiers():
    h = holdings(CRWV=("rh_taxable", 100, 50.0, True), SNOW=("rh_ira", 100, 100.0, False))
    [crwv, snow] = recommend_covered_calls(h, Source({}, {}), NOW)
    assert crwv.status == snow.status == "skip" and "uncapped" in crwv.reason and "uncapped" in snow.reason
    assert (crwv.tier, snow.tier, crwv.mode) == ("aggressive", "proven", "guardrail")


# ---------------------------------------------------------------- rolls
def roll_source(weekly: dict[date, dict[float, tuple[float, float, float]]]):
    class RollSource:
        def option_chain(self, symbol, expiry=None, right="C"):
            if expiry not in weekly:
                raise ValueError("No expirations")
            return chain(weekly[expiry], expiry)

    return RollSource()


def test_rolls_cover_the_same_expiry_and_three_weeks_and_keep_only_net_credits():
    w1, w2, w3 = EXPIRY + timedelta(days=7), EXPIRY + timedelta(days=14), EXPIRY + timedelta(days=21)
    src = roll_source({
        EXPIRY: {100.0: (1.4, 1.5, 0.5), 102.0: (0.6, 0.7, 0.3)},  # same expiry, higher strike: sells 0.60 vs buy 1.50: debit
        w1: {100.0: (2.60, 2.70, 0.55), 102.0: (1.80, 1.90, 0.4), 105.0: (0.70, 0.80, 0.2)},
        w2: {100.0: (3.50, 3.60, 0.55), 110.0: (0.40, 0.50, 0.1)},
        w3: {100.0: (1.0, 1.1, 0.5)},  # sells 1.00 vs buy 1.50: debit
    })
    credits, best_debit, blocked = find_rolls("TEST", 100.0, EXPIRY, 1.50, src, None, NOW.date(), 3)
    assert blocked is None
    assert [(r.weeks_out, r.strike, round(r.net_credit, 2)) for r in credits] == [
        (1, 100.0, 1.10), (1, 102.0, 0.30), (2, 100.0, 2.00),
    ]  # best first: nearest expiry, then the larger credit; a 105 strike (-0.70) and w3 debits are dropped
    assert best_debit is not None and best_debit.net_credit < 0


def test_rolls_never_go_to_a_lower_strike_and_never_into_earnings():
    from tool.data.earnings import EarningsEvent

    class Events:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, EXPIRY + timedelta(days=3), "bmo")]

    w1 = EXPIRY + timedelta(days=7)
    src = roll_source({w1: {95.0: (6.0, 6.1, 0.8), 100.0: (2.6, 2.7, 0.5)}})
    credits, _, blocked = find_rolls("TEST", 100.0, EXPIRY, 1.5, src, None, NOW.date(), 1)
    assert [r.strike for r in credits] == [100.0]  # 95 is a lower strike
    credits, _, blocked = find_rolls("TEST", 100.0, EXPIRY, 1.5, src, Events(), NOW.date(), 1)
    assert credits == [] and "earnings" in blocked


def test_the_monitor_attaches_rolls_to_a_held_call_and_stores_them(tmp_path):
    w1 = EXPIRY + timedelta(days=7)
    chains = {
        ("TEST", EXPIRY): chain({100.0: (0.10, 0.15, 0.1), 102.0: (0.05, 0.08, 0.05)}),
        ("TEST", w1): chain({100.0: (0.40, 0.45, 0.2)}, w1),
    }

    class S(Source):
        def option_chain(self, symbol, expiry=None, right="C"):
            if (symbol, expiry) not in self.chains:
                raise ValueError("no chain")
            return self.chains[(symbol, expiry)].copy()

    h = Holdings(
        lots=[Lot(symbol="TEST", account="rh_taxable", shares=100, cost=80.0, no_covered_calls=True)],
        short_options=[ShortOption(symbol="TEST", type="call", strike=100.0, expiry=EXPIRY, qty=1)],
    )
    [row] = monitor_covered_calls(h, S({"TEST": 90.0}, chains), now=NOW)
    assert row.action == "HOLD" and [(r.weeks_out, r.strike) for r in row.rolls] == [(1, 100.0)]  # 0.40 - 0.15 = 0.25

    import json

    from tool.store import Store

    store = Store(tmp_path / "t.sqlite")
    store.save_cc_snapshot([row])
    saved = json.loads(store.latest_cc_snapshot().iloc[0]["rolls"])
    assert saved[0]["strike"] == 100.0 and saved[0]["expiry"] == str(w1) and saved[0]["net_credit"] == pytest.approx(0.25)
    store.close()


# ---------------------------------------------------------------- storage
def test_income_and_guardrail_rows_for_one_position_can_both_be_saved(tmp_path):
    from dataclasses import replace

    from tool.store import Store

    h = holdings(NVDA=("rh_taxable", 100, 90.0, False))
    src = Source({"NVDA": 100.0}, {("NVDA", EXPIRY): chain({100.0: (2.0, 2.1, 0.5)})})
    [income] = recommend_income_calls(h, src, NOW)
    twin = replace(income, mode="guardrail", net_pct=None, target_pct=None, meets_target=None)
    store = Store(tmp_path / "t.sqlite")
    store.save_covered_calls([twin, income])  # same run, symbol and account: only `mode` differs
    df = store.latest_covered_calls()
    assert sorted(df["mode"]) == ["guardrail", "income"]
    row = df[df["mode"] == "income"].iloc[0]
    assert row["tier"] == "safe" and row["meets_target"] == 1 and row["net_pct"] == pytest.approx(0.02)
    assert row["source"] == "schwab,finnhub"
    store.close()


def test_an_old_covered_calls_table_is_rebuilt_with_mode_in_the_key(tmp_path):
    import sqlite3

    from tool.store import Store

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE covered_calls (run_ts TEXT NOT NULL, symbol TEXT NOT NULL, account TEXT NOT NULL, status TEXT NOT NULL,"
        " reason TEXT, free_lots INTEGER, expiry TEXT, strike REAL, bid REAL, delta REAL, notes TEXT,"
        " PRIMARY KEY (run_ts, symbol, account));"
        "INSERT INTO covered_calls VALUES ('2026-10-01T10:00:00-04:00','NVDA','rh_taxable','recommend','ok',1,'2026-10-02',200,1.0,0.3,'');"
    )
    conn.commit()
    conn.close()
    store = Store(path)
    pk = {r[1] for r in store._conn.execute("PRAGMA table_info(covered_calls)") if r[5]}
    assert "mode" in pk
    old = store._conn.execute("SELECT symbol, mode, strike FROM covered_calls").fetchall()
    assert old == [("NVDA", "guardrail", 200.0)]  # the old row survived, as a guardrail row
    store.close()

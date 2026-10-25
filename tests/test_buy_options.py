from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.buy_options import (
    Plan,
    candidate_symbols,
    make_plan,
    pick_lotto,
    pick_swing,
    recommend_buys,
    trigger_status,
)
from tool.holdings import Holdings, Lot
from tool.levels import NY

NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)  # Friday after close: weekly expiry 10/9, monthly 10/16


class FakeSource:
    def __init__(self, bars, spot, weekly, monthly):
        self._bars, self._spot, self._weekly, self._monthly = bars, spot, weekly, monthly

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars

    def spot(self, symbol):
        return self._spot

    def option_chain(self, symbol, expiry=None, right="C"):
        chain = (self._weekly if expiry == date(2026, 10, 9) else self._monthly).copy()
        chain["expiry"] = expiry
        return chain


def flat_bars() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame(
        {"open": [100.0] * 25, "high": [102.0] * 25, "low": [98.0] * 25, "close": [100.0] * 25, "volume": [1e6] * 25},
        index=dates,
    )


def breakout_bars() -> pd.DataFrame:
    """The flat 25 sessions plus a final session that closes at 104 above the 102 high on 2.5x volume."""
    bars = flat_bars().copy()
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [101.0, 105.0, 100.5, 104.0, 2.5e6]
    return bars


def falling_bars() -> pd.DataFrame:
    closes = np.linspace(120.0, 100.0, 25)
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame(
        {"open": closes, "high": closes + 2, "low": closes - 2, "close": closes, "volume": [1e6] * 25}, index=dates
    )


def test_breakout_gives_a_call_plan_with_trigger_stop_and_targets():
    from tool.levels import compute_levels

    bars = breakout_bars()
    plan, kind = make_plan("TEST", bars, 103.5, NOW)
    atr = compute_levels(bars, now=NOW, spot=103.5).atr10
    assert kind == "breakout" and plan.side == "call"
    assert plan.trigger == pytest.approx(102.0)  # the prior 20-day high
    # structural stop: the lower of (entry - 0.5 ATR) and the recent swing low (98), within 2 ATR of the entry
    assert plan.stop == pytest.approx(max(min(102.0 - 0.5 * atr, 98.0), 102.0 - 2 * atr))
    assert plan.t1 > plan.trigger and plan.t2 > plan.t1


def test_downtrend_gives_a_put_plan_below_weekly_s1():
    plan, kind = make_plan("TEST", falling_bars(), 100.0, NOW)
    assert kind == "downtrend" and plan.side == "put"
    assert plan.stop > plan.trigger > plan.chase_limit > plan.t1 > plan.t2


def test_range_name_has_no_plan():
    plan, kind = make_plan("TEST", flat_bars(), 100.5, NOW)
    assert plan is None and kind == "range"


def test_trigger_status_for_calls_and_puts():
    call = Plan("call", 102.0, 104.0, 98.0, 110.0, 116.0, "breakout")
    assert [trigger_status(call, s) for s in (100.0, 103.0, 105.0)] == ["WAITING", "TRIGGERED", "EXTENDED"]
    put = Plan("put", 98.0, 96.0, 102.0, 90.0, 84.0, "downtrend")
    assert [trigger_status(put, s) for s in (100.0, 97.0, 95.0)] == ["WAITING", "TRIGGERED", "EXTENDED"]


def test_swing_pick_is_closest_to_half_delta():
    chain = pd.DataFrame({"strike": [100.0, 103.0, 105.0], "delta": [0.70, 0.52, 0.35], "ask": [5.0, 3.0, 2.0]})
    assert pick_swing(chain)["strike"] == 103.0


def test_lotto_pick_maximizes_payoff_multiple_and_ignores_unreachable_strikes():
    chain = pd.DataFrame(
        {"strike": [105.0, 108.0, 112.0], "delta": [0.30, 0.15, 0.05], "ask": [1.5, 0.6, 0.15]}
    )
    pick = pick_lotto(chain, "call", spot=103.0, t1=110.0)
    assert pick["strike"] == 108.0 and pick["multiple"] == pytest.approx((110 - 108) / 0.6)


def test_call_recommendations_carry_entry_stop_target_and_lotto_multiple():
    plan, _ = make_plan("TEST", breakout_bars(), 103.5, NOW)
    weekly = pd.DataFrame(
        {"strike": [plan.t1 - 5, plan.t1 - 2, plan.t1 + 2], "delta": [0.30, 0.15, 0.05], "ask": [1.5, 0.6, 0.15]}
    )
    monthly = pd.DataFrame({"strike": [100.0, 103.0, 105.0], "delta": [0.70, 0.52, 0.35], "ask": [5.0, 3.0, 2.0]})
    out = recommend_buys(["TEST"], FakeSource(breakout_bars(), 103.5, weekly, monthly), now=NOW)
    [lotto] = out["lotto_call"]
    [swing] = out["long_call"]
    assert lotto.status == "recommend" and lotto.strike == pytest.approx(plan.t1 - 2)
    assert lotto.cost_per_contract == pytest.approx(60.0)
    assert lotto.trigger == "TRIGGERED" and lotto.expiry == date(2026, 10, 9)
    assert "Buy calls only if TEST trades above $102.00" in lotto.entry_rule
    assert "whole premium ($60)" in lotto.stop_rule and "Sell half at 2x ($1.20)" in lotto.target_rule
    assert swing.strike == 103.0 and swing.expiry == date(2026, 10, 16)
    assert f"closes below ${plan.stop:.2f}" in swing.stop_rule
    assert f"Sell half at ${plan.t1:.2f} (T1)" in swing.target_rule


def test_put_recommendations_trigger_below_support_with_stop_above():
    plan, _ = make_plan("TEST", falling_bars(), 100.0, NOW)
    t1 = plan.t1
    weekly = pd.DataFrame(
        {"strike": [t1 + 2, t1 + 1], "delta": [-0.20, -0.12], "ask": [0.5, 0.2]}
    )
    monthly = pd.DataFrame({"strike": [100.0, 98.0], "delta": [-0.52, -0.40], "ask": [3.0, 2.0]})
    out = recommend_buys(["TEST"], FakeSource(falling_bars(), 100.0, weekly, monthly), now=NOW)
    [lotto] = out["lotto_put"]
    assert lotto.status == "recommend" and lotto.strike == pytest.approx(t1 + 1)
    assert lotto.entry_rule.startswith("Buy puts only if TEST trades below")
    assert lotto.trigger == "WAITING"  # the stock has not yet broken below weekly S1
    [swing] = out["long_put"]
    assert swing.strike == 100.0 and "closes above" in swing.stop_rule


def test_range_symbol_is_reported_as_no_directional_trade():
    weekly = pd.DataFrame({"strike": [105.0], "delta": [0.2], "ask": [1.0]})
    out = recommend_buys(["TEST"], FakeSource(flat_bars(), 100.5, weekly, weekly), now=NOW)
    [rec] = out["long_call"]
    assert rec.status == "skip" and "no directional trade" in rec.reason


def test_candidates_come_from_holdings_and_watchlist_ranked_by_strength():
    setups = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "kind": ["breakout", "pullback", "downtrend", "downtrend", "breakout", "range"],
            "entry": [1.0, 1.0, None, None, 1.0, None],
            "rr": [2.0, 2.5, None, None, 3.0, None],
            "change_20d": [0.1, 0.05, -0.20, -0.05, 0.2, 0.0],
        }
    )
    holdings = Holdings(
        lots=[Lot(symbol="AAA", account="rh_taxable", shares=100, cost=1.0)],
        watchlist_scan=["BBB", "CCC", "DDD", "FFF"],
    )
    calls, puts = candidate_symbols(setups, holdings)
    assert calls == ["BBB", "AAA"]  # EEE is not held or watched
    assert puts == ["CCC", "DDD"]  # most negative 20-day change first


def test_buy_ideas_expose_numeric_entry_stop_t1_t2():
    plan, _ = make_plan("TEST", breakout_bars(), 103.5, NOW)
    weekly = pd.DataFrame({"strike": [plan.t1 - 2], "delta": [0.15], "ask": [0.6]})
    monthly = pd.DataFrame({"strike": [103.0], "delta": [0.52], "ask": [3.0]})
    out = recommend_buys(["TEST"], FakeSource(breakout_bars(), 103.5, weekly, monthly), now=NOW)
    [lotto] = out["lotto_call"]
    [swing] = out["long_call"]
    assert (lotto.entry_level, lotto.stop_level) == (102.0, 102.0)
    assert (lotto.t1, lotto.t2) == (pytest.approx(plan.t1), pytest.approx(plan.t2))
    assert lotto.stop_option is None  # a lotto risks the whole premium
    assert (swing.entry_level, swing.stop_level) == (102.0, pytest.approx(plan.stop))
    assert (swing.t1, swing.t2) == (pytest.approx(plan.t1), pytest.approx(plan.t2))
    assert swing.stop_option == pytest.approx(1.5)


def test_zero_dte_call_uses_daily_pivots_and_todays_expiry():
    from tool.buy_options import recommend_odte

    chain = pd.DataFrame({"strike": [101.0, 102.0, 103.0], "delta": [0.45, 0.35, 0.20], "ask": [1.0, 0.6, 0.2]})
    source = FakeSource(flat_bars(), 101.0, chain, chain)
    rec = recommend_odte("TEST", source, NOW, events=None)
    # daily pivots from the flat bar: P 100, R1 102, R2 104, S1 98 -> stop halfway between P and S1 = 99
    assert rec.status == "recommend" and rec.expiry == NOW.date() and rec.expiry_type == "0DTE"
    assert rec.strike == 101.0  # the only strike whose intrinsic value at R1 beats its ask
    assert (rec.entry_level, rec.stop_level, rec.t1, rec.t2) == (100.0, 99.0, 102.0, 104.0)
    assert rec.trigger == "TRIGGERED"
    assert "daily pivot" in rec.entry_rule and "close it before the bell" in rec.stop_rule


def test_zero_dte_notes_when_nothing_expires_today():
    from tool.buy_options import recommend_odte

    class NextExpiryOnly(FakeSource):
        def option_chain(self, symbol, expiry=None, right="C"):
            chain = self._monthly.copy()
            chain["expiry"] = date(2026, 10, 9)  # the adapter returns the next listed expiry
            return chain

    chain = pd.DataFrame({"strike": [101.0], "delta": [0.40], "ask": [1.0]})
    rec = recommend_odte("TEST", NextExpiryOnly(flat_bars(), 101.0, chain, chain), NOW, events=None)
    assert rec.expiry_type == "next expiry" and any("no contract expires today" in n for n in rec.notes)


def test_mag7_name_without_a_swing_setup_falls_back_to_weekly_pivots():
    from tool.buy_options import recommend_mag7_calls

    weekly = pd.DataFrame({"strike": [100.0, 101.0, 101.5], "delta": [0.25, 0.20, 0.15], "ask": [0.8, 0.5, 0.3]})
    monthly = pd.DataFrame({"strike": [100.0, 101.0], "delta": [0.58, 0.48], "ask": [3.0, 2.0]})
    recs = recommend_mag7_calls("TEST", FakeSource(flat_bars(), 100.5, weekly, monthly), NOW, events=None)
    lotto = next(r for r in recs if r.style == "lotto")
    swing = next(r for r in recs if r.style == "swing")
    # flat bars give a 'range' setup, so the plan comes from the weekly pivots (P 100, R1 102, R2 104, S1 98)
    assert (lotto.entry_level, lotto.stop_level, lotto.t1, lotto.t2) == (100.0, 100.0, 102.0, 104.0)
    assert "weekly pivot" in lotto.entry_rule and lotto.strike == 100.0
    assert swing.strike == 101.0 and (swing.stop_level, swing.t1) == (98.0, 102.0)  # delta 0.48 is closest to 0.50


def test_special_sections_cover_the_index_etfs_and_all_seven_stocks():
    from tool.buy_options import INDEX_ETFS, MAG7, recommend_special

    chain = pd.DataFrame({"strike": [101.0], "delta": [0.40], "ask": [1.0]})
    out = recommend_special(FakeSource(flat_bars(), 101.0, chain, chain), now=NOW)
    assert {r.symbol for r in out["odte_call"]} == set(INDEX_ETFS + MAG7)
    assert {r.symbol for r in out["mag7_lotto_call"]} == set(MAG7)
    assert {r.symbol for r in out["mag7_swing_call"]} == set(MAG7)

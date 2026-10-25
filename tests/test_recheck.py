from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.config import ScreenerConfig
from tool.levels import NY
from tool.recheck import recheck_row, recheck_setups
from tool.screener import Setup, reward_to_risk
from tool.store import Store

CFG = ScreenerConfig()
NOW = datetime(2026, 10, 22, 11, 0, tzinfo=NY)


def make(symbol, **kw) -> Setup:
    base = dict(symbol=symbol, kind="breakout", spot=104.0, atr=4.0, sma20=100.0, prior_high=102.0, change_20d=0.05,
                entry=102.0, stop=96.0, t1=110.0, t2=114.0, rr=reward_to_risk(110.0, 104.0, 96.0), volume_ratio=2.0,
                last_close=104.0, rsi=60.0, atr14=4.0, sma50=98.0, sma50_rising=True, sma200=90.0, score=0.60)
    base.update(kw)
    return Setup(**base)


def row_of(tmp_path, setup: Setup) -> pd.Series:
    store = Store(tmp_path / "r.sqlite")
    store.save_setups([setup], day=date(2026, 10, 22))
    row = store.latest_setups().iloc[0]
    store.close()
    return row


class QuoteSource:
    def __init__(self, prices):
        self.prices, self.asked = prices, []

    def quotes(self, symbols, ttl=None):
        self.asked.append(list(symbols))
        return {s: self.prices[s] for s in symbols if s in self.prices}


# ---- one row
def test_a_breakout_that_fell_back_under_its_entry_fails_and_is_excluded(tmp_path):
    update = recheck_row(row_of(tmp_path, make("AAA")), 101.0, CFG)
    assert update["kind"] == "failed" and update["excluded"].startswith("failed") and update["spot"] == 101.0


def test_a_breakout_still_above_its_entry_stays_recommended(tmp_path):
    update = recheck_row(row_of(tmp_path, make("AAA")), 103.0, CFG)
    assert update["kind"] == "breakout" and update["excluded"] is None


def test_rr_is_recomputed_from_the_new_fill(tmp_path):
    row = row_of(tmp_path, make("AAA"))  # entry 102, stop 96, T1 110
    higher = recheck_row(row, 106.0, CFG)
    assert higher["rr"] == pytest.approx((110 - 106) / (106 - 96))  # fill = spot when price is above the entry
    at_entry = recheck_row(row_of(tmp_path, make("BBB", kind="pullback", spot=100.0, entry=102.0)), 101.0, CFG)
    assert at_entry["rr"] == pytest.approx((110 - 102) / (102 - 96))  # below the entry the fill is the entry


def test_the_score_moves_only_by_the_rr_term(tmp_path):
    row = row_of(tmp_path, make("AAA"))
    old_rr = row["rr"]
    update = recheck_row(row, 106.0, CFG)
    expected = row["score"] + CFG.w_rr * (update["rr"] / CFG.rr_norm_cap - old_rr / CFG.rr_norm_cap)
    assert update["score"] == pytest.approx(expected) and update["score"] < row["score"]


def test_a_setup_that_ran_more_than_two_atr_above_its_entry_is_overextended(tmp_path):
    row = row_of(tmp_path, make("AAA", kind="pullback", entry=100.0, spot=101.0))
    update = recheck_row(row, 109.0, CFG)  # 9 above entry > 2 x ATR(14) of 4
    assert update["kind"] == "pullback" and update["excluded"].startswith("overextended")


def test_rows_without_atr14_skip_the_overextension_check(tmp_path):
    row = row_of(tmp_path, make("AAA", kind="pullback", entry=100.0, spot=101.0, atr14=None))
    assert recheck_row(row, 150.0, CFG)["excluded"] is None


# ---- the whole pass
def test_recheck_updates_live_setups_with_one_batched_request_and_leaves_excluded_rows(tmp_path):
    store = Store(tmp_path / "r.sqlite")
    store.save_setups([
        make("BRK"),  # breakout that will fall back
        make("PULL", kind="pullback", entry=100.0, spot=101.0),  # stays fine
        make("OLD", excluded="low volume (1.0x vs 1.5x needed)"),  # already excluded: untouched
        make("RNG", kind="range", entry=None, stop=None, t1=None, t2=None, rr=None, score=None),  # not a setup
    ], day=date(2026, 10, 22))
    src = QuoteSource({"BRK": 100.0, "PULL": 102.0, "OLD": 90.0})
    result = recheck_setups(store, src, CFG, NOW)
    assert src.asked == [["BRK", "PULL"]]  # one request, only the live setups
    assert result.checked == 2 and result.failed == ["BRK"] and result.overextended == []
    latest = store.latest_setups().set_index("symbol")
    assert latest.loc["BRK", "kind"] == "failed" and latest.loc["BRK", "spot"] == 100.0
    assert latest.loc["BRK", "rechecked_at"].startswith("2026-10-22T11:00")
    assert latest.loc["PULL", "spot"] == 102.0 and latest.loc["PULL", "kind"] == "pullback"
    assert latest.loc["OLD", "spot"] == 104.0 and pd.isna(latest.loc["OLD", "rechecked_at"])  # untouched
    store.close()


def test_recheck_with_no_live_setups_makes_no_request(tmp_path):
    store = Store(tmp_path / "r.sqlite")
    src = QuoteSource({})
    assert recheck_setups(store, src, CFG, NOW).checked == 0 and src.asked == []
    store.save_setups([make("OLD", excluded="x")], day=date(2026, 10, 22))
    assert recheck_setups(store, src, CFG, NOW).checked == 0 and src.asked == []
    store.close()


def test_a_symbol_without_a_quote_is_reported_not_changed(tmp_path):
    store = Store(tmp_path / "r.sqlite")
    store.save_setups([make("AAA")], day=date(2026, 10, 22))
    result = recheck_setups(store, QuoteSource({}), CFG, NOW)
    assert result.no_quote == ["AAA"] and result.checked == 0
    assert store.latest_setups().iloc[0]["kind"] == "breakout"
    store.close()


def test_a_setup_whose_price_falls_to_its_stop_is_excluded_as_stopped(tmp_path):
    row = row_of(tmp_path, make("AAA", kind="pullback", entry=100.0, spot=101.0))  # stop 96
    update = recheck_row(row, 95.5, CFG)
    assert update["excluded_code"] == "stopped" and update["kind"] == "pullback"

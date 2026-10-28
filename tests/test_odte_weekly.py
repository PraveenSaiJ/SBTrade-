from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from tool.buy_options import (
    INDEX_ETFS, MAG7, parallel_map, recommend_buys, recommend_odte, recommend_odte_all, recommend_special,
)
from tool.levels import NY

NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)  # Friday after close: weekly expiry 10/9, monthly 10/16


class Source:
    def __init__(self, bars, spot, chain):
        self.bars, self._spot, self.chain, self.asked = bars, spot, chain, []

    def daily_bars(self, symbol, lookback_days=90):
        return self.bars

    def spot(self, symbol):
        return self._spot

    def option_chain(self, symbol, expiry=None, right="C", **options):
        self.asked.append((symbol, expiry, right))
        out = self.chain[right].copy()
        out["expiry"] = expiry
        return out


def flat_bars() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame({"open": [100.0] * 25, "high": [102.0] * 25, "low": [98.0] * 25, "close": [100.0] * 25,
                         "volume": [1e6] * 25}, index=dates)


def falling_bars() -> pd.DataFrame:
    closes = np.linspace(120.0, 100.0, 25)
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame({"open": closes, "high": closes + 2, "low": closes - 2, "close": closes, "volume": [1e6] * 25}, index=dates)


CHAINS = {
    "C": pd.DataFrame({"strike": [101.0, 102.0], "delta": [0.40, 0.30], "ask": [0.8, 0.4]}),
    "P": pd.DataFrame({"strike": [99.0, 98.0], "delta": [-0.40, -0.30], "ask": [0.8, 0.4]}),
}


def test_zero_dte_put_uses_daily_pivots_below_the_pivot_with_s1_s2_targets():
    rec = recommend_odte("TEST", Source(flat_bars(), 99.0, CHAINS), NOW, events=None, side="put")
    # flat bar: P 100, R1 102, S1 98, S2 96 -> trigger below P, stop halfway to R1, targets S1 and S2
    assert rec.status == "recommend" and rec.side == "put" and rec.expiry_type == "0DTE"
    assert (rec.entry_level, rec.stop_level, rec.t1, rec.t2) == (100.0, 101.0, 98.0, 96.0)
    assert rec.trigger == "TRIGGERED" and rec.strike in (99.0, 98.0)
    assert "daily pivot" in rec.entry_rule and "S1" in rec.target_rule and "above $101.00" in rec.stop_rule


def test_zero_dte_put_asks_the_put_chain_and_a_put_can_be_waiting_or_extended():
    source = Source(flat_bars(), 101.0, CHAINS)
    rec = recommend_odte("TEST", source, NOW, events=None, side="put")
    assert source.asked == [("TEST", NOW.date(), "P")] and rec.trigger == "WAITING"  # still above the pivot


def test_zero_dte_all_builds_calls_and_puts_for_the_etfs_and_mag7_in_order():
    out = recommend_odte_all(Source(flat_bars(), 99.0, CHAINS), now=NOW, workers=4)
    assert set(out) == {"odte_call", "odte_put"}
    symbols = list(dict.fromkeys(INDEX_ETFS + MAG7))
    assert [r.symbol for r in out["odte_call"]] == symbols and [r.symbol for r in out["odte_put"]] == symbols
    assert all(r.side == "call" for r in out["odte_call"]) and all(r.side == "put" for r in out["odte_put"])


def test_the_buys_job_no_longer_builds_0dte_and_special_can_leave_it_out():
    out = recommend_special(Source(flat_bars(), 101.0, CHAINS), now=NOW, odte=False)
    assert out["odte_call"] == [] and len(out["mag7_lotto_call"]) == 7


def test_weekly_job_builds_only_weekly_expiries_and_buys_only_monthly():
    weekly_only = recommend_buys(["TEST"], Source(falling_bars(), 100.0, CHAINS), now=NOW, include=("weekly",))
    rows = [r for recs in weekly_only.values() for r in recs]
    assert rows and {r.expiry_type for r in rows} == {"weekly"}
    assert weekly_only["long_put"] == [] and weekly_only["long_call"] == []

    monthly_only = recommend_buys(["TEST"], Source(falling_bars(), 100.0, CHAINS), now=NOW, include=("monthly",))
    rows = [r for recs in monthly_only.values() for r in recs]
    assert rows and {r.expiry_type for r in rows} == {"monthly"}
    assert monthly_only["weekly_put"] == [] and monthly_only["lotto_put"] == []


def test_parallel_results_match_the_serial_ones_and_keep_the_input_order():
    symbols = ["A", "B", "C", "D", "E"]
    serial = recommend_buys(symbols, Source(falling_bars(), 100.0, CHAINS), now=NOW, include=("weekly",))
    threaded = recommend_buys(symbols, Source(falling_bars(), 100.0, CHAINS), now=NOW, include=("weekly",), workers=4)
    for family in serial:
        assert [(r.symbol, r.status, r.strike) for r in serial[family]] == [(r.symbol, r.status, r.strike) for r in threaded[family]]
    assert parallel_map(lambda x: x * 2, [3, 1, 2], workers=3) == [6, 2, 4]
    assert parallel_map(lambda x: x, [], workers=3) == []

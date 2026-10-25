from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.levels import NY
from tool.naked_calls import default_symbols, margin_per_contract, recommend_one, _pick_strike
from tool.holdings import Holdings, Lot


class FakeSource:
    def __init__(self, bars, spot, chain):
        self._bars, self._spot, self._chain = bars, spot, chain

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars

    def spot(self, symbol):
        return self._spot

    def option_chain(self, symbol, expiry=None, right="C"):
        chain = self._chain.copy()
        chain["expiry"] = expiry
        return chain


def _flat_bars() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame(
        {"open": [100.0] * 25, "high": [102.0] * 25, "low": [98.0] * 25, "close": [100.0] * 25, "volume": [1e6] * 25},
        index=dates,
    )


NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)


def test_margin_uses_larger_of_the_two_reg_t_formulas():
    # spot 100, strike 105, premium 2: 20%*S - OTM + premium = 2000 - 500 + 200 = 1700 (beats 10%*S + premium = 1200)
    assert margin_per_contract(100.0, 105.0, 2.0) == pytest.approx(1700.0)


def test_margin_floor_when_far_otm():
    # strike 150 (no OTM deduction beyond the floor): 10%*S + premium = 1000 + 200
    assert margin_per_contract(100.0, 150.0, 2.0) == pytest.approx(1200.0)


def test_pick_strike_prefers_delta_band():
    chain = pd.DataFrame({"strike": [100.0, 104.0, 106.0, 110.0], "delta": [0.4, 0.3, 0.2, 0.1]})
    row, in_band = _pick_strike(chain, floor=102.0)
    assert row["strike"] == 106.0 and in_band


def test_pick_strike_falls_back_outside_band():
    chain = pd.DataFrame({"strike": [104.0, 110.0], "delta": [0.3, 0.1]})
    row, in_band = _pick_strike(chain, floor=103.0)
    assert row["strike"] == 104.0 and not in_band


def test_breakout_is_skipped_as_naked_call_candidate():
    source = FakeSource(_flat_bars(), spot=103.0, chain=pd.DataFrame({"strike": [105.0], "delta": [0.2], "bid": [1.0], "ask": [1.1]}))
    rec = recommend_one("TEST", source, NOW, None, "weekly", date(2026, 10, 9))
    assert rec.status == "skip"
    assert "no naked calls into strength" in rec.reason


def test_flat_name_gets_recommendation_with_risk_notes():
    chain = pd.DataFrame(
        {"strike": [103.0, 105.0, 108.0], "delta": [0.30, 0.20, 0.10], "bid": [1.5, 1.0, 0.4], "ask": [1.55, 1.05, 0.45]}
    )
    source = FakeSource(_flat_bars(), spot=100.0, chain=chain)
    rec = recommend_one("TEST", source, NOW, None, "weekly", date(2026, 10, 9))
    assert rec.status == "recommend"
    assert rec.strike == 105.0  # first strike above the 104 weekly R2 floor in the 0.15-0.25 band
    assert rec.premium_per_contract == pytest.approx(100.0)
    assert rec.margin_per_contract and rec.return_on_margin == pytest.approx(100.0 / rec.margin_per_contract)
    assert any("UNLIMITED RISK" in n for n in rec.notes)


def test_default_symbols_excludes_held_names():
    holdings = Holdings(
        lots=[Lot(symbol="NVDA", account="rh_taxable", shares=100, cost=180.0)],
        watchlist_scan=["NVDA", "AMD", "MU"],
    )
    assert default_symbols(holdings) == ["AMD", "MU"]



def test_selling_levels_are_numeric_entry_stop_t1_t2():
    chain = pd.DataFrame({"strike": [105.0], "delta": [0.20], "bid": [1.00], "ask": [1.05]})
    rec = recommend_one("TEST", FakeSource(_flat_bars(), 100.0, chain), NOW, None, "weekly", date(2026, 10, 9))
    assert rec.status == "recommend"
    # resistance floor $104 (weekly R2), ATR 4: sell only below $102, stop on a close above $104
    assert rec.entry_level == pytest.approx(102.0) and rec.stop_level == pytest.approx(104.0)
    assert rec.stop_option == pytest.approx(2.00) and rec.t1 == pytest.approx(0.50) and rec.t2 == pytest.approx(0.20)
    assert "Sell only while TEST is below $102.00" in rec.entry_rule
    assert "closes above $104.00" in rec.stop_rule
    assert "T1: buy back at $0.50" in rec.target_rule and "T2: buy back at $0.20" in rec.target_rule

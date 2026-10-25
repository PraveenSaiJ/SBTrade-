from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.expiries import monthly_expiry
from tool.levels import NY
from tool.puts import naked_put_margin, recommend_one


class FakeSource:
    def __init__(self, bars, spot, chain):
        self._bars, self._spot, self._chain = bars, spot, chain

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars

    def spot(self, symbol):
        return self._spot

    def option_chain(self, symbol, expiry=None, right="P"):
        chain = self._chain.copy()
        chain["expiry"] = expiry
        return chain


def _flat_bars() -> pd.DataFrame:
    # H102/L98/C100 every day -> weekly S1 = 98, 20 sessions back close = 100
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame(
        {"open": [100.0] * 25, "high": [102.0] * 25, "low": [98.0] * 25, "close": [100.0] * 25, "volume": [1e6] * 25},
        index=dates,
    )


NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)
EXPIRY = date(2026, 10, 9)


def _put_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [94.0, 96.0, 98.0, 100.0],
            "bid": [0.50, 1.20, 2.10, 3.00],
            "ask": [0.55, 1.25, 2.20, 3.10],
            "delta": [-0.10, -0.20, -0.45, -0.60],
        }
    )


def test_csp_picks_highest_strike_at_or_below_s1_in_delta_band():
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, _put_chain()), NOW, None, "weekly", EXPIRY)
    assert rec.status == "recommend"
    assert rec.strike == 96.0  # 98 is out of band (|delta| .45); 96 is |delta| .20
    assert rec.collateral_per_contract == pytest.approx(9600.0)
    assert rec.effective_cost == pytest.approx(96.0 - 1.20)
    assert rec.return_on_collateral == pytest.approx(1.20 / 96.0)
    assert rec.annualized_return == pytest.approx(rec.return_on_collateral * 365 / 7)


def test_low_return_put_is_recommended_with_a_warning_note():
    # $96 strike, $0.10 bid, 7 days out, in-band delta: ~5.4% annualized on $9,600 of collateral
    chain = pd.DataFrame({"strike": [96.0], "bid": [0.10], "ask": [0.12], "delta": [-0.20]})
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, chain), NOW, None, "weekly", EXPIRY)
    assert rec.status == "recommend"
    assert any("LOW RETURN" in n for n in rec.notes)


def test_put_outside_delta_band_falls_back_with_a_note():
    chain = pd.DataFrame({"strike": [90.0, 94.0], "bid": [0.20, 0.45], "ask": [0.25, 0.50], "delta": [-0.05, -0.10]})
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, chain), NOW, None, "weekly", EXPIRY)
    assert rec.status == "recommend"
    assert rec.strike == 94.0  # best annualized among eligible strikes
    assert any("outside the band" in n for n in rec.notes)


def test_naked_put_reports_max_loss_and_margin():
    rec = recommend_one("TEST", "naked_put", FakeSource(_flat_bars(), 100.0, _put_chain()), NOW, None, "weekly", EXPIRY)
    assert rec.status == "recommend"
    assert rec.max_loss_per_contract == pytest.approx((96.0 - 1.20) * 100)
    assert rec.margin_per_contract == pytest.approx(naked_put_margin(100.0, 96.0, 1.20))
    assert rec.collateral_per_contract is None


def test_put_margin_uses_larger_reg_t_formula():
    # spot 100, strike 96, premium 1: (20%*S - OTM + premium) = 20 - 4 + 1 = 17 -> 1700 beats (10%*K + premium) -> 1060
    assert naked_put_margin(100.0, 96.0, 1.0) == pytest.approx(1700.0)


def test_downtrend_name_gets_no_put():
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 96.0, _put_chain()), NOW, None, "weekly", EXPIRY)
    assert rec.status == "skip"
    assert "downtrend" in rec.reason


def test_no_strike_at_or_below_s1_is_skipped():
    chain = pd.DataFrame({"strike": [100.0, 102.0], "bid": [3.0, 4.0], "ask": [3.1, 4.1], "delta": [-0.5, -0.6]})
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, chain), NOW, None, "weekly", EXPIRY)
    assert rec.status == "skip"
    assert "at or below weekly S1" in rec.reason


def test_monthly_expiry_is_third_friday():
    assert monthly_expiry(date(2026, 10, 6)) == date(2026, 10, 16)


def test_monthly_expiry_rolls_to_next_month_when_too_close():
    assert monthly_expiry(date(2026, 10, 12)) == date(2026, 11, 20)


def test_chain_with_no_delta_data_does_not_crash_and_notes_it():
    chain = pd.DataFrame({"strike": [94.0, 96.0], "bid": [0.45, 0.80], "ask": [0.50, 0.85], "delta": [None, None]})
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, chain), NOW, None, "weekly", EXPIRY)
    assert rec.status == "recommend"
    assert rec.delta is None or rec.delta != rec.delta  # no delta available
    assert any("outside the band" in n for n in rec.notes)


def test_put_levels_are_numeric_entry_stop_t1_t2():
    rec = recommend_one("TEST", "csp", FakeSource(_flat_bars(), 100.0, _put_chain()), NOW, None, "weekly", EXPIRY)
    assert rec.strike == 96.0 and rec.bid == 1.20
    # weekly S1 is $98, ATR 4: sell only above $100, stop on a close below $98
    assert rec.entry_level == pytest.approx(100.0) and rec.stop_level == pytest.approx(98.0)
    assert rec.stop_option == pytest.approx(2.40) and rec.t1 == pytest.approx(0.60) and rec.t2 == pytest.approx(0.24)
    assert "Sell only while TEST is above $100.00" in rec.entry_rule
    assert "T1: buy back at $0.60" in rec.target_rule and "$94.80 per share" in rec.target_rule
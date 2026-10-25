"""Tests for tool.covered_calls against the rules in CLAUDE.md "Covered-call
engine rules" and "Per-ticker overrides", using a fake DataSource so no
IBKR/Schwab connection is needed."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from tool.covered_calls import recommend_covered_calls
from tool.holdings import Holdings, Lot, ShortOption
from tool.levels import NY


class FakeDataSource:
    def __init__(self, bars: dict[str, pd.DataFrame], spots: dict[str, float], chains: dict[str, pd.DataFrame]):
        self._bars = bars
        self._spots = spots
        self._chains = chains

    def daily_bars(self, symbol, lookback_days=90):
        return self._bars[symbol]

    def spot(self, symbol):
        return self._spots[symbol]

    def option_chain(self, symbol, expiry=None, right="C"):
        chain = self._chains[symbol].copy()
        chain["expiry"] = expiry
        return chain


def _msft_week_bars() -> pd.DataFrame:
    """Sep 28-Oct 2 2026, engineered so weekly H/L/C == the CLAUDE.md fixture
    (H 522.85, L 502.22, C 517.53 -> R1 526.18, R2 534.83)."""
    dates = pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"])
    return pd.DataFrame(
        {
            "open": [503.0, 504.0, 508.0, 515.0, 516.0],
            "high": [506.0, 510.0, 522.85, 519.0, 522.50],
            "low": [502.22, 503.0, 507.0, 512.0, 513.66],
            "close": [504.0, 508.0, 515.0, 516.0, 517.53],
            "volume": [1_000_000] * 5,
        },
        index=dates,
    )


def _msft_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [520.0, 525.0, 530.0, 535.0],
            "bid": [7.80, 5.20, 3.10, 1.60],
            "ask": [8.00, 5.40, 3.30, 1.80],
            "delta": [0.35, 0.28, 0.22, 0.15],
        }
    )


def _msft_bars_with_history() -> pd.DataFrame:
    """20 quiet prior business days (H well below the fixture week) so
    prior_20d_high is non-NaN and a breakout is unambiguous."""
    history_dates = pd.bdate_range(end="2026-09-25", periods=20)
    history = pd.DataFrame(
        {"open": [100.0] * 20, "high": [100.0] * 20, "low": [95.0] * 20, "close": [98.0] * 20, "volume": [500_000] * 20},
        index=history_dates,
    )
    return pd.concat([history, _msft_week_bars()])


FRIDAY_AFTER_CLOSE = datetime(2026, 10, 2, 17, 0, tzinfo=NY)


def test_recommends_weekly_r1_strike_with_target_delta():
    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=200, cost=408.84)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},  # not breaking out (< prior high)
        chains={"MSFT": _msft_chain()},
    )
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "recommend"
    assert rec.free_lots == 2
    assert rec.strike == 530.0  # first strike >= R1 (526.18) with delta in [0.20, 0.30]
    assert rec.delta == pytest.approx(0.22)
    assert rec.bid == pytest.approx(3.10)
    assert rec.premium_per_lot == pytest.approx(310.0)
    assert rec.total_premium == pytest.approx(620.0)
    assert rec.liquidity_ok is True
    assert "Taxable" in rec.tax_note


def test_breakout_uses_weekly_r2_floor():
    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=408.84)])
    source = FakeDataSource(
        bars={"MSFT": _msft_bars_with_history()},
        spots={"MSFT": 550.0},  # clears every historical high -> breaking out
        chains={"MSFT": _msft_chain()},
    )
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "recommend"
    assert rec.strike == 535.0  # first strike >= R2 (534.83)
    assert any("breaking out" in n for n in rec.notes)


def test_never_strikes_below_cost_basis():
    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=528.00)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},
        chains={"MSFT": _msft_chain()},
    )
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "recommend"
    assert rec.strike >= 528.00
    assert any("cost basis" in n for n in rec.notes)


def test_no_covered_calls_override_is_skipped():
    holdings = Holdings(
        lots=[Lot(symbol="CRWV", account="rh_taxable", shares=501, cost=89.27, no_covered_calls=True)]
    )
    source = FakeDataSource(bars={}, spots={}, chains={})
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "skip"
    assert "no_covered_calls" in rec.reason


def test_fully_covered_position_is_skipped():
    holdings = Holdings(
        lots=[Lot(symbol="OSCR", account="rh_taxable", shares=1000, cost=24.07)],
        short_options=[
            ShortOption(symbol="OSCR", type="call", strike=31.0, expiry=datetime(2026, 10, 9).date(), qty=10)
        ],
    )
    source = FakeDataSource(bars={}, spots={}, chains={})
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "skip"
    assert rec.free_lots == 0
    assert "free lots" in rec.reason


def test_ira_account_has_no_tax_note():
    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_ira", shares=100, cost=400.0)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},
        chains={"MSFT": _msft_chain()},
    )
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.tax_note == "IRA — no tax effect if assigned"


# --- ALAB wheel override ---------------------------------------------------

def _alab_week_bars() -> pd.DataFrame:
    """A made-up Sep 28-Oct 2 2026 week with weekly H 380 / L 360 / C 370
    -> P 370, R2 390 (P + (H-L))."""
    dates = pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"])
    return pd.DataFrame(
        {
            "open": [363.0, 365.0, 370.0, 372.0, 368.0],
            "high": [365.0, 370.0, 380.0, 375.0, 372.0],
            "low": [360.0, 361.0, 366.0, 368.0, 365.0],
            "close": [362.0, 366.0, 372.0, 371.0, 370.0],
            "volume": [500_000] * 5,
        },
        index=dates,
    )


def _alab_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [375.0, 380.0, 385.0, 390.0, 395.0],
            "bid": [4.00, 3.00, 2.20, 1.60, 1.10],
            "ask": [4.20, 3.20, 2.40, 1.80, 1.30],
            "delta": [0.40, 0.33, 0.27, 0.21, 0.16],
        }
    )


def _alab_holdings() -> Holdings:
    return Holdings(
        lots=[Lot(symbol="ALAB", account="rh_taxable", shares=100, cost=242.45, strategy="wheel")],
        alab_rules={"atm_call_trigger": 370.0, "fallback_strike_above": "weekly_R2", "fallback_deadline": "tuesday_close"},
    )


def test_alab_wheel_waits_if_trigger_not_touched_monday():
    now = datetime(2026, 10, 5, 9, 0, tzinfo=NY)  # Monday, before any touch
    source = FakeDataSource(
        bars={"ALAB": _alab_week_bars()},
        spots={"ALAB": 360.0},  # below the 370 trigger
        chains={"ALAB": _alab_chain()},
    )
    [rec] = recommend_covered_calls(_alab_holdings(), source, now=now)
    assert rec.status == "wait"
    assert "trigger" in rec.reason


def test_alab_wheel_sells_atm_when_trigger_touched():
    now = datetime(2026, 10, 6, 9, 0, tzinfo=NY)  # Tuesday, trigger touched
    source = FakeDataSource(
        bars={"ALAB": _alab_week_bars()},
        spots={"ALAB": 372.0},  # >= 370 trigger
        chains={"ALAB": _alab_chain()},
    )
    [rec] = recommend_covered_calls(_alab_holdings(), source, now=now)
    assert rec.status == "recommend"
    assert rec.strike == 375.0  # nearest strike to the 370 trigger
    assert any("trigger" in n and "touched" in n for n in rec.notes)


def test_alab_wheel_falls_back_to_above_r2_after_tuesday():
    now = datetime(2026, 10, 7, 9, 0, tzinfo=NY)  # Wednesday: past the deadline
    source = FakeDataSource(
        bars={"ALAB": _alab_week_bars()},
        spots={"ALAB": 360.0},  # never touched the trigger
        chains={"ALAB": _alab_chain()},
    )
    [rec] = recommend_covered_calls(_alab_holdings(), source, now=now)
    assert rec.status == "recommend"
    assert rec.strike == 390.0  # first strike >= weekly R2 (390)
    assert any("fallback" in n for n in rec.notes)


# --- earnings guard ----------------------------------------------------------

class FakeEvents:
    def __init__(self, earnings_dates):
        self._dates = earnings_dates

    def earnings(self, symbol, start, end):
        from tool.data.earnings import EarningsEvent

        return [EarningsEvent(symbol=symbol, date=d, hour=None) for d in self._dates if start <= d <= end]


def test_expiry_with_earnings_inside_is_skipped():
    from datetime import date

    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=408.84)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},
        chains={"MSFT": _msft_chain()},
    )
    events = FakeEvents([date(2026, 10, 7)])  # before Fri Oct 9 expiry
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE, events=events)
    assert rec.status == "skip"
    assert "earnings 2026-10-07" in rec.reason


def test_earnings_after_expiry_does_not_block_and_is_noted():
    from datetime import date

    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=408.84)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},
        chains={"MSFT": _msft_chain()},
    )
    events = FakeEvents([date(2026, 10, 27)])
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE, events=events)
    assert rec.status == "recommend"
    assert any("no earnings before" in n for n in rec.notes)


def test_missing_events_source_is_marked_unverified():
    holdings = Holdings(lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=408.84)])
    source = FakeDataSource(
        bars={"MSFT": _msft_week_bars()},
        spots={"MSFT": 517.53},
        chains={"MSFT": _msft_chain()},
    )
    [rec] = recommend_covered_calls(holdings, source, now=FRIDAY_AFTER_CLOSE)
    assert rec.status == "recommend"
    assert any("earnings unverified" in n for n in rec.notes)

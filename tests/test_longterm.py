from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tool.levels import NY
from tool.longterm import quality_checks, recommend_long_term, trend_checks

NOW = datetime(2026, 10, 9, 17, 0, tzinfo=NY)

GOOD = {
    "revChangeTTM": 18.0, "operatingMarginTTM": 40.0, "returnOnEquity": 34.0, "epsChangePercentTTM": 31.0,
    "totalDebtToEquity": 25.0, "peRatio": 29.0, "pegRatio": 0.9, "high52": 150.0,
}


def bars(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(end="2026-10-09", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes, "volume": [1e6] * len(closes)},
        index=dates,
    )


def uptrend(end=120.0, n=260):
    return bars(np.linspace(80.0, end, n))


def test_quality_checks_pass_for_a_strong_company_and_fail_for_a_loss_maker():
    assert all(ok for _, ok in quality_checks(GOOD))
    bad = {**GOOD, "operatingMarginTTM": -5.0, "returnOnEquity": -3.0, "epsChangePercentTTM": -20.0, "peRatio": -40.0, "pegRatio": -1.0}
    assert sum(ok for _, ok in quality_checks(bad)) == 2  # only revenue growth and debt pass


def test_absurd_pe_for_tiny_earnings_fails_the_valuation_check():
    checks = dict(quality_checks({**GOOD, "peRatio": 7135.0, "pegRatio": -1.8}))
    assert checks["sane valuation (P/E 0-40 or PEG 0-2)"] is False


def test_trend_checks():
    assert [ok for _, ok in trend_checks(100, 95, 90, 110)] == [True, True, True]
    assert [ok for _, ok in trend_checks(80, 95, 90, 120)] == [False, True, False]


def test_recommend_in_buy_zone_with_stop_and_targets():
    # steady uptrend ending 4% above its 50-day average -> not extended
    rec = recommend_long_term("TEST", uptrend(), GOOD, now=NOW)
    assert rec.status == "recommend" and rec.hold_months == 12
    assert rec.entry <= rec.spot and rec.stop < rec.entry < rec.t1 < rec.t2
    assert rec.stop >= rec.entry * 0.88 - 1e-9  # never more than 12% below entry
    assert rec.t1 == pytest.approx(150.0) and rec.t2 == pytest.approx(150.0 + (150.0 - rec.entry))
    assert rec.quality == 6 and rec.trend == 3 and "quality 6/6" in rec.reason


def test_extended_stock_waits_for_a_pullback_to_the_50_day():
    closes = list(np.linspace(80, 100, 255)) + [112, 118, 124, 130, 136]
    rec = recommend_long_term("TEST", bars(closes), {**GOOD, "high52": 140.0}, now=NOW)
    assert rec.status == "wait" and "wait for a pullback" in rec.reason
    sma50 = float(pd.Series(closes).tail(50).mean())
    assert rec.entry == pytest.approx(sma50) and rec.entry < rec.spot


def test_downtrend_fails_even_with_good_fundamentals():
    rec = recommend_long_term("TEST", bars(np.linspace(150, 90, 260)), GOOD, now=NOW)
    assert rec.status == "skip" and "above the 200-day average" in rec.reason


def test_weak_fundamentals_fail_even_in_an_uptrend():
    weak = {**GOOD, "revChangeTTM": 2.0, "operatingMarginTTM": 1.0, "returnOnEquity": 1.0, "epsChangePercentTTM": -5.0}
    rec = recommend_long_term("TEST", uptrend(), weak, now=NOW)
    assert rec.status == "skip" and rec.quality == 2


def test_short_history_and_missing_fundamentals_are_skipped_with_a_reason():
    assert "needs 200 daily bars" in recommend_long_term("TEST", bars(np.linspace(80, 100, 90)), GOOD, now=NOW).reason
    assert "no fundamentals" in recommend_long_term("TEST", uptrend(), None, now=NOW).reason

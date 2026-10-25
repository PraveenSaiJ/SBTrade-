"""Tests for tool.levels — verified against the MSFT fixtures in CLAUDE.md.

Daily  (Oct 2 2026 bar): H 522.50, L 513.66, C 517.53
  -> P 517.90, R1 522.13, S1 513.29, R2 526.74, S2 509.06
Weekly (Sep 28-Oct 2 2026): H 522.85, L 502.22, C 517.53
  -> P 514.20, R1 526.18, S1 505.55, R2 534.83, S2 493.57
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from tool.data.base import DataSource
from tool.data.ibkr import IBKRDataSource
from tool.data.schwab import SchwabDataSource
from tool.levels import (
    NY,
    atr,
    classic_pivots,
    compute_levels,
    last_completed_daily_bar,
    last_completed_weekly_bar,
    prior_n_day_high,
    sma,
)


def test_classic_pivots_daily_fixture():
    piv = classic_pivots(high=522.50, low=513.66, close=517.53)
    assert round(piv.p, 2) == 517.90
    assert round(piv.r1, 2) == 522.13
    assert round(piv.s1, 2) == 513.29
    assert round(piv.r2, 2) == 526.74
    assert round(piv.s2, 2) == 509.06


def test_classic_pivots_weekly_fixture():
    piv = classic_pivots(high=522.85, low=502.22, close=517.53)
    assert round(piv.p, 2) == 514.20
    assert round(piv.r1, 2) == 526.18
    assert round(piv.s1, 2) == 505.55
    assert round(piv.r2, 2) == 534.83
    assert round(piv.s2, 2) == 493.57


def _msft_week_bars() -> pd.DataFrame:
    """Sep 28-Oct 2 2026 (Mon-Fri), engineered so daily == Oct 2 fixture and
    weekly H/L/C == the Sep28-Oct2 fixture."""
    dates = pd.to_datetime(
        ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
    )
    data = {
        "open": [503.0, 504.0, 508.0, 515.0, 516.0],
        "high": [506.0, 510.0, 522.85, 519.0, 522.50],
        "low": [502.22, 503.0, 507.0, 512.0, 513.66],
        "close": [504.0, 508.0, 515.0, 516.0, 517.53],
        "volume": [1_000_000] * 5,
    }
    return pd.DataFrame(data, index=dates)


def test_last_completed_daily_bar_after_close_uses_todays_bar():
    bars = _msft_week_bars()
    now = datetime(2026, 10, 2, 17, 0, tzinfo=NY)  # Friday, after 16:15 close
    row = last_completed_daily_bar(bars, now=now)
    assert row.name == pd.Timestamp("2026-10-02")
    assert row["high"] == 522.50
    assert row["low"] == 513.66
    assert row["close"] == 517.53


def test_last_completed_daily_bar_before_close_uses_prior_day():
    bars = _msft_week_bars()
    now = datetime(2026, 10, 2, 10, 0, tzinfo=NY)  # Friday, before close
    row = last_completed_daily_bar(bars, now=now)
    assert row.name == pd.Timestamp("2026-10-01")


def test_weekly_bar_friday_afternoon_uses_this_week():
    bars = _msft_week_bars()
    now = datetime(2026, 10, 2, 17, 0, tzinfo=NY)  # Friday afternoon run
    weekly = last_completed_weekly_bar(bars, now=now)
    assert weekly.week_start == pd.Timestamp("2026-09-28").date()
    assert weekly.week_end == pd.Timestamp("2026-10-02").date()
    assert weekly.high == 522.85
    assert weekly.low == 502.22
    assert weekly.close == 517.53
    assert round(weekly.pivots.p, 2) == 514.20


def test_weekly_bar_midweek_uses_prior_week():
    # Extend with the following Tuesday, mid-week: current week isn't done yet.
    bars = _msft_week_bars()
    extra = pd.DataFrame(
        {"open": [518.0], "high": [520.0], "low": [516.0], "close": [519.0], "volume": [1_000_000]},
        index=pd.to_datetime(["2026-10-06"]),  # Tuesday
    )
    bars = pd.concat([bars, extra])
    now = datetime(2026, 10, 6, 12, 0, tzinfo=NY)  # Tuesday midday
    weekly = last_completed_weekly_bar(bars, now=now)
    # Should still resolve to the prior (Sep28-Oct2) completed week, not the
    # in-progress current week.
    assert weekly.week_start == pd.Timestamp("2026-09-28").date()
    assert weekly.week_end == pd.Timestamp("2026-10-02").date()
    assert weekly.close == 517.53


def test_weekly_bar_monday_morning_uses_prior_week():
    bars = _msft_week_bars()
    now = datetime(2026, 10, 5, 9, 0, tzinfo=NY)  # Monday morning run
    weekly = last_completed_weekly_bar(bars, now=now)
    assert weekly.week_start == pd.Timestamp("2026-09-28").date()
    assert weekly.week_end == pd.Timestamp("2026-10-02").date()


def _flat_atr_bars() -> pd.DataFrame:
    """26 business days: 25 flat bars (H102/L98/C100, TR=4 throughout) then one
    gapped-up bar (H150/L140/C145) to exercise ATR, SMA, and the 20D-high
    exclusion of "today" with hand-computable expected values."""
    dates = pd.bdate_range("2026-01-05", periods=26)
    highs = [102.0] * 25 + [150.0]
    lows = [98.0] * 25 + [140.0]
    closes = [100.0] * 25 + [145.0]
    opens = closes
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1_000_000] * 26},
        index=dates,
    )


def test_atr_sma_prior_high_deterministic():
    bars = _flat_atr_bars()
    last_date = bars.index[-1]
    now = datetime(last_date.year, last_date.month, last_date.day, 17, 0, tzinfo=NY)

    atr10 = atr(bars, 10).iloc[-1]
    sma20 = sma(bars["close"], 20).iloc[-1]
    prior_high = prior_n_day_high(bars, 20).iloc[-1]

    assert atr10 == pytest.approx(8.6, abs=1e-9)
    assert sma20 == pytest.approx(102.25, abs=1e-9)
    assert prior_high == pytest.approx(102.0, abs=1e-9)

    result = compute_levels(bars, now=now, symbol="TEST")
    assert result.atr10 == pytest.approx(8.6, abs=1e-9)
    assert result.sma20 == pytest.approx(102.25, abs=1e-9)
    assert result.prior_20d_high == pytest.approx(102.0, abs=1e-9)


def test_data_sources_satisfy_protocol():
    assert isinstance(IBKRDataSource(), DataSource)
    assert hasattr(SchwabDataSource, "daily_bars")
    assert hasattr(SchwabDataSource, "spot")


def test_schwab_retry_recovers_from_transient_network_errors(monkeypatch):
    import httpx

    from tool.data import schwab

    monkeypatch.setattr(schwab.time, "sleep", lambda s: None)
    calls = {"n": 0}

    @schwab._retry
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("getaddrinfo failed")
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3


def test_schwab_retry_gives_up_after_four_attempts(monkeypatch):
    import httpx

    from tool.data import schwab

    monkeypatch.setattr(schwab.time, "sleep", lambda s: None)
    calls = {"n": 0}

    @schwab._retry
    def always_down():
        calls["n"] += 1
        raise httpx.ConnectError("getaddrinfo failed")

    with pytest.raises(httpx.ConnectError):
        always_down()
    assert calls["n"] == 4


def test_schwab_retry_backs_off_on_rate_limit_but_not_other_http_errors(monkeypatch):
    import httpx

    from tool.data import schwab

    monkeypatch.setattr(schwab.time, "sleep", lambda s: None)

    def status_error(code):
        request = httpx.Request("GET", "https://example.test")
        return httpx.HTTPStatusError("err", request=request, response=httpx.Response(code, request=request))

    calls = {"n": 0}

    @schwab._retry
    def rate_limited_twice():
        calls["n"] += 1
        if calls["n"] < 3:
            raise status_error(429)
        return "ok"

    assert rate_limited_twice() == "ok" and calls["n"] == 3

    calls["n"] = 0

    @schwab._retry
    def not_found():
        calls["n"] += 1
        raise status_error(404)

    with pytest.raises(httpx.HTTPStatusError):
        not_found()
    assert calls["n"] == 1  # a 404 is not retried

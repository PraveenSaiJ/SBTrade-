"""Pivot levels, completed-bar selection, ATR/SMA/20-day-high.

All "now" handling is in America/New_York. A daily bar is considered complete
once it's past 16:15 ET on its date, or its date is before today (NY). Weekly
levels use the last completed Monday-Friday week: H = week max, L = week min,
C = last close in the week. See CLAUDE.md "Level math" for the fixtures this
module is verified against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tool.indicators import atr, sma  # noqa: F401  (re-exported: callers import them from here)

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 15)


@dataclass(frozen=True)
class Pivots:
    p: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


@dataclass(frozen=True)
class DailyLevels:
    bar_date: date
    high: float
    low: float
    close: float
    pivots: Pivots


@dataclass(frozen=True)
class WeeklyLevels:
    week_start: date
    week_end: date
    high: float
    low: float
    close: float
    pivots: Pivots


@dataclass(frozen=True)
class LevelsResult:
    daily: DailyLevels
    weekly: WeeklyLevels
    atr10: float
    sma20: float
    prior_20d_high: float
    symbol: str | None = None
    spot: float | None = None


def classic_pivots(high: float, low: float, close: float) -> Pivots:
    """Classic floor pivots from a completed bar's (H, L, C)."""
    p = (high + low + close) / 3
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return Pivots(p=p, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)


def _now_ny(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(NY)
    if now.tzinfo is None:
        return now.replace(tzinfo=NY)
    return now.astimezone(NY)


def _as_ny_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """Normalize a bar DataFrame's index to plain America/New_York dates.

    Daily bars are assumed to carry no meaningful intraday time component when
    the index is tz-naive (the date itself is the NY trading date).
    """
    if index.tz is not None:
        index = index.tz_convert(NY)
    return index.date


def _daily_cutoff_date(now: datetime | None = None) -> date:
    """Last calendar date whose daily bar may be treated as complete."""
    n = _now_ny(now)
    return n.date() if n.time() >= MARKET_CLOSE else n.date() - timedelta(days=1)


def completed_daily_bars(daily: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    cutoff = _daily_cutoff_date(now)
    dates = _as_ny_dates(daily.index)
    completed = daily.loc[dates <= cutoff]
    if completed.empty:
        raise ValueError("No completed daily bar available on or before the cutoff date")
    return completed


def last_completed_daily_bar(daily: pd.DataFrame, now: datetime | None = None) -> pd.Series:
    """Return the OHLC row for the last completed daily bar."""
    return completed_daily_bars(daily, now).iloc[-1]


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def last_completed_week_range(now: datetime | None = None) -> tuple[date, date]:
    """Return (monday, friday) of the last completed Mon-Fri week."""
    cutoff = _daily_cutoff_date(now)
    monday = _week_start(cutoff)
    friday = monday + timedelta(days=4)
    if friday > cutoff:
        monday -= timedelta(days=7)
        friday -= timedelta(days=7)
    return monday, friday


def last_completed_weekly_bar(daily: pd.DataFrame, now: datetime | None = None) -> WeeklyLevels:
    """Return H/L/C + pivots for the last completed Mon-Fri week."""
    monday, friday = last_completed_week_range(now)
    dates = _as_ny_dates(daily.index)
    mask = (dates >= monday) & (dates <= friday)
    week_bars = daily.loc[mask]
    if week_bars.empty:
        raise ValueError(f"No daily bars found for completed week {monday}..{friday}")
    high = float(week_bars["high"].max())
    low = float(week_bars["low"].min())
    close = float(week_bars["close"].iloc[-1])
    return WeeklyLevels(
        week_start=monday,
        week_end=friday,
        high=high,
        low=low,
        close=close,
        pivots=classic_pivots(high, low, close),
    )


def prior_n_day_high(daily: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling `period`-day high of prior bars, excluding the current bar."""
    return daily["high"].shift(1).rolling(window=period).max()


def compute_levels(
    daily: pd.DataFrame,
    now: datetime | None = None,
    spot: float | None = None,
    symbol: str | None = None,
) -> LevelsResult:
    """Compute daily + weekly pivot levels and swing-screener inputs."""
    completed = completed_daily_bars(daily, now)
    daily_row = completed.iloc[-1]
    daily_levels = DailyLevels(
        bar_date=daily_row.name.date() if hasattr(daily_row.name, "date") else daily_row.name,
        high=float(daily_row["high"]),
        low=float(daily_row["low"]),
        close=float(daily_row["close"]),
        pivots=classic_pivots(float(daily_row["high"]), float(daily_row["low"]), float(daily_row["close"])),
    )
    weekly_levels = last_completed_weekly_bar(daily, now)
    atr10 = float(atr(completed, 10).iloc[-1])
    sma20 = float(sma(completed["close"], 20).iloc[-1])
    prior_high = float(prior_n_day_high(completed, 20).iloc[-1])
    return LevelsResult(
        daily=daily_levels,
        weekly=weekly_levels,
        atr10=atr10,
        sma20=sma20,
        prior_20d_high=prior_high,
        symbol=symbol,
        spot=spot,
    )

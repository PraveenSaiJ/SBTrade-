"""Long-term stock recommendations: quality (Schwab fundamentals) plus trend (daily bars).

Recommend-only. Rules, all visible in the Reason text:
- Quality: at least 4 of 6 -- revenue growth >= 10%, operating margin >= 10%, ROE >= 10%, EPS growth >= 10%,
  debt/equity <= 100%, and a sane valuation (P/E between 0 and 40, or PEG between 0 and 2).
- Trend (all required): close above the 200-day average and the 50-day average above the 200-day.
- Entry: the 50-day average (buy pullbacks). Over 10% above it the idea is "wait" for a pullback to that level.
- Stop: the 200-day average, but never more than 12% below the entry.
- T1: the 52-week high (or +20% if the high is under +10%). T2: the same distance again beyond T1.
- Minimum hold: 12 months, the long-term capital gains threshold in a taxable account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from tool.indicators import sma
from tool.levels import completed_daily_bars
from tool.config import get_config

_L = get_config().longterm
MIN_QUALITY_PASSES = _L.min_quality_passes
EXTENDED_OVER_SMA50 = _L.extended_over_sma50
MAX_STOP_DISTANCE = _L.max_stop_distance
HOLD_MONTHS = _L.hold_months
MIN_BARS = _L.min_bars


@dataclass(frozen=True)
class LongTermRec:
    symbol: str
    status: str  # "recommend" | "wait" | "skip"
    reason: str
    spot: float | None = None
    entry: float | None = None
    hold_months: int = HOLD_MONTHS
    stop: float | None = None
    t1: float | None = None
    t2: float | None = None
    quality: int | None = None  # out of 6
    trend: int | None = None  # out of 3
    notes: list[str] = field(default_factory=list)


def _num(f: dict, key: str) -> float | None:
    value = f.get(key)
    return float(value) if isinstance(value, (int, float)) and not pd.isna(value) else None


def quality_checks(f: dict) -> list[tuple[str, bool]]:
    rev, op, roe = _num(f, "revChangeTTM"), _num(f, "operatingMarginTTM"), _num(f, "returnOnEquity")
    eps_growth, debt = _num(f, "epsChangePercentTTM"), _num(f, "totalDebtToEquity")
    pe, peg = _num(f, "peRatio"), _num(f, "pegRatio")
    return [
        ("revenue growth >= 10%", rev is not None and rev >= 10),
        ("operating margin >= 10%", op is not None and op >= 10),
        ("ROE >= 10%", roe is not None and roe >= 10),
        ("EPS growth >= 10%", eps_growth is not None and eps_growth >= 10),
        ("debt/equity <= 100%", debt is not None and debt <= 100),
        ("sane valuation (P/E 0-40 or PEG 0-2)", (pe is not None and 0 < pe <= 40) or (peg is not None and 0 < peg <= 2)),
    ]


def trend_checks(spot: float, sma50: float, sma200: float, high52: float) -> list[tuple[str, bool]]:
    return [
        ("above the 200-day average", spot > sma200),
        ("50-day above the 200-day", sma50 > sma200),
        ("within 25% of the 52-week high", spot >= 0.75 * high52),
    ]


def recommend_long_term(
    symbol: str, bars: pd.DataFrame, fundamentals: dict | None, now: datetime | None = None
) -> LongTermRec:
    if not fundamentals:
        return LongTermRec(symbol, "skip", "no fundamentals available from Schwab")
    try:
        closes = completed_daily_bars(bars, now)["close"]
    except ValueError as exc:
        return LongTermRec(symbol, "skip", str(exc))
    if len(closes) < MIN_BARS:
        return LongTermRec(symbol, "skip", f"needs {MIN_BARS} daily bars, has {len(closes)}")

    spot = float(closes.iloc[-1])
    sma50 = float(sma(closes, 50).iloc[-1])
    sma200 = float(sma(closes, 200).iloc[-1])
    high52 = _num(fundamentals, "high52") or float(closes.tail(252).max())

    quality = quality_checks(fundamentals)
    trend = trend_checks(spot, sma50, sma200, high52)
    q_pass, t_pass = sum(ok for _, ok in quality), sum(ok for _, ok in trend)
    failed = [name for name, ok in quality + trend if not ok]
    summary = f"quality {q_pass}/6, trend {t_pass}/3"

    if q_pass < MIN_QUALITY_PASSES or not (trend[0][1] and trend[1][1]):
        return LongTermRec(
            symbol, "skip", f"{summary}; fails: {', '.join(failed)}", spot=spot, quality=q_pass, trend=t_pass
        )

    extended = spot > sma50 * (1 + EXTENDED_OVER_SMA50)
    entry = sma50 if extended else min(spot, sma50)
    stop = max(sma200, entry * (1 - MAX_STOP_DISTANCE))
    t1 = high52 if high52 >= entry * 1.10 else entry * 1.20
    t2 = t1 + (t1 - entry)
    passed = ", ".join(name for name, ok in quality + trend if ok)
    notes = [f"fails: {', '.join(failed)}"] if failed else []
    if extended:
        reason = f"{summary}; {spot / sma50 - 1:.0%} above the 50-day, so wait for a pullback to ${entry:.2f}. Passes: {passed}"
        return LongTermRec(symbol, "wait", reason, spot, entry, HOLD_MONTHS, stop, t1, t2, q_pass, t_pass, notes)
    return LongTermRec(
        symbol, "recommend", f"{summary}. Passes: {passed}", spot, entry, HOLD_MONTHS, stop, t1, t2, q_pass, t_pass, notes
    )

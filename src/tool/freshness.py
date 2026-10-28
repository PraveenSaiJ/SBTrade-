"""How old is a section's data, and is it stale? (CLAUDE.md: every recommendation shows its data timestamp.)

Kinds:
- intraday: quotes that move; stale if older than 30 minutes while the market is open.
- hourly:   refreshed a few times a day; stale if older than 2 hours while the market is open.
- daily:    refreshed once a day; stale only if it predates the latest trading session.
Every kind is also stale when it predates the latest trading session (holidays are not modelled).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from tool.levels import NY
from tool.market_calendar import is_trading_day, previous_trading_day
from tool.config import get_config

_F = get_config().freshness


def _clock(text: str) -> time:
    hour, minute = text.split(":")
    return time(int(hour), int(minute))


MARKET_OPEN = _clock(_F.market_open)
MARKET_CLOSE = _clock(_F.market_close)
MAX_AGE = {"intraday": timedelta(minutes=_F.intraday_max_minutes), "hourly": timedelta(hours=_F.hourly_max_hours)}


def previous_weekday(day: date) -> date:
    """The previous trading day (weekends and NYSE holidays skipped)."""
    return previous_trading_day(day)


def market_is_open(now: datetime) -> bool:
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MARKET_CLOSE


def last_session_date(now: datetime) -> date:
    """The most recent trading day that has started: today once the market opens, else the prior weekday."""
    if not is_trading_day(now.date()):
        return previous_weekday(now.date() + timedelta(days=1))
    return now.date() if now.time() >= MARKET_OPEN else previous_weekday(now.date())


MORNING_GRACE_END = _clock(_F.morning_grace_end)  # the morning jobs (buys 9:40, options 10:00) may still be running


def in_morning_window(now: datetime) -> bool:
    """Market open but the first scheduled runs of the day have not all finished yet."""
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MORNING_GRACE_END


def humanize(age: timedelta) -> str:
    minutes = int(age.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} h ago"
    return f"{hours // 24} days ago"


def parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    return ts.astimezone(NY) if ts.tzinfo else ts.replace(tzinfo=NY)


def assess(ts: datetime, now: datetime, kind: str) -> tuple[bool, str]:
    """Return (is_stale, 'N min ago')."""
    stale = ts.astimezone(NY).date() < last_session_date(now)
    limit = MAX_AGE.get(kind)
    if limit is not None and market_is_open(now) and now - ts > limit:
        stale = True
    return stale, humanize(now - ts)

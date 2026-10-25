"""Expiry selection shared by the option engines: weekly, monthly, and earnings-driven."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from tool.covered_calls import next_weekly_expiry
from tool.data.earnings import EventsUnavailable, FinnhubEvents

import httpx


def monthly_expiry(today: date) -> date:
    """Third Friday of this month if it is at least 7 days out, else next month's."""

    def third_friday(year: int, month: int) -> date:
        first = date(year, month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        return first_friday + timedelta(days=14)

    candidate = third_friday(today.year, today.month)
    if candidate - today >= timedelta(days=7):
        return candidate
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return third_friday(year, month)


def expiry_candidates(
    symbol: str,
    now: datetime,
    events: FinnhubEvents | None,
) -> list[tuple[str, date]]:
    """[(type, expiry)] with type in {weekly, monthly, earnings}. Earnings only when a date is found."""
    today = now.date()
    out = [("weekly", next_weekly_expiry(now)), ("monthly", monthly_expiry(today))]
    if events is None:
        return out
    try:
        upcoming = events.earnings(symbol, today, today + timedelta(days=45))
    except (EventsUnavailable, httpx.HTTPError):
        return out
    if upcoming:
        first = min(e.date for e in upcoming)
        friday = first + timedelta(days=(4 - first.weekday()) % 7)
        if friday > today and friday not in {d for _, d in out}:
            out.append(("earnings", friday))
    return out

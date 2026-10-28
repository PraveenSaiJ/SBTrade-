"""NYSE trading calendar: full-day holidays by rule (with weekend observance) and early closes.

`is_trading_day` is what scheduled jobs and the freshness checks use so a holiday is not read as a missing or stale session.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date) -> date | None:
    """A holiday on Saturday is observed Friday; on Sunday, Monday. (New Year's on Saturday is not observed at all.)"""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def holidays(year: int) -> dict[date, str]:
    """Full-day NYSE closures falling in `year`."""
    out: dict[date, str] = {}

    def add(day: date | None, name: str) -> None:
        if day is not None and day.year == year:
            out[day] = name

    new_year = date(year, 1, 1)
    add(None if new_year.weekday() == 5 else _observed(new_year), "New Year's Day")
    add(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    add(_nth_weekday(year, 2, 0, 3), "Presidents' Day")
    add(_easter(year) - timedelta(days=2), "Good Friday")
    add(_last_weekday(year, 5, 0), "Memorial Day")
    if year >= 2022:
        add(_observed(date(year, 6, 19)), "Juneteenth")
    add(_observed(date(year, 7, 4)), "Independence Day")
    add(_nth_weekday(year, 9, 0, 1), "Labor Day")
    add(_nth_weekday(year, 11, 3, 4), "Thanksgiving Day")
    add(_observed(date(year, 12, 25)), "Christmas Day")
    return out


def holiday_name(day: date) -> str | None:
    # a New Year's Day observed on Dec 31 belongs to next year's list; the rule above never creates that case (Saturday is skipped)
    return holidays(day.year).get(day)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and holiday_name(day) is None


def previous_trading_day(day: date) -> date:
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def next_trading_day(day: date) -> date:
    day += timedelta(days=1)
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day


def early_close(day: date) -> str | None:
    """Name of the 1:00 PM early close, if `day` is one (day after Thanksgiving, Christmas Eve, July 3 on a weekday)."""
    if not is_trading_day(day):
        return None
    if day == _nth_weekday(day.year, 11, 3, 4) + timedelta(days=1):
        return "Day after Thanksgiving"
    if day.month == 12 and day.day == 24:
        return "Christmas Eve"
    if day.month == 7 and day.day == 3 and is_trading_day(date(day.year, 7, 3)) and date(day.year, 7, 4).weekday() < 5:
        return "Day before Independence Day"
    return None


def next_holiday(day: date) -> tuple[date, str] | None:
    """The first full-day closure after `day` (within the next year)."""
    for offset in range(1, 400):
        d = day + timedelta(days=offset)
        name = holiday_name(d)
        if name:
            return d, name
    return None

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from tool.freshness import assess, humanize, last_session_date, market_is_open, parse_ts
from tool.levels import NY


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


TUESDAY_3PM = at(2026, 10, 13, 15, 27)


def test_last_session_date_rolls_back_before_the_open_and_over_weekends():
    assert last_session_date(at(2026, 10, 13, 10)) == date(2026, 10, 13)  # Tuesday, market open
    assert last_session_date(at(2026, 10, 13, 8)) == date(2026, 10, 12)  # Tuesday before the open -> Monday
    assert last_session_date(at(2026, 10, 12, 8)) == date(2026, 10, 9)  # Monday before the open -> Friday
    assert last_session_date(at(2026, 10, 10, 12)) == date(2026, 10, 9)  # Saturday -> Friday
    assert last_session_date(at(2026, 10, 11, 12)) == date(2026, 10, 9)  # Sunday -> Friday


def test_market_hours():
    assert market_is_open(at(2026, 10, 13, 9, 30)) and not market_is_open(at(2026, 10, 13, 16, 0))
    assert not market_is_open(at(2026, 10, 10, 11))  # Saturday


def test_friday_data_viewed_on_tuesday_is_stale_for_every_kind():
    friday = at(2026, 10, 9, 14, 19)
    for kind in ("intraday", "hourly", "daily"):
        stale, age = assess(friday, TUESDAY_3PM, kind)
        assert stale and "days ago" in age


def test_intraday_data_goes_stale_after_30_minutes_while_the_market_is_open():
    assert assess(TUESDAY_3PM - timedelta(minutes=20), TUESDAY_3PM, "intraday")[0] is False
    assert assess(TUESDAY_3PM - timedelta(minutes=45), TUESDAY_3PM, "intraday")[0] is True


def test_daily_data_from_this_morning_is_fresh_all_day():
    morning = at(2026, 10, 13, 9, 25)
    assert assess(morning, TUESDAY_3PM, "daily") == (False, "6 h ago")
    assert assess(morning, TUESDAY_3PM, "hourly")[0] is True  # but an hourly section is overdue


def test_intraday_data_is_not_stale_after_the_close_if_it_is_from_the_last_session():
    after_close = at(2026, 10, 13, 18, 0)
    assert assess(at(2026, 10, 13, 15, 50), after_close, "intraday")[0] is False


def test_yesterdays_data_is_fine_before_the_open_but_stale_once_the_market_is_open():
    yesterday = at(2026, 10, 12, 15, 50)
    assert assess(yesterday, at(2026, 10, 13, 8, 0), "daily")[0] is False
    assert assess(yesterday, at(2026, 10, 13, 9, 45), "daily")[0] is True


@pytest.mark.parametrize(
    "age, text",
    [(timedelta(seconds=20), "just now"), (timedelta(minutes=12), "12 min ago"), (timedelta(hours=5), "5 h ago"),
     (timedelta(days=4), "4 days ago")],
)
def test_humanize(age, text):
    assert humanize(age) == text


def test_parse_ts_handles_offsets_and_naive_values():
    assert parse_ts("2026-10-13T10:10:18-04:00") == at(2026, 10, 13, 10, 10) + timedelta(seconds=18)
    assert parse_ts("2026-10-13T10:10:00").tzinfo is not None


def test_morning_window_is_open_market_before_1030():
    from tool.freshness import in_morning_window

    assert in_morning_window(at(2026, 10, 13, 9, 45)) and not in_morning_window(at(2026, 10, 13, 8, 59))
    assert not in_morning_window(at(2026, 10, 13, 10, 30)) and not in_morning_window(at(2026, 10, 10, 10))

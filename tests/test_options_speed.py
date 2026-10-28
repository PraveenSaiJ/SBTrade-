from __future__ import annotations

import threading
import time
from datetime import date, timedelta

import pandas as pd

from tool.cli import OPTION_PARTS
from tool.data.earnings import CalendarEvents, EarningsEvent
from tool.data.memo import MemoSource, clusters
from tool.data.schwab import SchwabDataSource, _chain_frame
from tool.jobs import JOB_SPECS, RUN_ALL_ORDER

TODAY = date(2026, 10, 27)
WEEKLY, MONTHLY, FAR = date(2026, 10, 30), date(2026, 11, 20), date(2026, 12, 18)


def frame(expiry: date) -> pd.DataFrame:
    return pd.DataFrame({"expiry": [expiry], "strike": [100.0], "bid": [1.0], "ask": [1.1]})


class Source:
    """Counts requests; `option_chains` returns every Friday between start and end, `option_chain` the first on/after the date."""

    def __init__(self) -> None:
        self.single, self.ranges, self.lock = 0, [], threading.Lock()

    def fridays(self, start: date, end: date) -> list[date]:
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        return [d for d in days if d.weekday() == 4]

    def option_chains(self, symbol, right, start, end, strike_count):
        with self.lock:
            self.ranges.append((symbol, right, start, end, strike_count))
        time.sleep(0.01)
        return {d: frame(d) for d in self.fridays(start, end)}

    def option_chain(self, symbol, expiry=None, right="C", window_days=21, strike_count=20):
        with self.lock:
            self.single += 1
        time.sleep(0.01)
        return frame(next(d for d in self.fridays(expiry, expiry + timedelta(days=window_days))))

    def spot(self, symbol):
        return 100.0


def test_clusters_group_nearby_expiries_and_split_far_ones():
    assert clusters([MONTHLY, WEEKLY]) == [(WEEKLY, MONTHLY)]
    assert clusters([WEEKLY, MONTHLY, FAR]) == [(WEEKLY, MONTHLY), (FAR, FAR)]  # December is over 35 days past the weekly
    assert clusters([]) == []


def test_hinted_expiries_cost_one_request_and_each_ask_gets_the_same_chain_as_a_direct_request():
    src, memo = Source(), MemoSource(Source())
    memo._source = src
    memo.hint_chains("AAA", "C", [WEEKLY, MONTHLY])
    weekly = memo.option_chain("AAA", expiry=WEEKLY, right="C")
    monthly = memo.option_chain("AAA", expiry=MONTHLY, right="C")
    assert len(src.ranges) == 1 and src.single == 0 and memo.requests == 1
    assert weekly["expiry"].iloc[0] == WEEKLY and monthly["expiry"].iloc[0] == MONTHLY
    assert src.option_chain("AAA", expiry=WEEKLY)["expiry"].iloc[0] == weekly["expiry"].iloc[0]


def test_a_hinted_symbol_that_is_never_asked_for_costs_nothing():
    src = Source()
    memo = MemoSource(src)
    memo.hint_chains("SKIPPED", "P", [WEEKLY, MONTHLY])
    assert memo.requests == 0 and src.ranges == []


def test_asks_the_range_does_not_cover_fall_back_to_their_own_request():
    src = Source()
    memo = MemoSource(src)
    memo.hint_chains("AAA", "P", [WEEKLY])
    memo.option_chain("AAA", expiry=WEEKLY, right="P")
    memo.option_chain("AAA", expiry=FAR, right="P")  # months past the fetched range
    assert len(src.ranges) == 1 and src.single == 1


def test_a_different_strike_count_is_not_served_from_the_range():
    src = Source()
    memo = MemoSource(src)
    memo.hint_chains("AAA", "C", [WEEKLY, MONTHLY])
    memo.option_chain("AAA", expiry=WEEKLY, right="C")
    memo.option_chain("AAA", expiry=WEEKLY, right="C", strike_count=40)
    assert src.single == 1


def test_a_failed_range_request_falls_back_to_the_single_asks():
    class Broken(Source):
        def option_chains(self, *a, **k):
            raise RuntimeError("429")

    src = Broken()
    memo = MemoSource(src)
    memo.hint_chains("AAA", "C", [WEEKLY])
    assert memo.option_chain("AAA", expiry=WEEKLY, right="C")["expiry"].iloc[0] == WEEKLY and src.single == 1


def test_threads_asking_for_the_same_chain_or_price_cause_one_request():
    src = Source()
    memo = MemoSource(src)
    results = []

    def ask():
        results.append(memo.option_chain("AAA", expiry=WEEKLY, right="C"))
        memo.spot("AAA")

    threads = [threading.Thread(target=ask) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 8 and src.single == 1 and memo.requests == 2


def test_the_memo_hands_out_copies_so_one_family_cannot_change_anothers_chain():
    memo = MemoSource(Source())
    first = memo.option_chain("AAA", expiry=WEEKLY, right="C")
    first.loc[0, "bid"] = 99.0
    assert memo.option_chain("AAA", expiry=WEEKLY, right="C").loc[0, "bid"] == 1.0


# ---------------------------------------------------------------- earnings asks from the calendar
class Finnhub:
    def __init__(self) -> None:
        self.calendar_calls, self.symbol_calls = 0, []

    def cached_calendar(self, today, back=14, ahead=60):
        self.calendar_calls += 1
        return [EarningsEvent("AAA", date(2026, 11, 5), "amc"), EarningsEvent("BBB", date(2026, 12, 20), None)], True

    def earnings(self, symbol, start, end):
        self.symbol_calls.append((symbol, start, end))
        return [EarningsEvent(symbol, date(2027, 2, 1), None)]


def test_earnings_asks_inside_the_calendar_range_make_no_per_symbol_request():
    api = Finnhub()
    events = CalendarEvents(api, TODAY)
    assert [e.date for e in events.earnings("AAA", TODAY, date(2026, 11, 20))] == [date(2026, 11, 5)]
    assert events.earnings("AAA", TODAY, date(2026, 11, 1)) == []  # before the report
    assert events.earnings("ZZZ", TODAY, date(2026, 11, 20)) == []
    assert [e.date for e in events.earnings("BBB", TODAY, date(2026, 12, 25))] == [date(2026, 12, 20)]  # today+60 is 12/26
    assert api.calendar_calls == 1 and api.symbol_calls == []
    events.earnings("BBB", TODAY, date(2027, 1, 5))  # past the calendar: one request for that symbol
    assert api.symbol_calls == [("BBB", TODAY, date(2027, 1, 5))]


def test_asks_beyond_the_calendar_go_to_finnhub_once_per_range():
    api = Finnhub()
    events = CalendarEvents(api, TODAY)
    far = TODAY + timedelta(days=300)
    events.earnings("AAA", TODAY, far)
    events.earnings("AAA", TODAY, far)
    assert len(api.symbol_calls) == 1


def test_a_failed_calendar_falls_back_to_asking_about_the_symbol():
    class NoCalendar(Finnhub):
        def cached_calendar(self, *a, **k):
            raise RuntimeError("429")

    api = NoCalendar()
    got = CalendarEvents(api, TODAY).earnings("AAA", TODAY, date(2026, 11, 20))
    assert got and len(api.symbol_calls) == 1


# ---------------------------------------------------------------- schwab parsing and the job split
def test_chain_frame_survives_an_expiry_with_no_contracts_and_marks_missing_data():
    assert _chain_frame({}, WEEKLY, "C").empty
    df = _chain_frame({"100.0": [{"bid": 1.0, "ask": 1.2, "delta": 2.0, "volatility": -999.0}]}, WEEKLY, "C")
    assert df.loc[0, "mid"] == 1.1 and pd.isna(df.loc[0, "delta"]) and pd.isna(df.loc[0, "iv"])


def test_option_chain_is_the_first_listed_expiry_on_or_after_the_date_from_one_range_request():
    asked = []
    source = SchwabDataSource.__new__(SchwabDataSource)
    source.option_chains = lambda symbol, right, start, end, strike_count: (
        asked.append((start, end)) or {WEEKLY: frame(WEEKLY), MONTHLY: frame(MONTHLY)})
    assert source.option_chain("AAA", expiry=date(2026, 10, 31)).loc[0, "expiry"] == MONTHLY
    assert asked == [(date(2026, 10, 31), date(2026, 11, 21))]
    try:
        source.option_chain("AAA", expiry=date(2026, 12, 1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_the_call_and_put_halves_cover_every_family_once_and_have_their_own_jobs():
    calls, puts = OPTION_PARTS["calls"], OPTION_PARTS["puts"]
    assert not set(calls) & set(puts) and len(calls) + len(puts) == 9
    assert JOB_SPECS["options_calls"]["args"] == ["options", "--part", "calls"]
    assert JOB_SPECS["options_puts"]["args"] == ["options", "--part", "puts"]
    assert "options_calls" in RUN_ALL_ORDER and "options_puts" in RUN_ALL_ORDER and "options" not in RUN_ALL_ORDER

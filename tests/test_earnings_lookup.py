from __future__ import annotations

from datetime import date

import httpx

from tool.data.earnings import EarningsEvent, EventsUnavailable, FinnhubEvents, split_reports
from tool.screener import Setup, earnings_label, flag_earnings

TODAY = date(2026, 10, 22)  # a Thursday


def ev(symbol, day, hour=None):
    return EarningsEvent(symbol, day, hour)


# ---- last / next split
def test_split_reports_picks_latest_past_and_earliest_future():
    events = [
        ev("AAA", date(2026, 10, 8), "bmo"), ev("AAA", date(2026, 10, 15), "amc"),  # two past: the later wins
        ev("AAA", date(2026, 11, 3), "bmo"), ev("AAA", date(2026, 12, 1), "amc"),  # two future: the earlier wins
        ev("BBB", date(2026, 10, 22), "bmo"),  # today, before the open: already reported
    ]
    nxt, last = split_reports(events, TODAY)
    assert last["AAA"] == (date(2026, 10, 15), "amc") and nxt["AAA"] == (date(2026, 11, 3), "bmo")
    assert last["BBB"] == (date(2026, 10, 22), "bmo") and "BBB" not in nxt


def test_a_report_today_after_the_close_is_still_ahead():
    nxt, last = split_reports([ev("CCC", TODAY, "amc")], TODAY)
    assert nxt["CCC"] == (TODAY, "amc") and "CCC" not in last


# ---- labels
def test_label_reported_within_ten_trading_days_beats_the_next_date():
    label = earnings_label(date(2026, 11, 3), "bmo", date(2026, 10, 15), "amc", True, True, TODAY)
    assert label == "Reported Oct 15 (amc)"


def test_label_upcoming_none_and_unknown():
    assert earnings_label(date(2026, 10, 28), "bmo", None, None, False, True, TODAY) == "Oct 28 (in 6d, bmo)"
    assert earnings_label(date(2026, 10, 28), None, None, None, False, True, TODAY) == "Oct 28 (in 6d)"
    assert earnings_label(None, None, None, None, False, True, TODAY) == "None"
    assert earnings_label(None, None, None, None, False, False, TODAY) == "Unknown"  # calendar did not load


def test_flag_earnings_sets_post_earnings_only_inside_ten_trading_days():
    base = Setup(symbol="AAA", kind="breakout", spot=1.0, atr=1.0, sma20=1.0, prior_high=1.0, change_20d=0.0)
    recent = flag_earnings(base, {"AAA": (date(2026, 11, 3), "bmo")}, TODAY, {"AAA": (date(2026, 10, 15), "amc")})
    assert recent.post_earnings and recent.last_hour == "amc" and recent.earnings_hour == "bmo"
    assert recent.earnings == date(2026, 11, 3)  # the next report is kept for the earnings-risk filter
    stale = flag_earnings(base, {}, TODAY, {"AAA": (date(2026, 9, 15), "amc")})
    assert not stale.post_earnings and stale.last_reported == date(2026, 9, 15)


# ---- cached fetch
def _client(requests: list):
    def handler(request: httpx.Request) -> httpx.Response:
        day = request.url.params["from"]
        requests.append((day, request.url.params["to"]))
        return httpx.Response(200, json={"earningsCalendar": [{"symbol": "S" + day[-2:], "date": day, "hour": "amc"}]})

    return httpx.Client(base_url="https://finnhub.io/api/v1", transport=httpx.MockTransport(handler))


def test_calendar_is_fetched_per_day_over_minus_14_to_plus_60_and_cached_for_the_day(tmp_path):
    requests: list = []
    api = FinnhubEvents(api_key="k", client=_client(requests))
    path = tmp_path / "cal.json"
    events, from_cache = api.cached_calendar(TODAY, path=path, pause=0)
    assert not from_cache and len(requests) == 75  # 14 back + today + 60 ahead
    assert requests[0] == ("2026-10-08", "2026-10-08") and requests[-1] == ("2026-12-21", "2026-12-21")
    assert all(a == b for a, b in requests)  # never a wide request, which Finnhub truncates
    assert len(events) == 75 and events[0].hour == "amc"

    again, from_cache = api.cached_calendar(TODAY, path=path, pause=0)
    assert from_cache and len(requests) == 75 and len(again) == 75  # no new requests


def test_cache_from_a_previous_day_is_refetched(tmp_path):
    requests: list = []
    api = FinnhubEvents(api_key="k", client=_client(requests))
    path = tmp_path / "cal.json"
    api.cached_calendar(date(2026, 10, 21), path=path, pause=0)
    before = len(requests)
    _, from_cache = api.cached_calendar(TODAY, path=path, pause=0)
    assert not from_cache and len(requests) == before + 75


def test_failed_fetch_caches_nothing(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(base_url="https://finnhub.io/api/v1", transport=httpx.MockTransport(handler))
    path = tmp_path / "cal.json"
    try:
        FinnhubEvents(api_key="k", client=client).cached_calendar(TODAY, path=path, pause=0)
    except EventsUnavailable as exc:
        assert "token" not in str(exc)  # the API key must never end up in an error message
    else:
        raise AssertionError("expected the failed fetch to raise")
    assert not path.exists()

from __future__ import annotations

import json
import time
from datetime import date, datetime
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from tool.config import Config, load_all
from tool.data import schwab
from tool.data.barcache import cached_daily_bars, last_complete_session
from tool.data.ratelimit import RateLimiter
from tool.data.schwab import SchwabAuthRequired, SchwabDataSource, auth_status
from tool.levels import NY
from tool.store import Store


# ---------------------------------------------------------------- config
def test_config_defaults_equal_the_constants_they_replaced():
    from tool import (buy_options, cc_monitor, covered_calls, freshness, longterm, naked_calls, performance, puts,
                      screener, sections, whales)

    assert (naked_calls.DELTA_LOW, naked_calls.DELTA_HIGH, naked_calls.MAX_SPREAD_PCT) == (0.15, 0.25, 0.10)
    assert (naked_calls.BREAKOUT_NEAR, naked_calls.HIGH_RISK_STRIKE_ABOVE_BREAKOUT) == (0.98, 1.05)
    assert (puts.DELTA_LOW, puts.DELTA_HIGH, puts.SUSPECT_ANNUALIZED, puts.LOW_RETURN_NOTE) == (0.15, 0.25, 3.0, 0.15)
    assert (covered_calls.TARGET_DELTA_LOW, covered_calls.TARGET_DELTA_HIGH) == (0.20, 0.30)
    assert buy_options.SWING_DELTA == (0.40, 0.60) and buy_options.LOTTO_DELTA == (0.05, 0.25)
    assert (buy_options.LOTTO_MIN_PRICE_CAP, buy_options.LOTTO_PRICE_CAP_PCT) == (1.50, 0.004)
    assert buy_options.ODTE_DELTA == (0.10, 0.45) and buy_options.ODTE_PRICE_CAP_PCT == 0.008
    assert buy_options.MAG7 == ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    assert buy_options.INDEX_ETFS == ["SPY", "QQQ", "IWM"]
    assert (whales.WHALE_PREMIUM, whales.LARGE_PREMIUM, whales.MIN_VOLUME) == (1_000_000.0, 250_000.0, 300)
    assert (whales.FRESH_VOL_OI, whales.INDEX_SCALE) == (1.0, 10.0)
    assert (freshness.MARKET_OPEN.hour, freshness.MARKET_OPEN.minute) == (9, 30)
    assert (freshness.MARKET_CLOSE.hour, freshness.MARKET_CLOSE.minute) == (16, 0)
    assert (freshness.MORNING_GRACE_END.hour, freshness.MORNING_GRACE_END.minute) == (10, 30)
    assert freshness.MAX_AGE["intraday"].total_seconds() == 1800 and freshness.MAX_AGE["hourly"].total_seconds() == 7200
    assert (longterm.MIN_QUALITY_PASSES, longterm.EXTENDED_OVER_SMA50, longterm.MAX_STOP_DISTANCE) == (4, 0.10, 0.12)
    assert (longterm.HOLD_MONTHS, longterm.MIN_BARS) == (12, 200)
    assert performance.HORIZON_DAYS == 10
    assert (sections.HIGH_MAX, sections.MODERATE_MAX) == (0.20, 0.35)
    assert (cc_monitor.NEAR_STRIKE, cc_monitor.MIN_EXTRINSIC) == (0.01, 0.50)
    assert screener.STOP_ATR == 1.5 and screener.EARNINGS_WINDOW_DAYS == 60


def test_every_section_can_be_overridden_from_yaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "options:\n  naked_delta_low: 0.10\n  mag7: [AAPL, MSFT]\nwhales:\n  min_volume: 500\n"
        "account:\n  account_size: 50000\n  risk_pct: 0.02\ndata:\n  rate_limit_per_min: 60\nbogus:\n  x: 1\n",
        encoding="utf-8",
    )
    cfg = load_all(path)
    assert cfg.options.naked_delta_low == 0.10 and cfg.options.mag7 == ("AAPL", "MSFT")
    assert cfg.whales.min_volume == 500 and cfg.account.account_size == 50000 and cfg.account.risk_pct == 0.02
    assert cfg.data.rate_limit_per_min == 60 and cfg.options.naked_delta_high == 0.25  # untouched
    assert load_all(tmp_path / "missing.yaml") == Config()


# ---------------------------------------------------------------- rate limiter
class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.slept += seconds


def test_rate_limiter_lets_the_budget_through_then_waits_for_the_window():
    clock = FakeClock()
    limiter = RateLimiter(3, 60.0, clock=clock.now, sleep=clock.sleep)
    assert [limiter.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire() == pytest.approx(60.0)  # the 4th call waits for the first to age out
    assert clock.slept == pytest.approx(60.0)


def test_rate_limiter_does_not_wait_when_calls_are_spread_out():
    clock = FakeClock()
    limiter = RateLimiter(2, 60.0, clock=clock.now, sleep=clock.sleep)
    for _ in range(6):
        limiter.acquire()
        clock.t += 31.0  # two calls per 62 seconds stays under 2 per 60
    assert clock.slept == 0.0


# ---------------------------------------------------------------- Schwab adapter
class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(self.status_code, request=request))

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, prices=None, status=200):
        self.calls: list[list[str]] = []
        self.prices, self.status = prices, status

    def get_quotes(self, symbols):
        self.calls.append(list(symbols))
        if self.status != 200:
            return FakeResponse({}, self.status)
        prices = self.prices if self.prices is not None else {s: 100.0 for s in symbols}
        return FakeResponse({s: {"quote": {"lastPrice": prices[s]}} for s in symbols if s in prices})


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHWAB_APP_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    monkeypatch.setattr(schwab, "mark_auth_required", lambda reason, path=None: None)
    src = SchwabDataSource(token_path=tmp_path / "token.json")
    src._limiter = RateLimiter(10_000)  # never waits in tests
    return src


def test_quotes_are_batched_and_spot_uses_the_cache(source):
    source._client = FakeClient()
    symbols = [f"S{i}" for i in range(250)]
    prices = source.quotes(symbols)
    assert len(prices) == 250 and [len(c) for c in source._client.calls] == [100, 100, 50]
    assert source.spot("S7") == 100.0 and len(source._client.calls) == 3  # served from the cache


def test_spot_refetches_after_the_cache_expires_and_missing_symbols_raise(source):
    source._client = FakeClient(prices={"AAA": 10.0})
    assert source.spot("AAA") == 10.0
    source._spots["AAA"] = (10.0, time.monotonic() - 1)  # expired
    assert source.spot("AAA") == 10.0 and len(source._client.calls) == 2
    with pytest.raises(ValueError):
        source.spot("ZZZ")


def test_prefetch_spots_makes_following_spot_calls_free(source):
    source._client = FakeClient()
    source.prefetch_spots([f"S{i}" for i in range(20)], ttl=300)
    for i in range(20):
        source.spot(f"S{i}")
    assert len(source._client.calls) == 1


def test_every_request_goes_through_the_rate_limiter(source):
    class Counting:
        n = 0

        def acquire(self):
            Counting.n += 1

    source._limiter = Counting()
    source._client = FakeClient()
    source.quotes(["A", "B"])
    assert Counting.n == 1


def test_a_401_becomes_a_reauth_error(source):
    source._client = FakeClient(status=401)
    with pytest.raises(SchwabAuthRequired):
        source.quotes(["AAA"])


def test_an_expired_token_file_raises_reauth_before_any_request(source, tmp_path):
    (tmp_path / "token.json").write_text(json.dumps({"creation_timestamp": time.time() - 8 * 86400}), encoding="utf-8")
    with pytest.raises(SchwabAuthRequired):
        source._get_client()


def test_auth_status_covers_missing_old_flagged_and_fresh_tokens(tmp_path):
    token, flag = tmp_path / "token.json", tmp_path / "flag"
    assert auth_status(token, flag)[0] is False  # no token
    token.write_text(json.dumps({"creation_timestamp": time.time() - 8 * 86400}), encoding="utf-8")
    ok, message = auth_status(token, flag)
    assert not ok and "expired" in message
    token.write_text(json.dumps({"creation_timestamp": time.time() - 3600}), encoding="utf-8")
    assert auth_status(token, flag) == (True, "")
    flag.write_text("invalid_grant", encoding="utf-8")
    assert auth_status(token, flag)[0] is False


# ---------------------------------------------------------------- bar cache
def make_bars(start: str, end: str) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    return pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1e6}, index=days.tz_localize(NY)
    )


class Fetcher:
    def __init__(self, bars: pd.DataFrame):
        self.bars, self.days = bars, []

    def __call__(self, days: int) -> pd.DataFrame:
        self.days.append(days)
        return self.bars


NOW = datetime(2026, 10, 1, 10, 0, tzinfo=NY)  # Thursday, market open: the last complete session is Wednesday Sep 30


def test_last_complete_session_rolls_back_over_open_hours_and_weekends():
    assert last_complete_session(datetime(2026, 10, 1, 10, 0, tzinfo=NY)) == date(2026, 9, 30)
    assert last_complete_session(datetime(2026, 10, 1, 16, 30, tzinfo=NY)) == date(2026, 10, 1)
    assert last_complete_session(datetime(2026, 10, 5, 8, 0, tzinfo=NY)) == date(2026, 10, 2)  # Monday morning -> Friday
    assert last_complete_session(datetime(2026, 10, 3, 12, 0, tzinfo=NY)) == date(2026, 10, 2)  # Saturday -> Friday


def test_cache_miss_fetches_everything_and_hit_fetches_nothing(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    fetch = Fetcher(make_bars("2026-06-01", "2026-09-30"))
    first = cached_daily_bars(store, "AAA", 120, fetch, now=NOW)
    assert fetch.days == [120] and len(first) == len(fetch.bars)
    second = cached_daily_bars(store, "AAA", 120, fetch, now=NOW)
    assert fetch.days == [120]  # no new request
    assert len(second) == len(first) and second.index[-1].date() == date(2026, 9, 30)
    store.close()


def test_a_new_session_fetches_only_the_tail_and_merges(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    cached_daily_bars(store, "AAA", 120, Fetcher(make_bars("2026-06-01", "2026-09-30")), now=NOW)
    tail = Fetcher(make_bars("2026-09-28", "2026-10-01"))
    friday = datetime(2026, 10, 2, 10, 0, tzinfo=NY)  # last complete session is now Thursday Oct 1
    merged = cached_daily_bars(store, "AAA", 120, tail, now=friday)
    assert tail.days == [7]  # (Oct 2 - Sep 30) + 5, not the 120-day history
    assert merged.index[-1].date() == date(2026, 10, 1) and not merged.index.normalize().duplicated().any()
    store.close()


def test_a_bar_saved_intraday_is_not_trusted_and_gets_refetched(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    cached_daily_bars(store, "AAA", 120, Fetcher(make_bars("2026-06-01", "2026-09-30")), now=NOW)
    # an older version saved Sep 30's bar at 11:00 that same day, before the close: a partial snapshot
    store._conn.execute(
        "UPDATE daily_bars SET fetched_at = '2026-09-30T11:00:00-04:00' WHERE symbol = 'AAA' AND bar_date = '2026-09-30'"
    )
    store._conn.commit()
    assert store.load_trusted_daily_bars("AAA").index[-1].date() == date(2026, 9, 29)
    tail = Fetcher(make_bars("2026-09-28", "2026-09-30"))
    cached_daily_bars(store, "AAA", 120, tail, now=NOW)
    assert len(tail.days) == 1  # refetched the tail instead of serving the snapshot
    store.close()


def test_symbols_with_short_history_are_fetched_in_full(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    fetch = Fetcher(make_bars("2026-09-01", "2026-09-30"))  # a recent IPO: no bars back to the requested start
    cached_daily_bars(store, "NEW", 120, fetch, now=NOW)
    cached_daily_bars(store, "NEW", 120, fetch, now=NOW)
    assert fetch.days == [120, 120]
    store.close()


# ---------------------------------------------------------------- provenance
def test_every_recommendation_table_records_generated_at_data_as_of_and_source(tmp_path):
    from tool.buy_options import BuyRec
    from tool.cc_monitor import MonitorRow
    from tool.longterm import LongTermRec
    from tool.screener import Setup
    from tool.whales import WhaleRow

    store = Store(tmp_path / "t.sqlite")
    store.save_setups([Setup(symbol="AAA", kind="range", spot=1.0, atr=1.0, sma20=1.0, prior_high=1.0, change_20d=0.0,
                             data_as_of=date(2026, 10, 1))], day=date(2026, 10, 2))
    store.save_option_recs([("long_call", [BuyRec("AAA", "call", "swing", "skip", "no contract")])])
    store.save_longterm([LongTermRec("AAA", "skip", "reason")])
    store.save_cc_snapshot([MonitorRow("AAA", "HOLD", 10.0, date(2026, 10, 9), 1, 9.0, 0.5, 0.2, 0.1, 0.1, 7, "ok")])
    store.save_whales([WhaleRow("AAA", "C", date(2026, 10, 2), 0, 10.0, 9.0, 0.1, 1000, 500, 2.0, 2.0, 200000.0,
                                "ask", "bullish", "Large", 0.4, 0.3, True)])
    store.save_covered_calls([SimpleNamespace(symbol="AAA", account="taxable", status="skip", reason="r", free_lots=1,
                                              expiry=None, strike=None, bid=None, delta=None, notes=[])])
    for table in ("setups", "option_recs", "longterm_recs", "cc_snapshots", "whale_flow", "covered_calls"):
        row = store._conn.execute(f"SELECT generated_at, data_as_of, source FROM {table}").fetchone()
        assert row is not None and all(row), table
    setup = store._conn.execute("SELECT data_as_of, source FROM setups").fetchone()
    assert setup == ("2026-10-01", "schwab,finnhub")  # the bar date the setup rests on, not the scan date
    store.close()


# ---------------------------------------------------------------- IV history
def test_atm_iv_and_the_daily_recording(tmp_path):
    from tool.ivhistory import atm_iv, current_iv_rank, record_atm_iv

    chain = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "iv": [0.35, 0.30, 0.33]})
    assert atm_iv(chain, 101.0) == 0.30
    assert atm_iv(pd.DataFrame({"strike": [100.0], "iv": [None]}), 100.0) is None

    class Source:
        def option_chain(self, symbol, expiry=None, right="C"):
            return chain

        def spot(self, symbol):
            return 101.0

    store = Store(tmp_path / "t.sqlite")
    today = date(2026, 10, 2)
    assert record_atm_iv(Source(), store, ["AAA", "BBB", "AAA"], today, date(2026, 10, 16)) == 2
    assert record_atm_iv(Source(), store, ["AAA", "BBB"], today, date(2026, 10, 16)) == 0  # already recorded today
    assert store.iv_history("AAA") == [0.30]
    assert current_iv_rank(store, "AAA", 0.30) is None  # one day of history is not a rank
    store.close()

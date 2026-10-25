from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.cc_monitor import assess, monitor_covered_calls
from tool.holdings import Holdings, Lot, ShortOption
from tool.levels import NY

TODAY = date(2026, 10, 7)
EXPIRY = date(2026, 10, 9)
NOW = datetime(2026, 10, 7, 11, 0, tzinfo=NY)


def quote(bid, ask, delta=0.3):
    return pd.Series({"bid": bid, "ask": ask, "delta": delta})


def test_far_out_of_the_money_call_is_held():
    row = assess("TEST", 110.0, EXPIRY, 2, 100.0, quote(0.20, 0.25), TODAY)
    assert row.action == "HOLD" and row.distance_pct == pytest.approx(0.10) and row.dte == 2


def test_stock_within_one_percent_of_the_strike_is_a_close():
    row = assess("TEST", 100.0, EXPIRY, 1, 99.5, quote(0.80, 0.90), TODAY)
    assert row.action == "CLOSE" and "within" in row.reason


def test_in_the_money_with_little_time_value_is_an_early_assignment_warning():
    # spot 103 vs strike 100: intrinsic 3.00, mid 3.20 -> $0.20 of time value (< $0.50)
    row = assess("TEST", 100.0, EXPIRY, 1, 103.0, quote(3.15, 3.25), TODAY)
    assert row.action == "CLOSE" and row.extrinsic == pytest.approx(0.20) and "early-assignment" in row.reason


def test_in_the_money_with_plenty_of_time_value_still_closes_on_the_upside_rule():
    row = assess("TEST", 100.0, EXPIRY, 1, 101.0, quote(2.40, 2.60), TODAY)
    assert row.action == "CLOSE" and row.extrinsic == pytest.approx(1.50)
    assert "above the strike" in row.reason


class FakeSource:
    def __init__(self, spots, chains):
        self._spots, self._chains = spots, chains

    def spot(self, symbol):
        return self._spots[symbol]

    def daily_bars(self, symbol, lookback_days=90):
        raise RuntimeError("not needed")

    def option_chain(self, symbol, expiry=None, right="C"):
        chain = self._chains[(symbol, expiry)].copy()
        chain["expiry"] = expiry
        return chain


def holdings_with_short_call():
    return Holdings(
        lots=[Lot(symbol="TEST", account="rh_taxable", shares=100, cost=80.0, no_covered_calls=True)],
        short_options=[ShortOption(symbol="TEST", type="call", strike=100.0, expiry=EXPIRY, qty=1)],
    )


def test_close_becomes_a_roll_when_a_net_credit_exists():
    chains = {
        ("TEST", EXPIRY): pd.DataFrame({"strike": [100.0], "bid": [1.40], "ask": [1.50], "delta": [0.55]}),
        ("TEST", date(2026, 10, 16)): pd.DataFrame({"strike": [100.0, 102.0], "bid": [2.60, 1.80], "ask": [2.70, 1.90], "delta": [0.6, 0.4]}),
    }
    rows = monitor_covered_calls(holdings_with_short_call(), FakeSource({"TEST": 99.5}, chains), now=NOW)
    [row] = rows
    assert row.action == "ROLL"
    assert "2026-10-16" in row.detail and "$100.00" in row.detail and "net credit of $1.10" in row.detail  # 2.60 - 1.50


def test_close_stays_close_when_the_roll_would_cost_a_debit():
    chains = {
        ("TEST", EXPIRY): pd.DataFrame({"strike": [100.0], "bid": [3.00], "ask": [3.10], "delta": [0.8]}),
        ("TEST", date(2026, 10, 16)): pd.DataFrame({"strike": [100.0], "bid": [2.00], "ask": [2.10], "delta": [0.6]}),
    }
    [row] = monitor_covered_calls(holdings_with_short_call(), FakeSource({"TEST": 101.0}, chains), now=NOW)
    assert row.action == "CLOSE" and "no credit roll" in row.detail and "accept assignment" in row.detail


def test_roll_into_earnings_is_refused():
    from tool.data.earnings import EarningsEvent

    class FakeEvents:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, date(2026, 10, 14), None)]

    chains = {
        ("TEST", EXPIRY): pd.DataFrame({"strike": [100.0], "bid": [1.40], "ask": [1.50], "delta": [0.55]}),
        ("TEST", date(2026, 10, 16)): pd.DataFrame({"strike": [100.0], "bid": [2.60], "ask": [2.70], "delta": [0.6]}),
    }
    [row] = monitor_covered_calls(holdings_with_short_call(), FakeSource({"TEST": 99.5}, chains), now=NOW, events=FakeEvents())
    assert row.action == "CLOSE" and "earnings" in row.detail


def test_held_call_far_from_the_strike_is_just_held():
    chains = {("TEST", EXPIRY): pd.DataFrame({"strike": [100.0], "bid": [0.10], "ask": [0.15], "delta": [0.1]})}
    [row] = monitor_covered_calls(holdings_with_short_call(), FakeSource({"TEST": 90.0}, chains), now=NOW)
    assert row.action == "HOLD" and row.detail is None


def test_snapshots_are_saved_and_those_older_than_a_week_are_deleted(tmp_path):
    from tool.cc_monitor import MonitorRow
    from tool.store import Store

    store = Store(tmp_path / "t.sqlite")
    row = MonitorRow("TEST", "HOLD", 100.0, EXPIRY, 1, 90.0, 0.2, 0.1, 0.2, 0.11, 2, "far from the strike")
    store._conn.execute(
        "INSERT INTO cc_snapshots (ts, symbol, action) VALUES (?, ?, ?)", ("2026-09-01T10:00:00-04:00", "OLD", "HOLD")
    )
    store._conn.commit()
    assert store.save_cc_snapshot([row]) == 1
    symbols = {r[0] for r in store._conn.execute("SELECT symbol FROM cc_snapshots")}
    assert symbols == {"TEST"}  # the September row was pruned
    latest = store.latest_cc_snapshot()
    assert list(latest["symbol"]) == ["TEST"] and latest["action"].iloc[0] == "HOLD"
    assert len(store.cc_history("TEST")) == 1
    store.close()


def test_deep_in_the_money_call_without_a_quote_is_not_given_a_false_credit():
    # ZS-like: stock 234, strike 197.50, the strike is outside the quoted chain, so the buy-back price is unknown
    chains = {
        ("TEST", EXPIRY): pd.DataFrame({"strike": [230.0], "bid": [5.0], "ask": [5.1], "delta": [0.5]}),
        ("TEST", date(2026, 10, 16)): pd.DataFrame({"strike": [210.0], "bid": [22.75], "ask": [23.0], "delta": [0.9]}),
    }
    holdings = Holdings(
        lots=[Lot(symbol="TEST", account="rh_taxable", shares=100, cost=80.0, no_covered_calls=True)],
        short_options=[ShortOption(symbol="TEST", type="call", strike=197.5, expiry=EXPIRY, qty=1)],
    )
    chains[("TEST", EXPIRY)] = pd.DataFrame({"strike": [230.0], "bid": [5.0], "ask": [5.1], "delta": [0.5]})
    [row] = monitor_covered_calls(holdings, FakeSource({"TEST": 234.24}, chains), now=NOW)
    assert row.action == "CLOSE"  # the roll costs a debit: buy back at least $36.74 of intrinsic, sell only $22.75
    assert "no credit roll" in row.detail and "estimated at intrinsic value" in row.detail


def test_weekly_call_after_the_last_session_shows_the_expiry_outcome():
    after_close = datetime(2026, 10, 9, 17, 30, tzinfo=NY)
    chains = {}
    itm = Holdings(short_options=[ShortOption(symbol="ITM", type="call", strike=100.0, expiry=EXPIRY.replace(day=9), qty=1)])
    otm = Holdings(short_options=[ShortOption(symbol="OTM", type="call", strike=120.0, expiry=EXPIRY.replace(day=9), qty=1)])
    [a] = monitor_covered_calls(itm, FakeSource({"ITM": 105.0}, chains), now=after_close)
    [b] = monitor_covered_calls(otm, FakeSource({"OTM": 105.0}, chains), now=after_close)
    assert a.action == "EXPIRES" and "called away at $100.00" in a.reason
    assert b.action == "EXPIRES" and "expires worthless" in b.reason

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from tool.config import AccountConfig, CallConfig
from tool.data.earnings import EarningsEvent
from tool.earnings_calls import expiry_target, reaction_day, recommend_earnings_calls, upcoming_reports
from tool.levels import NY

MON = datetime(2026, 10, 26, 10, 0, tzinfo=NY)
RANK = lambda symbol, closes: (70.0, "HV")  # noqa: E731


def test_reaction_day_and_expiry_follow_the_report_time():
    assert reaction_day(date(2026, 10, 27), "amc") == date(2026, 10, 28)
    assert reaction_day(date(2026, 10, 27), "bmo") == date(2026, 10, 27)
    assert expiry_target(date(2026, 10, 27), "amc") == date(2026, 10, 30)
    assert expiry_target(date(2026, 10, 30), "amc") == date(2026, 11, 6)  # Friday after the close reacts Monday: next Friday
    assert expiry_target(date(2026, 10, 30), "bmo") == date(2026, 10, 30)


def test_upcoming_reports_keep_only_the_pool_inside_the_window():
    events = [EarningsEvent("AAA", date(2026, 10, 28), "amc"), EarningsEvent("BBB", date(2026, 11, 20), None),
              EarningsEvent("CCC", date(2026, 10, 28), "amc"), EarningsEvent("DDD", date(2026, 10, 26), "bmo"),
              EarningsEvent("EEE", date(2026, 10, 26), "amc")]
    got = upcoming_reports(events, date(2026, 10, 26), {"AAA", "BBB", "DDD", "EEE"}, 7)
    assert set(got) == {"AAA", "EEE"}  # BBB too far, CCC not in the pool, DDD already reported this morning


class Source:
    def __init__(self, expiry_offset=0):
        self.asked, self.offset = [], expiry_offset

    def daily_bars(self, symbol, lookback_days=90):
        return pd.DataFrame({"close": np.linspace(80, 100, 300)})

    def spot(self, symbol):
        return 100.0

    def option_chain(self, symbol, expiry=None, right="C", window_days=21, strike_count=20):
        self.asked.append((symbol, expiry))
        rows = [{"expiry": expiry, "strike": 100.0, "bid": 4.0, "ask": 4.2, "delta": 0.50, "iv": 0.60, "theta": -0.4,
                 "open_interest": 900},
                {"expiry": expiry, "strike": 90.0, "bid": 11.0, "ask": 11.2, "delta": 0.80, "iv": 0.60, "theta": -0.3,
                 "open_interest": 900}]
        return pd.DataFrame(rows)


def test_one_near_the_money_call_per_reporting_name_with_the_crush_flag_and_arithmetic():
    events = [EarningsEvent("AAA", date(2026, 10, 28), "amc"), EarningsEvent("ZZZ", date(2026, 10, 29), "bmo")]
    source = Source()
    recs = recommend_earnings_calls(["AAA", "ZZZ", "OFF"], events, source, MON, rank_fn=RANK, cfg=CallConfig(),
                                    account=AccountConfig(account_size=100_000, risk_pct=0.01))
    assert [r.symbol for r in recs] == ["AAA", "ZZZ"] and all(r.status == "recommend" for r in recs)
    aaa = recs[0]
    assert aaa.expiry == date(2026, 10, 30) and aaa.strike == 100.0 and aaa.delta == 0.50
    assert aaa.breakeven == 104.2 and aaa.trigger == "EVENT" and aaa.max_loss_per_contract == 420.0
    assert aaa.be_vs_move is not None and aaa.contracts == 2
    assert aaa.flags[0].startswith("IV CRUSH") and "rank 70" in aaa.flags[0]
    assert source.asked[0] == ("AAA", date(2026, 10, 30))


def test_no_expiry_after_the_report_or_no_delta_in_band_is_a_skip_with_a_reason():
    events = [EarningsEvent("AAA", date(2026, 10, 28), "amc")]

    class Early(Source):
        def option_chain(self, *a, **k):
            chain = super().option_chain(*a, **k)
            return chain.assign(expiry=date(2026, 10, 27))

    [early] = recommend_earnings_calls(["AAA"], events, Early(), MON, rank_fn=RANK)
    assert early.status == "skip" and "no expiry after the report" in early.reason

    class Deep(Source):
        def option_chain(self, *a, **k):
            return super().option_chain(*a, **k).assign(delta=0.9)

    [deep] = recommend_earnings_calls(["AAA"], events, Deep(), MON, rank_fn=RANK)
    assert deep.status == "skip" and "no call with delta" in deep.reason


def test_a_failing_symbol_does_not_stop_the_others():
    class Flaky(Source):
        def spot(self, symbol):
            if symbol == "AAA":
                raise RuntimeError("no quote")
            return 100.0

    events = [EarningsEvent("AAA", date(2026, 10, 28), "amc"), EarningsEvent("BBB", date(2026, 10, 28), "amc")]
    recs = recommend_earnings_calls(["AAA", "BBB"], events, Flaky(), MON, rank_fn=RANK)
    assert [(r.symbol, r.status) for r in recs] == [("AAA", "skip"), ("BBB", "recommend")]

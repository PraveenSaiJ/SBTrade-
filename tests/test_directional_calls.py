from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.buy_options import BuyRec, Plan
from tool.config import AccountConfig, CallConfig
from tool.directional_calls import (
    call_metrics, leaps_candidates, longterm_candidates, pick_delta_call, position_contracts, recommend_directional,
    recommend_leaps, swing_candidates,
)
from tool.levels import NY
from tool.store import Store
from tool.support_puts import friday_window

CFG = CallConfig()
FRIDAY = datetime(2026, 10, 23, 11, 0, tzinfo=NY)


def calls(rows: dict[float, dict], expiry: date) -> pd.DataFrame:
    base = {"bid": 1.0, "ask": 1.1, "delta": 0.5, "iv": 0.30, "theta": -0.05, "open_interest": 500}
    return pd.DataFrame([{"expiry": expiry, "strike": k, **{**base, **v}} for k, v in rows.items()])


# ---------------------------------------------------------------- strike choice
def test_the_strike_is_the_one_nearest_the_middle_of_the_delta_band():
    chain = calls({90.0: {"delta": 0.74}, 95.0: {"delta": 0.66}, 100.0: {"delta": 0.58}, 85.0: {"delta": 0.72}}, date(2026, 11, 20))
    assert pick_delta_call(chain, 0.60, 0.70)["strike"] == 95.0  # 0.66 is nearest 0.65
    assert pick_delta_call(chain, 0.70, 0.80)["strike"] == 90.0  # 0.74 is nearer the 0.75 midpoint than 0.72


def test_leaps_band_picks_the_closest_to_075():
    chain = calls({90.0: {"delta": 0.74}, 85.0: {"delta": 0.72}}, date(2027, 9, 17))
    assert pick_delta_call(chain, 0.70, 0.80)["strike"] == 90.0


def test_no_call_in_the_band_or_without_an_ask_gives_none():
    chain = calls({100.0: {"delta": 0.50}, 90.0: {"delta": 0.65, "ask": 0.0}}, date(2026, 11, 20))
    assert pick_delta_call(chain, 0.60, 0.70) is None


def test_open_interest_breaks_a_delta_tie():
    chain = calls({95.0: {"delta": 0.65, "open_interest": 10}, 96.0: {"delta": 0.65, "open_interest": 900}}, date(2026, 11, 20))
    assert pick_delta_call(chain, 0.60, 0.70)["strike"] == 96.0


# ---------------------------------------------------------------- the numbers on each row
def test_break_even_expected_move_theta_and_time_value():
    row = pd.Series({"strike": 100.0, "ask": 8.0, "iv": 0.30, "theta": -0.12})
    m = call_metrics(row, spot=105.0, dte=30)
    move = 105.0 * 0.30 * math.sqrt(30 / 365)
    assert m["breakeven"] == 108.0 and m["breakeven_pct"] == pytest.approx(108 / 105 - 1)
    assert m["expected_move"] == pytest.approx(move) and m["expected_move_pct"] == pytest.approx(move / 105)
    assert m["be_vs_move"] == pytest.approx(3.0 / move)  # break-even is only $3 over the stock
    assert m["theta_pct"] == pytest.approx(0.12 / 8.0)
    assert m["extrinsic_pct"] == pytest.approx((8.0 - 5.0) / 8.0)  # $5 intrinsic, $3 of time value


def test_an_out_of_the_money_call_is_all_time_value_and_missing_inputs_stay_none():
    m = call_metrics(pd.Series({"strike": 110.0, "ask": 2.0, "iv": None, "theta": None}), spot=105.0, dte=10)
    assert m["extrinsic_pct"] == 1.0
    assert m["expected_move"] is None and m["be_vs_move"] is None and m["theta_pct"] is None


def test_position_size_risks_a_percent_of_the_account_on_the_premium():
    acct = AccountConfig(account_size=100_000, risk_pct=0.01)
    assert position_contracts(5.00, acct) == 2  # $1,000 of risk / $500 per contract
    assert position_contracts(25.00, acct) == 0  # one contract is already more than the risk budget
    assert position_contracts(5.00, AccountConfig()) is None  # no account size configured: nothing guessed
    assert position_contracts(0.0, acct) is None


def test_the_directional_window_is_the_first_friday_the_monthly_and_the_last():
    assert friday_window(FRIDAY, 14, 60, 3) == [date(2026, 11, 6), date(2026, 11, 20), date(2026, 12, 18)]
    assert friday_window(FRIDAY, 14, 60, 1) == [date(2026, 11, 6)]


# ---------------------------------------------------------------- candidates
def setups_frame() -> pd.DataFrame:
    def row(sym, kind, score, excluded=None, entry=100.0):
        ok = entry is not None
        return {"symbol": sym, "kind": kind, "score": score, "excluded": excluded, "entry": entry,
                "stop": entry - 5 if ok else None, "t1": entry + 8 if ok else None, "t2": entry + 12 if ok else None, "atr": 4.0}

    return pd.DataFrame([
        row("LOW", "pullback", 0.40), row("TOP", "breakout", 0.80), row("MID", "bull_flag", 0.60),
        row("GONE", "breakout", 0.90, excluded="low volume"), row("DOWN", "downtrend", 0.95, entry=None),
        row("RANGE", "range", 0.99, entry=None), row("MA", "ma_pullback", 0.55),
    ])


def longterm_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "MID", "status": "recommend", "entry": 50.0, "stop": 45.0, "t1": 70.0, "t2": 90.0, "quality": 6, "trend": 3},
        {"symbol": "LTONLY", "status": "wait", "entry": 20.0, "stop": 18.0, "t1": 30.0, "t2": 40.0, "quality": 5, "trend": 3},
        {"symbol": "SKIPPED", "status": "skip", "entry": 20.0, "stop": 18.0, "t1": 30.0, "t2": 40.0, "quality": 6, "trend": 3},
    ])


def test_directional_candidates_are_the_bullish_unexcluded_setups_best_score_first():
    cands = swing_candidates(setups_frame())
    assert [c.symbol for c in cands] == ["TOP", "MID", "MA", "LOW"]
    assert all(c.source == "swing" and c.plan.side == "call" for c in cands)
    assert cands[0].plan.trigger == 100.0 and cands[0].plan.stop == 95.0 and cands[0].plan.chase_limit == 102.0
    assert [c.symbol for c in swing_candidates(setups_frame(), limit=2)] == ["TOP", "MID"]


def test_leaps_candidates_union_long_term_picks_and_swing_setups_with_long_term_levels_winning():
    cands = {c.symbol: c for c in leaps_candidates(setups_frame(), longterm_frame())}
    assert set(cands) == {"MID", "LTONLY", "TOP", "MA", "LOW"}  # SKIPPED and the excluded or non-bullish names are out
    assert cands["MID"].source == "long-term" and cands["MID"].plan.trigger == 50.0  # the long-term entry, not the swing 100
    assert cands["TOP"].source == "swing" and cands["TOP"].plan.trigger == 100.0
    assert cands["LTONLY"].label == "long-term wait" and cands["MID"].label == "long-term buy zone"
    assert list(cands)[:2] == ["MID", "LTONLY"]  # long-term picks come first, best quality first
    assert longterm_candidates(pd.DataFrame()) == [] and swing_candidates(pd.DataFrame()) == []


# ---------------------------------------------------------------- the engines
class Source:
    def __init__(self, chain_for):
        self.chain_for, self.asked = chain_for, []

    def daily_bars(self, symbol, lookback_days=90):
        return pd.DataFrame({"close": np.linspace(80, 100, 300)})

    def spot(self, symbol):
        return 100.0

    def option_chain(self, symbol, expiry=None, right="C", window_days=21, strike_count=20):
        self.asked.append((symbol, expiry, window_days, strike_count))
        return self.chain_for(expiry)


def chain_for(expiry):
    # the actual expiry is the requested Friday; delta 0.65 sits at strike 95 and 0.45 at strike 100
    return calls({95.0: {"delta": 0.65, "ask": 8.0, "bid": 7.9, "theta": -0.10}, 100.0: {"delta": 0.45, "ask": 5.0}}, expiry)


RANK = lambda symbol, closes: (60.0, "HV")  # noqa: E731


def test_directional_rows_have_delta_in_band_dte_in_window_and_the_arithmetic():
    recs = recommend_directional(setups_frame(), Source(chain_for), FRIDAY, rank_fn=RANK, cfg=CallConfig(max_directional=2),
                                 account=AccountConfig(account_size=100_000, risk_pct=0.01))
    good = [r for r in recs if r.status == "recommend"]
    assert {r.symbol for r in good} == {"TOP", "MID"} and len(good) == 6  # 2 names x 3 expiries
    r = good[0]
    assert 0.60 <= r.delta <= 0.70 and 14 <= r.dte <= 60 and r.strike == 95.0 and r.ask == 8.0
    assert r.breakeven == 103.0 and r.breakeven_pct == pytest.approx(0.03)
    assert r.max_loss_per_contract == 800.0 and r.contracts == 1  # $1,000 risk budget / $800
    assert r.theta_pct == pytest.approx(0.10 / 8.0) and r.rank == 60.0 and r.rank_source == "HV"
    assert r.expiry_type.startswith("directional ") and r.idea_source in ("breakout", "bull_flag")
    assert r.trigger in ("TRIGGERED", "WAITING", "EXTENDED") and r.stop_option == 4.0
    assert len({(x.symbol, x.expiry_type) for x in good}) == len(good)  # one key per expiry


def test_names_that_list_only_monthly_expiries_do_not_repeat_the_same_contract():
    monthly = date(2026, 11, 20)

    def only_monthly(expiry):
        return chain_for(monthly)  # every target resolves to the monthly

    recs = recommend_directional(setups_frame(), Source(only_monthly), FRIDAY, rank_fn=RANK, cfg=CallConfig(max_directional=1))
    assert [r.expiry for r in recs if r.status == "recommend"] == [monthly]


def test_a_missing_band_gives_a_skip_row_with_the_reason():
    def wrong_delta(expiry):
        return calls({100.0: {"delta": 0.50}}, expiry)

    recs = recommend_directional(setups_frame(), Source(wrong_delta), FRIDAY, rank_fn=RANK, cfg=CallConfig(max_directional=1))
    assert recs and all(r.status == "skip" and "delta 0.60-0.70" in r.reason for r in recs)


def test_iv_crush_flag_needs_earnings_before_expiry_and_a_rank_above_50():
    from tool.data.earnings import EarningsEvent

    class Events:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, date(2026, 11, 10), "amc")] if end >= date(2026, 11, 10) else []

    cfg = CallConfig(max_directional=1)
    high = recommend_directional(setups_frame(), Source(chain_for), FRIDAY, events=Events(), rank_fn=RANK, cfg=cfg)
    crush = [r for r in high if r.status == "recommend" and r.expiry >= date(2026, 11, 20)]
    assert crush and all(any("IV CRUSH RISK" in f for f in r.flags) for r in crush)
    first = next(r for r in high if r.status == "recommend" and r.expiry == date(2026, 11, 6))
    assert not any("IV CRUSH" in f or "earnings" in f for f in first.flags)  # expires before the report

    low = recommend_directional(setups_frame(), Source(chain_for), FRIDAY, events=Events(),
                                rank_fn=lambda s, c: (30.0, "HV"), cfg=cfg)
    later = [r for r in low if r.status == "recommend" and r.expiry >= date(2026, 11, 20)]
    assert later and all(not any("IV CRUSH" in f for f in r.flags) and any("earnings" in f for f in r.flags) for r in later)


def test_break_even_above_the_expected_move_is_flagged():
    def pricey(expiry):
        return calls({95.0: {"delta": 0.65, "ask": 30.0, "iv": 0.10}}, expiry)  # break-even 125 against a ~1.4% move

    [rec, *_] = recommend_directional(setups_frame(), Source(pricey), FRIDAY, rank_fn=RANK, cfg=CallConfig(max_directional=1))
    assert rec.be_vs_move > 1.0 and any("expected move" in f for f in rec.flags)


def test_leaps_use_the_wide_window_270_days_out_and_only_flag_close_earnings():
    from tool.data.earnings import EarningsEvent

    far = date(2027, 9, 17)
    src = Source(lambda expiry: calls({70.0: {"delta": 0.75, "ask": 35.0, "theta": -0.03}}, far))

    class Events:
        def earnings(self, symbol, start, end):
            return [EarningsEvent(symbol, date(2027, 4, 28), "amc")]  # months away: irrelevant to a LEAPS flag

    recs = recommend_leaps(setups_frame(), longterm_frame(), src, FRIDAY, events=Events(), rank_fn=RANK,
                           cfg=CallConfig(max_leaps=3))
    good = [r for r in recs if r.status == "recommend"]
    assert len(good) == 3 and all(0.70 <= r.delta <= 0.80 and r.dte >= 270 for r in good)
    assert all(r.expiry_type == f"leaps {far.isoformat()}" for r in good)
    asked = src.asked[0]
    assert asked[1] == date(2027, 7, 20) and asked[2] == 200 and asked[3] == 40  # 270 days out, wide window, 40 strikes
    assert not any("earnings" in f or "IV CRUSH" in f for r in good for f in r.flags)
    assert [r.idea_source for r in good][:1] == ["long-term buy zone"]

    near = recommend_leaps(setups_frame(), longterm_frame(), src, FRIDAY, rank_fn=RANK, cfg=CallConfig(max_leaps=1),
                           events=type("E", (), {"earnings": lambda self, s, a, b: [EarningsEvent(s, date(2026, 11, 3), "bmo")]})())
    assert any("IV CRUSH" in f for f in near[0].flags)  # a report 11 days out does matter


# ---------------------------------------------------------------- storage and the chain adapter
def test_new_columns_round_trip_and_distinct_expiries_both_survive(tmp_path):
    recs = recommend_directional(setups_frame(), Source(chain_for), FRIDAY, rank_fn=RANK, cfg=CallConfig(max_directional=1))
    store = Store(tmp_path / "t.sqlite")
    store.save_option_recs([("directional_call", recs)])
    saved = store.latest_option_recs("directional_call")
    assert len(saved) == len(recs) and saved["expiry"].nunique() == 3
    row = saved[saved["status"] == "recommend"].iloc[0]
    assert row["breakeven"] == 103.0 and row["be_vs_move"] == pytest.approx(3.0 / (100 * 0.3 * math.sqrt(row["dte"] / 365)))
    assert row["theta_pct"] == pytest.approx(0.0125) and row["rank_source"] == "HV" and row["idea_source"]
    assert row["max_loss"] == 800.0 and row["source"] == "schwab"
    store.close()


def test_the_schwab_chain_keeps_theta_and_takes_window_and_strike_count(tmp_path, monkeypatch):
    from tool.data.schwab import SchwabDataSource

    monkeypatch.setenv("SCHWAB_APP_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    asked = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            leg = {"bid": 69.0, "ask": 70.0, "delta": 0.75, "volatility": 40.0, "openInterest": 10, "totalVolume": 5,
                   "last": 69.5, "theta": -0.04}
            return {"callExpDateMap": {"2027-09-17:330": {"305.0": [leg]}}}

    class Client:
        def get_option_chain(self, symbol, **kwargs):
            asked.update(kwargs, symbol=symbol)
            return Response()

    src = SchwabDataSource(token_path=tmp_path / "t.json")
    src._client = Client()
    from tool.data.ratelimit import RateLimiter

    src._limiter = RateLimiter(10_000)
    chain = src.option_chain("NVDA", expiry=date(2027, 7, 20), right="C", window_days=200, strike_count=40)
    assert asked["strike_count"] == 40 and (asked["to_date"] - asked["from_date"]).days == 200
    assert chain.iloc[0]["theta"] == -0.04 and chain.iloc[0]["expiry"] == date(2027, 9, 17) and chain.iloc[0]["iv"] == pytest.approx(0.40)
    src.option_chain("NVDA", expiry=date(2027, 7, 20))  # defaults are unchanged for every existing caller
    assert asked["strike_count"] == 20 and (asked["to_date"] - asked["from_date"]).days == 21

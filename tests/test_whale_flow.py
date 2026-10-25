from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.holdings import Holdings, LongOption, Lot, ShortOption
from tool.levels import NY
from tool.store import Store
from tool.whale_agg import (
    burst_flags, end_of_day, flow_bias, net_flow, position_flow, position_map, relation,
)
from tool.whales import WhaleRow, classify_intent, noise_reason, whales_from_chain

TODAY = date(2026, 10, 22)
EXPIRY = date(2026, 10, 23)


def chain(volume=5000, last=2.0, delta=0.5, oi=1000, strike=100.0, right="C", bid=1.9, ask=2.0):
    return pd.DataFrame([{"expiry": EXPIRY, "strike": strike, "right": right, "bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                          "delta": delta, "iv": 0.3, "open_interest": oi, "volume": volume, "last": last}])


# ---------------------------------------------------------------- opening vs closing
@pytest.mark.parametrize("volume,oi,expected", [
    (1500, 1000, "opening"), (1000, 1000, "opening"), (600, 1000, "likely opening"), (500, 1000, "likely opening"),
    (499, 1000, "closing or rolling"), (900, None, "unknown"), (900, 0, "unknown"),
])
def test_intent_from_volume_against_open_interest(volume, oi, expected):
    assert classify_intent(volume, oi) == expected


# ---------------------------------------------------------------- noise
def test_deep_in_the_money_is_noise_but_an_ordinary_itm_call_is_not():
    assert "deep in the money" in noise_reason("AAA", "C", 80.0, 100.0, 0.92, 21.0, TODAY)
    assert "deep in the money" in noise_reason("AAA", "P", 120.0, 100.0, -0.95, 21.0, TODAY)
    assert noise_reason("AAA", "C", 95.0, 100.0, 0.65, 7.0, TODAY) is None


def test_dividend_capture_needs_an_itm_call_thin_time_value_and_an_imminent_ex_date():
    fund = {"XYZ": {"dividendAmount": 4.0, "dividendDate": "2026-10-24 00:00:00.0"}}  # $1.00 a quarter, ex in 2 days
    thin = noise_reason("XYZ", "C", 90.0, 100.0, 0.80, 10.3, TODAY, fund)  # $0.30 of time value
    assert thin and "dividend capture" in thin
    assert noise_reason("XYZ", "C", 90.0, 100.0, 0.80, 12.0, TODAY, fund) is None  # $2.00 of time value: a real view
    assert noise_reason("XYZ", "P", 110.0, 100.0, -0.80, 10.3, TODAY, fund) is None  # puts are not dividend capture
    later = {"XYZ": {"dividendAmount": 4.0, "dividendDate": "2026-11-20 00:00:00.0"}}
    assert noise_reason("XYZ", "C", 90.0, 100.0, 0.80, 10.3, TODAY, later) is None  # ex-date a month away
    assert noise_reason("XYZ", "C", 90.0, 100.0, 0.80, 10.3, TODAY, None) is None


def test_rows_carry_intent_and_noise_from_the_chain():
    [row] = whales_from_chain("AAA", chain(volume=5000, oi=1000, delta=0.93), 100.0, TODAY)
    assert row.intent == "opening" and row.noise and "deep in the money" in row.noise
    [row] = whales_from_chain("AAA", chain(volume=5000, oi=20000), 100.0, TODAY)
    assert row.intent == "closing or rolling" and row.noise is None


# ---------------------------------------------------------------- burst
KEY = ("AAA", "C", EXPIRY.isoformat(), 100.0)


def test_burst_is_the_premium_added_since_the_previous_scan():
    big = whales_from_chain("AAA", chain(volume=5000), 100.0, TODAY, previous={KEY: 0})[0]
    assert big.burst_volume == 5000 and big.burst_premium == 5000 * 2.0 * 100 and big.is_burst  # $1M added at once
    small = whales_from_chain("AAA", chain(volume=5000), 100.0, TODAY, previous={KEY: 4000})[0]
    assert small.burst_volume == 1000 and small.burst_premium == 200_000 and not small.is_burst  # under $250K
    new = whales_from_chain("AAA", chain(volume=5000), 100.0, TODAY, previous={})[0]
    assert new.burst_volume == 5000  # a contract new to the snapshot counts from zero
    assert whales_from_chain("AAA", chain(volume=5000), 100.0, TODAY, previous=None)[0].burst_volume is None  # no scan to compare
    backwards = whales_from_chain("AAA", chain(volume=5000), 100.0, TODAY, previous={KEY: 9000})[0]
    assert backwards.burst_volume == 0  # a lower volume (a bad quote) never gives a negative burst


def test_the_snapshot_collects_every_contract_with_volume_not_just_whales():
    snap: list = []
    whales_from_chain("AAA", pd.concat([chain(volume=5000), chain(volume=400, strike=105.0, last=0.5, ask=0.6, bid=0.4)]),
                      100.0, TODAY, snapshot=snap)
    assert sorted(s[3] for s in snap) == [100.0, 105.0]  # the small one is included so the next scan can see it grow
    assert all(s[0] == "AAA" and s[4] >= 300 for s in snap)


def test_burst_flags_scale_for_the_index_etfs():
    df = pd.DataFrame({"symbol": ["AAA", "SPY", "SPY"], "burst_premium": [300_000.0, 300_000.0, 3_000_000.0]})
    assert list(burst_flags(df)) == [True, False, True]  # $250K for a stock, $2.5M for SPY
    assert list(burst_flags(pd.DataFrame({"symbol": ["AAA"]}))) == [False]


# ---------------------------------------------------------------- storage
def test_the_snapshot_is_replaced_and_only_comparable_scans_are_loaded(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    now = datetime(2026, 10, 22, 10, 30, tzinfo=NY)
    assert store.load_whale_snapshot(now) is None  # nothing yet
    store.save_whale_snapshot([("AAA", "C", "2026-10-23", 100.0, 5000), ("AAA", "P", "2026-10-23", 95.0, 800)],
                              "2026-10-22T10:15:00-04:00")
    assert store.load_whale_snapshot(now) == {("AAA", "C", "2026-10-23", 100.0): 5000, ("AAA", "P", "2026-10-23", 95.0): 800}
    store.save_whale_snapshot([("BBB", "C", "2026-10-23", 50.0, 400)], "2026-10-22T10:15:00-04:00")
    assert list(store.load_whale_snapshot(now)) == [("BBB", "C", "2026-10-23", 50.0)]  # replaced, not appended
    assert store.load_whale_snapshot(datetime(2026, 10, 22, 11, 5, tzinfo=NY)) is None  # 50 minutes old
    assert store.load_whale_snapshot(datetime(2026, 10, 23, 10, 20, tzinfo=NY)) is None  # a different day
    store.close()


def test_new_whale_columns_round_trip(tmp_path):
    row = WhaleRow("AAA", "C", EXPIRY, 1, 100.0, 99.0, 0.01, 5000, 1000, 5.0, 2.0, 1_000_000.0, "ask", "bullish", "WHALE",
                   0.5, 0.3, True, intent="opening", burst_volume=5000, burst_premium=1_000_000.0, is_burst=True,
                   noise="deep in the money")
    store = Store(tmp_path / "t.sqlite")
    store.save_whales([row])
    saved = store.latest_whales().iloc[0]
    assert (saved["intent"], saved["burst_volume"], saved["noise"]) == ("opening", 5000, "deep in the money")
    assert saved["burst_premium"] == 1_000_000.0 and saved["source"] == "schwab"
    store.close()


# ---------------------------------------------------------------- net flow over days
def history() -> pd.DataFrame:
    def r(ts, symbol, lean, premium, right="C", strike=100.0, noise=None):
        return {"ts": ts, "symbol": symbol, "right": right, "expiry": "2026-10-30", "strike": strike, "premium": premium,
                "lean": lean, "noise": noise}

    return pd.DataFrame([
        r("2026-10-20T10:00:00-04:00", "AAA", "bullish", 300_000.0),   # day 1, early scan of a contract...
        r("2026-10-20T15:45:00-04:00", "AAA", "bullish", 800_000.0),   # ...superseded by its last scan of the day
        r("2026-10-20T15:45:00-04:00", "AAA", "bearish", 400_000.0, right="P", strike=95.0),
        r("2026-10-21T15:45:00-04:00", "AAA", "bearish", 900_000.0, right="P", strike=95.0),
        r("2026-10-21T15:45:00-04:00", "AAA", "neutral", 5_000_000.0, strike=110.0),         # mid prints: left out
        r("2026-10-21T15:45:00-04:00", "AAA", "bullish", 7_000_000.0, strike=70.0, noise="deep ITM"),  # noise: left out
        r("2026-10-21T15:45:00-04:00", "BBB", "bullish", 600_000.0),
    ])


def test_end_of_day_keeps_each_contracts_last_row_and_drops_noise():
    eod = end_of_day(history())
    day1 = eod[eod["ts"].str[:10] == "2026-10-20"]
    assert sorted(day1["premium"]) == [400_000.0, 800_000.0]  # the 10:00 row of the same contract is gone
    assert not (eod["noise"].notna()).any()


def test_one_day_and_five_day_totals():
    h = history()
    one = net_flow(h, 1).set_index("symbol")
    assert one.loc["AAA", "bearish"] == 900_000.0 and one.loc["AAA", "bullish"] == 0.0 and one.loc["AAA", "net"] == -900_000.0
    assert one.loc["BBB", "net"] == 600_000.0
    five = net_flow(h, 5).set_index("symbol")
    assert five.loc["AAA", "bullish"] == 800_000.0 and five.loc["AAA", "bearish"] == 1_300_000.0  # 400K + 900K
    assert five.loc["AAA", "net"] == -500_000.0 and five.loc["AAA", "calls"] == 800_000.0 and five.loc["AAA", "puts"] == 1_300_000.0
    assert five.loc["AAA", "bias"] == "bearish" and one.loc["AAA", "bias"] == "bearish"  # 1.3M is 1.6x the 0.8M
    assert net_flow(pd.DataFrame(), 5).empty


def test_flow_bias_uses_the_ratio():
    assert flow_bias(900, 100) == "bullish" and flow_bias(100, 900) == "bearish"
    assert flow_bias(150, 100) == "mixed" and flow_bias(0, 0) == "none"


# ---------------------------------------------------------------- your positions
def holdings() -> Holdings:
    return Holdings(
        lots=[Lot(symbol="AAA", account="rh_taxable", shares=100, cost=50.0),
              Lot(symbol="EEE", account="rh_taxable", shares=100, cost=50.0)],
        short_options=[ShortOption("BBB", "call", 110.0, date(2026, 10, 30), 1), ShortOption("CCC", "put", 90.0, date(2026, 10, 30), 1),
                       ShortOption("EEE", "call", 120.0, date(2026, 10, 30), 1)],
        long_options=[LongOption("DDD", "put", 80.0, date(2026, 11, 20), 2)],
    )


def test_position_map_reads_shares_short_and_long_options_and_the_tracker():
    tracked = pd.DataFrame([
        {"symbol": "TTT", "opt_type": "call", "qty": 1, "strike": 50.0, "expiry": "2026-11-20", "status": "open"},
        {"symbol": "OLD", "opt_type": "call", "qty": 1, "strike": 50.0, "expiry": "2026-11-20", "status": "closed"},
    ])
    pm = position_map(holdings(), tracked, TODAY)
    assert {k: [(p.kind, p.exposure) for p in v] for k, v in pm.items()} == {
        "AAA": [("shares", "long")], "EEE": [("shares", "long"), ("short call", "short")],
        "BBB": [("short call", "short")], "CCC": [("short put", "long")], "DDD": [("long put", "short")],
        "TTT": [("long call", "long")],
    }  # the closed tracker entry is ignored


def flow(symbol: str, bias: str) -> pd.DataFrame:
    net = {"bullish": 1e6, "bearish": -1e6, "mixed": 0.0}[bias]
    return pd.DataFrame([{"symbol": symbol, "bullish": 0, "bearish": 0, "net": net, "calls": 0, "puts": 0, "contracts": 1, "bias": bias}])


@pytest.mark.parametrize("symbol,bias,status", [
    ("AAA", "bearish", "CONFLICT"), ("AAA", "bullish", "supports"),   # shares
    ("BBB", "bullish", "CONFLICT"), ("BBB", "bearish", "supports"),   # short call is hurt by a rally
    ("CCC", "bearish", "CONFLICT"), ("CCC", "bullish", "supports"),   # short put is hurt by a fall
    ("DDD", "bullish", "CONFLICT"), ("DDD", "bearish", "supports"),   # long put is hurt by a rally
    ("EEE", "bullish", "mixed"), ("EEE", "bearish", "mixed"),         # covered call: shares and short call disagree
    ("AAA", "mixed", "neutral"),
])
def test_conflict_matrix(symbol, bias, status):
    out = position_flow(position_map(holdings(), None, TODAY), flow(symbol, bias), pd.DataFrame())
    assert out.iloc[0]["status"] == status


def test_todays_flow_decides_and_the_five_day_flow_is_the_fallback():
    pm = position_map(holdings(), None, TODAY)
    quiet_today = pd.DataFrame(columns=["symbol", "net", "bias"])
    out = position_flow(pm, quiet_today, flow("AAA", "bearish"))
    assert out.iloc[0]["status"] == "CONFLICT" and out.iloc[0]["flow_today"] == "none" and out.iloc[0]["flow_5d"] == "bearish"
    out = position_flow(pm, flow("AAA", "bullish"), flow("AAA", "bearish"))
    assert out.iloc[0]["status"] == "supports"  # today's bullish flow wins over the 5-day bearish
    assert position_flow(pm, flow("ZZZ", "bullish"), pd.DataFrame()).empty  # flow on a name you do not hold is not listed
    assert relation(position_map(holdings(), None, TODAY)["AAA"][0], "none") == "neutral"


def test_options_past_their_expiry_are_not_positions():
    stale = Holdings(
        lots=[],
        short_options=[ShortOption("OLD", "call", 100.0, date(2026, 10, 16), 1)],
        long_options=[LongOption("OLD2", "call", 100.0, date(2026, 10, 16), 1)],
    )
    assert position_map(stale, None, TODAY) == {}
    tracked = pd.DataFrame([{"symbol": "T", "opt_type": "call", "qty": 1, "strike": 50.0, "expiry": "2026-10-16", "status": "open"}])
    assert position_map(Holdings(lots=[]), tracked, TODAY) == {}

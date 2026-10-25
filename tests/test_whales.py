from datetime import date

import pandas as pd

from tool.whales import lean, trade_side, whales_from_chain

TODAY = date(2026, 10, 16)


def _chain(**row):
    base = {"expiry": TODAY, "strike": 600.0, "right": "C", "bid": 2.0, "ask": 2.2, "mid": 2.1, "delta": 0.4,
            "iv": 0.2, "open_interest": 1000, "volume": 5000, "last": 2.2}
    base.update(row)
    return pd.DataFrame([base])


def test_trade_side_and_lean():
    assert trade_side(2.2, 2.0, 2.2) == "ask"
    assert trade_side(2.0, 2.0, 2.2) == "bid"
    assert trade_side(2.1, 2.0, 2.2) == "mid"
    assert lean("C", "ask") == "bullish" and lean("P", "ask") == "bearish"
    assert lean("C", "bid") == "bearish" and lean("P", "bid") == "bullish"
    assert lean("C", "mid") == "neutral"


def test_big_premium_contract_is_a_fresh_whale():
    rows = whales_from_chain("NVDA", _chain(), 598.0, TODAY)
    assert len(rows) == 1
    r = rows[0]
    assert r.premium == 5000 * 2.2 * 100  # $1.1M
    assert r.tier == "WHALE" and r.fresh and r.dte == 0 and r.lean == "bullish"


def test_small_or_quiet_contracts_are_ignored():
    assert whales_from_chain("NVDA", _chain(volume=100), 598.0, TODAY) == []
    assert whales_from_chain("NVDA", _chain(volume=400, last=1.0, mid=1.0), 598.0, TODAY) == []  # $40K


def test_missing_volume_column_is_safe():
    assert whales_from_chain("NVDA", _chain().drop(columns=["volume"]), 598.0, TODAY) == []

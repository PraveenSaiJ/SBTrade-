from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.config import AccountConfig, ScreenerConfig
from tool.indicators import recent_swing_low
from tool.levels import NY
from tool.screener import Setup, evaluate, position_size, structural_stop
from tool.signals import signal_rows
from tool.store import Store

CFG = ScreenerConfig()


# ---------------------------------------------------------------- structural stop
@pytest.mark.parametrize("swing_low,expected", [
    (97.0, 97.0),   # below the level - 0.5 ATR (98): the swing low wins, inside the 2 ATR cap (92)
    (90.0, 92.0),   # a very deep swing low is clamped to entry - 2 ATR
    (99.5, 98.0),   # a swing low above the buffer: the buffer wins
    (None, 94.0),   # nothing to lean on: the old entry - 1.5 ATR
])
def test_structural_stop(swing_low, expected):
    assert structural_stop(100.0, 4.0, swing_low, CFG) == pytest.approx(expected)


def test_the_stop_is_never_wider_than_two_atr():
    for low in (50.0, 80.0, 91.0, 95.0):
        assert 100.0 - structural_stop(100.0, 4.0, low, CFG) <= 2 * 4.0 + 1e-9


# ---------------------------------------------------------------- recent swing low
def test_recent_swing_low_is_the_newest_confirmed_fractal_below_the_entry():
    lows = pd.Series([10, 9, 7, 9, 10, 11, 8, 11, 12, 13.0])  # fractals at 7 (index 2) and 8 (index 6)
    assert recent_swing_low(lows, below=20.0, lookback=20, window=2) == 8.0  # newest first
    assert recent_swing_low(lows, below=7.5, lookback=20, window=2) == 7.0  # 8 is not below the entry


def test_recent_swing_low_falls_back_to_the_lowest_recent_low_and_then_none():
    rising = pd.Series(np.arange(10.0, 40.0))  # no fractal anywhere
    assert recent_swing_low(rising, below=100.0, lookback=20, window=2, fallback_bars=10) == 30.0
    assert recent_swing_low(rising, below=25.0, lookback=20, window=2, fallback_bars=10) is None


# ---------------------------------------------------------------- position sizing
def test_position_size_risks_a_percent_of_the_account_rounded_down():
    # $100k at 1% risks $1,000; $5 of risk per share -> 200 shares
    assert position_size(fill=100.0, stop=95.0, account_size=100_000, risk_pct=0.01) == (200, 20_000.0, 1_000.0)
    assert position_size(fill=100.0, stop=93.0, account_size=100_000, risk_pct=0.01)[0] == 142  # floor(142.8)


def test_position_size_never_exceeds_the_account():
    shares, value, risk = position_size(fill=600.0, stop=599.0, account_size=100_000, risk_pct=0.01)
    assert shares == 166 and value <= 100_000  # 1,000 shares by risk, but only 166 fit in the account


def test_position_size_is_none_without_an_account_or_with_a_bad_stop():
    assert position_size(100.0, 95.0, None, 0.01) is None
    assert position_size(100.0, 100.0, 100_000, 0.01) is None
    assert position_size(100.0, 105.0, 100_000, 0.01) is None


def _breakout_bars() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    bars = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1e6}, index=dates)
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [101.0, 105.0, 100.5, 104.0, 2.5e6]
    return bars


def test_evaluate_sizes_the_position_only_when_an_account_is_configured():
    now = datetime(2026, 10, 2, 17, 0, tzinfo=NY)
    none = evaluate("TEST", _breakout_bars(), spot=104.5, now=now, cfg=CFG, account=AccountConfig())
    assert none.shares is None
    sized = evaluate("TEST", _breakout_bars(), spot=104.5, now=now, cfg=CFG,
                     account=AccountConfig(account_size=50_000, risk_pct=0.01))
    fill = 104.5  # price above the 102 entry
    assert sized.shares == int(500 // (fill - sized.stop))
    assert sized.risk_dollars == pytest.approx(sized.shares * (fill - sized.stop))
    assert sized.position_value == pytest.approx(sized.shares * fill)


# ---------------------------------------------------------------- signals log
def make(symbol, **kw) -> Setup:
    base = dict(symbol=symbol, kind="breakout", spot=104.0, atr=4.0, sma20=100.0, prior_high=102.0, change_20d=0.05,
                entry=102.0, stop=96.0, t1=110.0, t2=114.0, rr=1.3, volume_ratio=2.0, score=0.6, shares=100,
                data_as_of=date(2026, 10, 1))
    base.update(kw)
    return Setup(**base)


def test_signals_log_recommended_and_excluded_rows_with_all_inputs(tmp_path):
    setups = [
        make("GOOD"),
        make("DROP", excluded="low volume (1.0x vs 1.5x needed)", excluded_code="low_volume"),
        make("RNG", kind="range", entry=None, stop=None, t1=None, t2=None, rr=None, score=None),  # not a signal
    ]
    rows = signal_rows(setups, "2026-10-02", CFG, "2026-10-02T09:05:00-04:00")
    assert [r["symbol"] for r in rows] == ["GOOD", "DROP"]
    store = Store(tmp_path / "s.sqlite")
    assert store.save_signals(rows) == 2
    df = store.signals("swing").set_index("symbol")
    assert df.loc["GOOD", "status"] == "recommended" and df.loc["DROP", "status"] == "excluded"
    assert df.loc["DROP", "excluded_code"] == "low_volume" and df.loc["GOOD", "shares"] == 100
    assert df.loc["GOOD", "data_as_of"] == "2026-10-01" and df.loc["GOOD", "source"] == "schwab,finnhub"
    import json

    inputs = json.loads(df.loc["GOOD", "inputs"])
    assert inputs["atr10"] == 4.0 and inputs["config"]["min_volume_ratio"] == 1.5  # inputs and the config used
    store.close()


def test_a_rescan_updates_the_row_but_keeps_a_recorded_verdict(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.save_signals(signal_rows([make("AAA")], "2026-10-02", CFG, "t1"))
    store._conn.execute("UPDATE signals SET verdict = 'Take', verdict_reason = 'clean' WHERE symbol = 'AAA'")
    store._conn.commit()
    store.save_signals(signal_rows([make("AAA", score=0.7)], "2026-10-02", CFG, "t2"))
    df = store.signals()
    assert len(df) == 1 and df.iloc[0]["score"] == 0.7 and df.iloc[0]["generated_at"] == "t2"
    assert df.iloc[0]["verdict"] == "Take" and df.iloc[0]["verdict_reason"] == "clean"
    store.close()


def test_recheck_updates_the_signal_status(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.save_signals(signal_rows([make("AAA")], "2026-10-02", CFG, "t1"))
    store.update_signal_status("2026-10-02", "swing", "AAA", "failed", "excluded", "failed", "failed: back under", 0.2, 0.5)
    row = store.signals().iloc[0]
    assert (row["setup"], row["status"], row["excluded_code"]) == ("failed", "excluded", "failed")
    store.close()


def test_exclusion_codes_cover_every_reason():
    from tool.screener import exclusion

    today = date(2026, 10, 13)
    ok = dict(last_close=104.0, rsi=60.0, atr14=4.0, sma50=98.0, sma50_rising=True, sma200=90.0)
    cases = [
        (make("A", kind="failed", **ok), "failed"),
        (make("S", spot=95.0, **ok), "stopped"),  # price already under the 96 stop
        (make("B", volume_ratio=1.0, **ok), "low_volume"),
        (make("C", earnings=date(2026, 10, 20), **ok), "earnings_hold"),
        (make("D", **{**ok, "last_close": 85.0}), "trend"),
        (make("E", **{**ok, "rsi": 80.0}), "overextended"),
    ]
    for setup, code in cases:
        assert exclusion(setup, CFG, today)[0] == code
    assert exclusion(make("F", earnings=date(2026, 10, 20), expected_move_pct=0.09, **ok), CFG, today,
                     allow_earnings=True, expected_move_pct=0.09)[0] == "expected_move"
    assert exclusion(make("G", **ok), CFG, today) is None

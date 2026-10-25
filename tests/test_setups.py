from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tool.config import ScreenerConfig
from tool.indicators import atr
from tool.levels import NY
from tool.screener import ACTIONABLE, LABELS, apply_filters, evaluate
from tool.setups import detect_bull_flag, detect_ma_pullback

CFG = ScreenerConfig()
NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)


def mk(close, vol=None, spread=1.0) -> pd.DataFrame:
    close = np.asarray(close, float)
    n = len(close)
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close,
         "volume": vol if vol is not None else np.full(n, 1e6)}, index=idx,
    )


UP = 100 + 0.25 * np.arange(250)  # a steady uptrend: SMA20 > SMA50 > SMA200, all rising


def uptrend_with_dip(dip) -> pd.DataFrame:
    return mk(np.concatenate([UP, np.asarray(dip) + UP[-1]]))


# ---------------------------------------------------------------- pullback to the 20/50-day average
def test_dip_to_the_sma50_that_holds_is_an_ma_pullback():
    bars = uptrend_with_dip([1, -1.5, -3.0, -4.2, -3.4])
    found = detect_ma_pullback(bars, float(atr(bars, 10).iloc[-1]), CFG)
    assert found is not None and found.average == "SMA50"
    assert found.entry == pytest.approx(bars["close"].rolling(50).mean().iloc[-1])


def test_a_close_back_above_the_sma20_reports_the_sma20():
    # a steeper run-up leaves the SMA20 far enough under the highs for a 3% dip to still close above it
    steep = np.concatenate([100 + 0.25 * np.arange(220), 155 + 1.2 * np.arange(1, 31)])
    bars = mk(np.concatenate([steep, steep[-1] + np.array([1, -2, -5, -7.5, -6.0])]))
    found = detect_ma_pullback(bars, float(atr(bars, 10).iloc[-1]), CFG)
    assert found is not None and found.average == "SMA20"
    assert found.entry == pytest.approx(bars["close"].rolling(20).mean().iloc[-1])


def test_a_stock_still_at_its_highs_is_not_a_pullback():
    bars = uptrend_with_dip([0.2, 0.4, 0.6, 0.8, 1.0])  # within 3% of the 20-day high
    assert detect_ma_pullback(bars, float(atr(bars, 10).iloc[-1]), CFG) is None


def test_no_pullback_when_the_close_is_under_the_average_it_touched():
    bars = uptrend_with_dip([1, -2, -6, -9, -12])  # falls through both averages
    assert detect_ma_pullback(bars, float(atr(bars, 10).iloc[-1]), CFG) is None


def test_no_pullback_in_a_downtrend():
    down = 200 - 0.25 * np.arange(250)
    bars = mk(np.concatenate([down, down[-1] + np.array([1, -0.5, 0, -0.5, 0.5])]))
    assert detect_ma_pullback(bars, float(atr(bars, 10).iloc[-1]), CFG) is None


def test_short_history_gives_no_ma_pullback():
    assert detect_ma_pullback(mk(UP[:100]), 1.0, CFG) is None


# ---------------------------------------------------------------- bull flag
POLE = np.linspace(100, 125, 12)
FLAG = np.array([123.5, 123.0, 122.8, 123.2, 122.9, 123.1])


def flag_bars(pole=POLE, flag=FLAG, pole_vol=3e6, flag_vol=0.8e6, extra=None, extra_vol=None) -> pd.DataFrame:
    close = np.concatenate([np.full(220, 100.0), pole, flag] + ([np.asarray(extra, float)] if extra is not None else []))
    vol = np.concatenate([np.full(220, 1e6), np.full(len(pole), pole_vol), np.full(len(flag), flag_vol)]
                         + ([np.asarray(extra_vol, float)] if extra_vol is not None else []))
    return mk(close, vol, spread=0.4)


def test_a_pole_then_a_tight_quiet_pause_is_a_bull_flag_waiting_to_break():
    found = detect_bull_flag(flag_bars(), CFG)
    assert found is not None and not found.broken
    assert found.entry == pytest.approx(123.6) and found.flag_low == pytest.approx(122.5)
    assert found.pole_gain > 0.15


def test_a_close_above_the_flag_high_is_a_broken_flag():
    found = detect_bull_flag(flag_bars(extra=[124.5], extra_vol=[5e6]), CFG)
    assert found is not None and found.broken and found.entry == pytest.approx(123.6)


@pytest.mark.parametrize("kwargs,why", [
    (dict(pole=np.linspace(100, 108, 12), flag=np.array([107.5, 107.0, 106.8, 107.2, 106.9, 107.1])), "pole under 15%"),
    (dict(flag=np.array([123.0, 113.0, 123.2, 113.1, 123.1, 113.2])), "flag range too wide"),
    (dict(flag_vol=4e6), "flag volume heavier than the pole"),
    (dict(flag=np.array([118.0, 112.0, 110.0, 111.0, 110.5, 111.5])), "retraced more than half the pole"),
])
def test_loose_or_loud_pauses_are_not_flags(kwargs, why):
    assert detect_bull_flag(flag_bars(**kwargs), CFG) is None, why


def test_short_history_gives_no_flag():
    assert detect_bull_flag(mk(np.linspace(100, 120, 15)), CFG) is None


# ---------------------------------------------------------------- through evaluate()
def test_evaluate_labels_a_flag_and_gives_it_the_flag_low_as_its_swing_low():
    setup = evaluate("TEST", flag_bars(), spot=123.0, now=NOW, cfg=CFG)
    assert setup.kind == "bull_flag" and setup.flag_broken is False
    assert setup.entry == pytest.approx(123.6)
    assert setup.stop <= 122.5 and setup.stop >= setup.entry - 2 * setup.atr  # at the flag low, within 2 ATR
    assert setup.kind in ACTIONABLE and "bull_flag" in LABELS


def test_a_broken_flag_that_falls_back_under_the_flag_high_fails():
    # 124.5 clears the flag high (123.6) but not the pole top (125.4), so it is a flag break, not a 20-day-high breakout
    bars = flag_bars(extra=[124.5], extra_vol=[5e6])
    assert evaluate("TEST", bars, spot=124.8, now=NOW, cfg=CFG).kind == "bull_flag"
    failed = evaluate("TEST", bars, spot=123.0, now=NOW, cfg=CFG)
    assert failed.kind == "failed" and any("flag high" in n for n in failed.notes)


def test_a_broken_flag_needs_the_breakout_volume_but_a_waiting_flag_does_not():
    waiting = evaluate("TEST", flag_bars(), spot=123.0, now=NOW, cfg=CFG)
    assert apply_filters(waiting, CFG, NOW.date()).excluded_code != "low_volume"
    quiet = evaluate("TEST", flag_bars(extra=[124.5], extra_vol=[1.2e6]), spot=124.8, now=NOW, cfg=CFG)
    assert quiet.kind == "bull_flag" and quiet.flag_broken
    assert apply_filters(quiet, CFG, NOW.date()).excluded_code == "low_volume"


def test_evaluate_labels_an_ma_pullback_with_the_average_as_entry():
    bars = uptrend_with_dip([1, -1.5, -3.0, -4.2, -3.4])
    setup = evaluate("TEST", bars, spot=float(bars["close"].iloc[-1]), now=NOW, cfg=CFG)
    assert setup.kind == "ma_pullback" and setup.entry == pytest.approx(setup.sma50)
    assert any("pulled back to the SMA50" in n for n in setup.notes)


def test_a_flag_break_that_also_clears_the_20_day_high_is_a_plain_breakout():
    bars = flag_bars(extra=[126.0], extra_vol=[5e6])  # above the pole top: breakout takes priority over bull_flag
    assert evaluate("TEST", bars, spot=127.0, now=NOW, cfg=CFG).kind == "breakout"

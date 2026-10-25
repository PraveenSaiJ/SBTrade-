from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tool import indicators as ind


def test_sma_of_a_known_series():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(closes, 3)
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.tolist()[2:] == [2.0, 3.0, 4.0]


def test_ema_follows_the_recurrence_with_alpha_two_over_period_plus_one():
    # period 3 -> alpha 0.5: y0 = 1, y1 = 1.5, y2 = 2.25, y3 = 3.125
    out = ind.ema(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert out.iloc[-1] == pytest.approx(3.125)
    assert out.iloc[:2].isna().all()  # not enough bars yet


def _bars(highs, lows, closes) -> pd.DataFrame:
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_atr_of_a_constant_range_is_that_range():
    n = 30
    out = ind.atr(_bars([102.0] * n, [100.0] * n, [101.0] * n), 10)
    assert out.iloc[-1] == pytest.approx(2.0)


def test_true_range_counts_the_gap_from_the_prior_close():
    bars = _bars([102.0, 110.0], [100.0, 108.0], [100.0, 109.0])  # gaps up 8 above the prior close of 100... high-prev=10
    assert ind.true_range(bars).tolist() == [2.0, 10.0]


def test_rsi_extremes_and_symmetry():
    assert ind.rsi(pd.Series(np.arange(1.0, 40.0)), 14) == 100.0  # only gains
    assert ind.rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), 14) < 1  # only losses
    zigzag = pd.Series([100.0 + (i % 2) for i in range(60)])  # +1, -1, +1, ...
    assert 45 < ind.rsi(zigzag, 14) < 55
    assert ind.rsi(pd.Series([1.0, 2.0]), 14) is None  # too short


def test_rsi_with_period_one_is_100_after_a_rise_and_0_after_a_fall():
    assert ind.rsi(pd.Series([1.0, 2.0]), 1) == 100.0
    assert ind.rsi(pd.Series([2.0, 1.0]), 1) == pytest.approx(0.0)


def test_volume_ratio_is_the_last_bar_over_the_prior_twenty_average():
    assert ind.volume_ratio(pd.Series([10.0] * 20 + [30.0]), 20) == pytest.approx(3.0)
    assert ind.volume_ratio(pd.Series([10.0] * 5), 20) is None
    assert ind.volume_ratio(pd.Series([0.0] * 20 + [5.0]), 20) is None  # no average to compare with


def test_highest_swing_high_ignores_the_unconfirmed_last_bars():
    highs = pd.Series([10, 11, 15, 11, 10, 12, 20, 12, 11, 30.0, 29.0])  # 30 has fewer than 2 bars after it
    assert ind.highest_swing_high(highs, lookback=60, window=2) == 20.0


def test_nearest_resistance_is_the_lowest_level_above_the_fill():
    assert ind.nearest_resistance(100.0, 108.0, 95.0, 120.0) == 108.0
    assert ind.nearest_resistance(100.0, None, 95.0, None) is None


def test_iv_rank_places_today_between_the_low_and_the_high():
    history = list(np.linspace(0.20, 0.60, 100))
    assert ind.iv_rank(0.40, history) == pytest.approx(50.0)
    assert ind.iv_rank(0.70, history) == 100.0  # above the range clips to 100
    assert ind.iv_rank(0.10, history) == 0.0


def test_iv_rank_is_none_without_a_real_history():
    assert ind.iv_rank(0.4, [0.3, 0.5]) is None  # too few days
    assert ind.iv_rank(0.4, [0.3] * 100) is None  # flat history: no range
    assert ind.iv_rank(None, list(np.linspace(0.2, 0.6, 100))) is None


def test_expected_move_formulas():
    assert ind.expected_move(100.0, 0.365, 365) == pytest.approx(36.5)  # one year at 36.5% vol
    assert ind.expected_move(100.0, 0.30, 0) == 0.0
    assert ind.expected_move(200.0, 0.30, 30) == pytest.approx(200 * 0.30 * (30 / 365) ** 0.5)
    assert ind.expected_move_from_straddle(3.20, 2.80) == pytest.approx(6.0)


def test_old_import_paths_still_work():
    from tool.levels import atr, sma
    from tool.screener import highest_swing_high, nearest_resistance, rsi

    assert atr is ind.atr and sma is ind.sma and rsi is ind.rsi
    assert highest_swing_high is ind.highest_swing_high and nearest_resistance is ind.nearest_resistance

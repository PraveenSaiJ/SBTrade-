from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.config import ScreenerConfig, load_config
from tool.levels import NY
from tool.screener import (
    Setup, apply_filters, cap_t1, evaluate, exclusion_reason, flag_earnings, highest_swing_high, nearest_resistance,
    reward_to_risk, rsi, score_setup, targets,
)

TODAY = date(2026, 10, 13)
CFG = ScreenerConfig()
NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)


def make(**kw) -> Setup:
    base = dict(symbol="TEST", kind="breakout", spot=104.0, atr=4.0, sma20=100.0, prior_high=102.0, change_20d=0.05,
                entry=102.0, stop=96.0, t1=110.0, t2=114.0, rr=1.0, volume_ratio=2.0, last_close=104.0, rsi=60.0,
                atr14=4.0, sma50=98.0, sma50_rising=True, sma200=90.0)
    base.update(kw)
    return Setup(**base)


# ---- R:R from the fill
def test_rr_uses_the_fill_not_the_entry():
    assert reward_to_risk(t1=110.0, fill=102.0, stop=96.0) == pytest.approx(8 / 6)
    assert reward_to_risk(t1=110.0, fill=106.0, stop=96.0) == pytest.approx(4 / 10)  # price ran above the entry


def test_targets_default_fill_is_the_entry_and_a_higher_fill_lowers_rr():
    _, _, _, rr_entry = targets(100.0, 4.0, [])
    _, _, _, rr_fill = targets(100.0, 4.0, [], fill=103.0)
    assert rr_entry == pytest.approx(8 / 6) and rr_fill == pytest.approx(5 / 9)


def _breakout_bars(last_close=104.0, volume=2.5e6) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    bars = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1e6}, index=dates)
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [101.0, 105.0, 100.5, last_close, volume]
    return bars


# ---- failed breakout
def test_breakout_that_fell_back_under_the_entry_is_failed_and_excluded_by_default():
    setup = evaluate("TEST", _breakout_bars(), spot=101.0, now=NOW)  # closed 104, trading 101 < entry 102
    assert setup.kind == "failed" and any("failed breakout" in n for n in setup.notes)
    dropped = apply_filters(setup, CFG, TODAY)
    assert dropped.excluded and dropped.excluded.startswith("failed")
    kept = apply_filters(setup, CFG, TODAY, show_failed=True)
    assert kept.excluded is None or not kept.excluded.startswith("failed")


def test_breakout_holding_above_the_entry_is_not_failed():
    assert evaluate("TEST", _breakout_bars(), spot=104.5, now=NOW).kind == "breakout"


# ---- volume
def test_breakout_needs_1_5x_volume():
    assert exclusion_reason(make(volume_ratio=1.2), CFG, TODAY).startswith("low volume")
    assert exclusion_reason(make(volume_ratio=1.5), CFG, TODAY) is None


# ---- earnings
def test_earnings_inside_the_hold_window_excludes_unless_allowed():
    inside = make(earnings=date(2026, 10, 20))  # 5 trading days away
    assert "within the 10-day hold" in exclusion_reason(inside, CFG, TODAY)
    assert exclusion_reason(inside, CFG, TODAY, allow_earnings=True) is None
    beyond = make(earnings=date(2026, 11, 5))  # 17 trading days away: shown, not excluded
    assert exclusion_reason(beyond, CFG, TODAY) is None


def test_expected_move_larger_than_the_stop_excludes_only_when_earnings_are_allowed_inside_the_hold():
    inside = make(earnings=date(2026, 10, 20))  # 5 trading days away; stop distance is 6/102 = 5.9%
    assert "expected move" in exclusion_reason(inside, CFG, TODAY, allow_earnings=True, expected_move_pct=0.09)
    assert exclusion_reason(inside, CFG, TODAY, allow_earnings=True, expected_move_pct=0.04) is None
    beyond = make(earnings=date(2026, 11, 5))  # 17 trading days away: after the hold, so the move is irrelevant
    assert exclusion_reason(beyond, CFG, TODAY, allow_earnings=True, expected_move_pct=0.09) is None


def test_recently_reported_is_flagged_post_earnings_and_not_called_none():
    base = make()
    recent = flag_earnings(base, {}, TODAY, {"TEST": date(2026, 10, 6)}, True, 10)
    assert recent.post_earnings and recent.last_reported == date(2026, 10, 6) and recent.earnings is None
    old = flag_earnings(base, {}, TODAY, {"TEST": date(2026, 9, 1)}, True, 10)
    assert not old.post_earnings
    assert flag_earnings(base, {}, TODAY, {}, calendar_ok=False).earnings_known is False


# ---- trend and overextension
def test_trend_filter_variants():
    assert exclusion_reason(make(last_close=88.0, spot=104.0, sma200=90.0), CFG, TODAY).startswith("below SMA200")
    assert exclusion_reason(make(sma200=None), CFG, TODAY).startswith("below SMA200")
    off = ScreenerConfig(trend_filter="off")
    assert exclusion_reason(make(last_close=88.0, sma200=90.0), off, TODAY) is None
    rising = ScreenerConfig(trend_filter="sma50_rising")
    assert exclusion_reason(make(sma50_rising=False), rising, TODAY).startswith("trend")


def test_overextension_by_rsi_and_by_distance_from_the_entry():
    assert "RSI" in exclusion_reason(make(rsi=80.0), CFG, TODAY)
    assert "ATR(14)" in exclusion_reason(make(spot=111.0, rsi=60.0), CFG, TODAY)  # 9 above entry > 2 x 4
    assert exclusion_reason(make(spot=109.0, rsi=60.0), CFG, TODAY) is None  # 7 <= 8


# ---- T1 capping and resistance
def test_t1_is_capped_at_resistance_and_rr_recomputed():
    t1, rr = cap_t1(t1=112.0, resistance=107.0, fill=102.0, stop=96.0)
    assert t1 == 107.0 and rr == pytest.approx(5 / 6)
    assert cap_t1(t1=105.0, resistance=107.0, fill=102.0, stop=96.0)[0] == 105.0  # already below resistance
    assert cap_t1(t1=112.0, resistance=None, fill=102.0, stop=96.0)[0] == 112.0


def test_nearest_resistance_is_the_lowest_level_above_the_fill():
    assert nearest_resistance(100.0, 108.0, 95.0, 120.0) == 108.0  # SMA200 is below the fill, ignored
    assert nearest_resistance(100.0, None, 95.0, None) is None


def test_highest_swing_high_ignores_unconfirmed_last_bars_and_picks_the_highest_peak():
    highs = pd.Series([10, 11, 15, 11, 10, 12, 20, 12, 11, 30.0, 29.0])  # 30 is in the last window, unconfirmed
    assert highest_swing_high(highs, lookback=60, window=2) == 20.0


def test_evaluate_caps_t1_at_a_strong_resistance_zone_above_the_fill():
    # a flat range at 100 with a ceiling that was hit again and again at 106 (older than the last 25 bars), then a breakout
    dates = pd.bdate_range(end="2026-10-02", periods=260)
    close = np.full(260, 100.0)
    high = close + 2
    high[10:235:8] = 106.0  # repeated rejections at 106: many touches and fractal highs
    bars = pd.DataFrame({"open": close, "high": high, "low": close - 2, "close": close, "volume": 1e6}, index=dates)
    bars.loc[bars.index[-1], ["high", "close", "volume"]] = [105.0, 104.0, 2.5e6]
    setup = evaluate("TEST", bars, spot=104.5, now=NOW)
    assert setup.kind == "breakout" and setup.resistance is not None and setup.resistance > 104.5
    assert 104.5 < setup.resistance <= 106.0  # the bottom edge of the 106 zone
    assert setup.t1 <= setup.resistance + 1e-9 and setup.t2 >= setup.t1
    assert any("T1 capped at the resistance zone" in n for n in setup.notes)


def test_evaluate_has_no_resistance_when_no_zone_sits_above_the_fill():
    dates = pd.bdate_range(end="2026-10-02", periods=260)
    close = np.full(260, 100.0)
    bars = pd.DataFrame({"open": close, "high": close + 2, "low": close - 2, "close": close, "volume": 1e6}, index=dates)
    bars.loc[bars.index[-1], ["high", "close", "volume"]] = [105.0, 104.0, 2.5e6]
    setup = evaluate("TEST", bars, spot=104.5, now=NOW)
    assert setup.kind == "breakout" and setup.resistance is None


# ---- RSI, score, config
def test_rsi_is_100_on_a_straight_climb_and_low_on_a_slide():
    assert rsi(pd.Series(np.arange(1.0, 40.0))) == 100.0
    assert rsi(pd.Series(np.arange(40.0, 1.0, -1.0))) < 5
    assert rsi(pd.Series([1.0, 2.0])) is None


def test_score_rewards_volume_rr_trend_and_penalises_earnings():
    strong = make(volume_ratio=3.0, rr=3.0)
    weak = make(volume_ratio=1.5, rr=1.0)
    assert score_setup(strong, TODAY, CFG) > score_setup(weak, TODAY, CFG)
    assert score_setup(strong, TODAY, CFG) == pytest.approx(0.35 + 0.25 + 0.20 + 0.20)  # full marks
    with_earnings = make(volume_ratio=3.0, rr=3.0, earnings=date(2026, 10, 20))
    assert score_setup(with_earnings, TODAY, CFG) == pytest.approx(0.35 + 0.25 + 0.20)


def test_apply_filters_records_reason_and_score_and_ignores_non_tradable_kinds():
    out = apply_filters(make(volume_ratio=1.0), CFG, TODAY)
    assert out.excluded.startswith("low volume") and out.score is not None
    range_setup = make(kind="range", entry=None)
    assert apply_filters(range_setup, CFG, TODAY) == range_setup


def test_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("screener:\n  min_volume_ratio: 2.0\n  trend_filter: off\n  bogus: 1\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.min_volume_ratio == 2.0 and cfg.trend_filter == "off" and cfg.hold_days == 10
    assert load_config(tmp_path / "missing.yaml") == ScreenerConfig()

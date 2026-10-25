from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from tool.config import LevelConfig, load_all
from tool.indicators import fractal_points, volume_profile
from tool.levels import NY
from tool.sr_levels import (
    Level, Zone, chart_data, compute_zones, fractal_levels, get_levels, merge_levels, moving_average_levels,
    nearest_resistance, nearest_support, oi_levels, prior_month_levels, volume_node_levels,
)

CFG = LevelConfig()
NOW = datetime(2026, 10, 15, 17, 0, tzinfo=NY)


def bars_from(highs, lows, closes=None, volume=1e6, end="2026-10-14") -> pd.DataFrame:
    highs, lows = np.asarray(highs, float), np.asarray(lows, float)
    closes = (highs + lows) / 2 if closes is None else np.asarray(closes, float)
    idx = pd.bdate_range(end=end, periods=len(highs))
    vol = np.full(len(highs), volume) if np.isscalar(volume) else np.asarray(volume, float)
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx)


# ---------------------------------------------------------------- fractals
def test_fractal_points_beat_the_bars_on_each_side_and_skip_the_unconfirmed_tail():
    values = pd.Series([1, 2, 5, 2, 1, 2, 3, 9, 3, 2, 8.0])  # 5 and 9 are swing highs; the 8 at the end has no bars after it
    assert fractal_points(values, 2, "high") == [5.0, 9.0]
    lows = pd.Series([9, 8, 3, 8, 9, 8, 7, 1, 7, 8, 2.0])
    assert fractal_points(lows, 2, "low") == [3.0, 1.0]
    assert fractal_points(pd.Series([1, 2, 3, 4, 5.0]), 1, "high") == []  # a straight climb has none


def test_fractal_levels_are_tagged_and_limited_to_the_lookback():
    bars = bars_from([10, 11, 20, 11, 10, 11, 12, 11, 10, 11], [8, 9, 18, 9, 8, 9, 10, 9, 8, 9])
    labels = {(l.source, l.price) for l in fractal_levels(bars, 2, 10)}
    assert ("swing high", 20.0) in labels
    assert not fractal_levels(bars, 2, 3)  # too short a window to confirm anything


# ---------------------------------------------------------------- the other sources
def test_moving_average_levels_need_enough_bars():
    closes = pd.Series(np.arange(1.0, 121.0))
    assert [l.source for l in moving_average_levels(closes)] == ["SMA50", "SMA100"]
    assert moving_average_levels(closes.iloc[:30]) == []


def test_prior_month_high_and_low():
    idx = pd.bdate_range("2026-09-01", "2026-10-14")
    highs = np.where(idx.month == 9, 100.0, 120.0)
    lows = np.where(idx.month == 9, 90.0, 110.0)
    highs[5], lows[8] = 130.0, 80.0  # September's extremes
    bars = pd.DataFrame({"open": 100.0, "high": highs, "low": lows, "close": 100.0, "volume": 1e6}, index=idx)
    out = {l.source: l.price for l in prior_month_levels(bars, NOW)}
    assert out == {"prior month high": 130.0, "prior month low": 80.0}
    assert prior_month_levels(bars.loc["2026-10-01":], NOW) == []  # no September bars


def test_volume_profile_puts_the_weight_where_the_volume_traded():
    # most days trade 99-101 on huge volume; a few wander far away on little
    highs = np.r_[np.full(50, 101.0), [130.0] * 5]
    lows = np.r_[np.full(50, 99.0), [120.0] * 5]
    vol = np.r_[np.full(50, 5e6), np.full(5, 1e5)]
    bars = bars_from(highs, lows, volume=vol)
    centers, volumes = volume_profile(bars, 40)
    assert abs(centers[int(volumes.argmax())] - 100.0) < 1.5 and volumes.sum() == pytest.approx(vol.sum())
    nodes = volume_node_levels(bars, CFG)
    assert nodes and all(n.source == "volume node" for n in nodes) and abs(nodes[0].price - 100.0) < 1.5
    assert volume_node_levels(bars_from([100, 100, 100], [100, 100, 100]), CFG) == []  # no price range, no profile


def test_open_interest_walls_call_above_put_below_inside_the_band():
    calls = pd.DataFrame({"strike": [95.0, 105.0, 110.0, 114.0, 140.0], "open_interest": [9000, 400, 3000, 1500, 99999]})
    puts = pd.DataFrame({"strike": [86.0, 90.0, 95.0, 105.0, 50.0], "open_interest": [200, 5000, 100, 7000, 99999]})
    out = oi_levels(calls, puts, 100.0, LevelConfig(oi_top_n=2))
    assert [(l.source, l.price) for l in out if l.source == "call wall"] == [("call wall", 110.0), ("call wall", 114.0)]
    assert [(l.source, l.price) for l in out if l.source == "put wall"] == [("put wall", 90.0), ("put wall", 86.0)]
    # 95 calls and 105 puts are on the wrong side; 140 and 50 are outside the 15% band
    assert oi_levels(None, None, 100.0, CFG) == [] and oi_levels(pd.DataFrame(), pd.DataFrame(), 100.0, CFG) == []


# ---------------------------------------------------------------- zones
def test_levels_within_half_an_atr_merge_and_others_stay_apart():
    levels = [Level(100.0, "a"), Level(100.4, "b"), Level(100.6, "c"), Level(105.0, "d")]
    groups = merge_levels(levels, 0.5)
    assert [len(g) for g in groups] == [3, 1]
    assert [len(g) for g in merge_levels(levels, 0.1)] == [1, 1, 1, 1]
    assert merge_levels([], 0.5) == []


def ranged_bars() -> pd.DataFrame:
    """A range around 100 that was rejected at 106 again and again and held at 94 less often."""
    n = 260
    highs, lows = np.full(n, 102.0), np.full(n, 98.0)
    highs[10:235:7] = 106.0  # about 32 rejections at 106
    lows[12:235:30] = 94.0  # a handful of bounces at 94
    return bars_from(highs, lows)


def test_zones_find_the_ceiling_and_the_floor_and_rank_the_busier_one_first():
    zones = compute_zones(ranged_bars(), 100.0, NOW, cfg=CFG)
    ceiling = next(z for z in zones if z.low <= 106.0 <= z.high)
    floor = next(z for z in zones if z.low <= 94.0 <= z.high)
    assert ceiling.kind == "resistance" and floor.kind == "support"
    assert ceiling.touches > floor.touches > 0 and "swing high" in ceiling.sources
    assert zones.index(ceiling) < zones.index(floor)  # more touches ranks first
    assert all(0.0 <= z.score <= 1.0 for z in zones) and [z.score for z in zones] == sorted((z.score for z in zones), reverse=True)
    assert ceiling.label.startswith("zone ") and "swing high" in ceiling.label


def test_strength_gate_and_nearest_zone_lookups():
    zones = [
        Zone(low=93.0, high=95.0, center=94.0, kind="support", sources=("swing low",), touches=6, volume=1e6, score=0.8),
        Zone(low=89.0, high=91.0, center=90.0, kind="support", sources=("SMA200",), touches=1, volume=1e6, score=0.1),
        Zone(low=105.0, high=107.0, center=106.0, kind="resistance", sources=("swing high",), touches=9, volume=1e6, score=0.9),
        Zone(low=103.0, high=104.0, center=103.5, kind="resistance", sources=("SMA50",), touches=1, volume=1e6, score=0.05),
    ]
    assert nearest_support(zones, 100.0).center == 94.0  # the weak zone at 90 is not a candidate even though it is below
    assert nearest_resistance(zones, 100.0).low == 105.0  # the weak 103-104 zone does not cap anything
    assert nearest_support(zones, 100.0, min_score=0.0).center == 94.0 and nearest_resistance(zones, 100.0, min_score=0.0).low == 103.0
    assert nearest_support(zones, 85.0) is None and nearest_resistance(zones, 110.0) is None
    assert nearest_resistance(zones, 106.0) is None  # a price inside a zone is not "below" it


def test_oi_walls_join_a_zone_when_a_chain_is_supplied():
    calls = pd.DataFrame({"strike": [106.0], "open_interest": [50_000]})
    with_oi = compute_zones(ranged_bars(), 100.0, NOW, oi={"calls": calls, "puts": None}, cfg=CFG)
    ceiling = next(z for z in with_oi if z.low <= 106.0 <= z.high)
    assert "call wall" in ceiling.sources
    assert "call wall" not in {s for z in compute_zones(ranged_bars(), 100.0, NOW, cfg=CFG) for s in z.sources}


def test_short_history_gives_no_zones():
    assert compute_zones(bars_from([101] * 20, [99] * 20), 100.0, NOW, cfg=CFG) == []


def test_get_levels_loads_bars_and_the_spot_and_survives_a_missing_chain():
    class Source:
        def daily_bars(self, symbol, lookback_days=90):
            return ranged_bars()

        def spot(self, symbol):
            return 100.0

        def option_chain(self, symbol, expiry=None, right="C", **kw):
            raise ValueError("no chain")

    zones = get_levels("TEST", Source(), NOW)
    assert zones and "call wall" not in {s for z in zones for s in z.sources}  # falls back to price-only zones
    assert [z.score for z in get_levels("TEST", Source(), NOW, with_oi=True)] == [z.score for z in zones]


# ---------------------------------------------------------------- chart data, config
def test_chart_data_has_candles_averages_and_zone_bands():
    bars = ranged_bars()
    zones = compute_zones(bars, 100.0, NOW, cfg=CFG)
    frames = chart_data(bars, zones, days=120, now=NOW)
    assert len(frames["candles"]) == 120 and {"open", "high", "low", "close", "date"} <= set(frames["candles"].columns)
    assert list(frames["averages"].columns) == ["date", "SMA50", "SMA100", "SMA200"] and frames["averages"]["SMA50"].notna().all()
    z = frames["zones"]
    assert len(z) and (z["start"] == frames["candles"]["date"].min()).all() and (z["end"] == frames["candles"]["date"].max()).all()
    assert set(z["kind"]) <= {"support", "resistance"}


def test_level_config_defaults_and_overrides(tmp_path):
    c = LevelConfig()
    assert (c.fractal_window, c.merge_atr, c.profile_days, c.profile_bins, c.oi_top_n, c.min_zone_score) == (3, 0.5, 60, 40, 3, 0.30)
    path = tmp_path / "settings.yaml"
    path.write_text("levels:\n  merge_atr: 0.75\n  min_zone_score: 0.5\n", encoding="utf-8")
    assert load_all(path).levels.merge_atr == 0.75 and load_all(path).levels.hvn_max_nodes == 3


def test_touches_count_separate_visits_not_bars_sitting_inside_a_zone():
    # 60 straight days inside a 99-101 band is ONE visit; a ceiling at 108 tested on 5 separate days is five
    n = 120
    highs, lows = np.full(n, 101.0), np.full(n, 99.0)
    highs[10:110:20] = 108.0
    zones = compute_zones(bars_from(highs, lows), 100.0, NOW, cfg=CFG)
    middle = next(z for z in zones if z.low <= 100.0 <= z.high)
    ceiling = next(z for z in zones if z.low <= 108.0 <= z.high)
    assert middle.touches == 1 and ceiling.touches == 5
    assert middle.volume > ceiling.volume  # but far more traded inside the band

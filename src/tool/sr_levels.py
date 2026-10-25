"""Support and resistance zones from price history (and, on request, option open interest).

Candidate levels come from fractal swing highs and lows, the SMA50/100/200, the prior calendar month's high and low,
high-volume nodes of a 60-day volume profile, and the largest open-interest strikes of the nearest monthly expiry. Levels within
half an ATR of each other merge into a zone; zones are ranked by how often price touched them and the volume on those days.
`get_levels(symbol, source)` is the entry point; Swing (resistance before T1) and the Support puts use the bars-only zones.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from tool.config import LevelConfig, get_config
from tool.indicators import atr, fractal_points, sma, volume_profile
from tool.levels import NY, completed_daily_bars


@dataclass(frozen=True)
class Level:
    price: float
    source: str  # swing high | swing low | SMA50 | SMA100 | SMA200 | prior month high | prior month low | volume node | call wall | put wall


@dataclass(frozen=True)
class Zone:
    low: float
    high: float
    center: float
    kind: str  # support | resistance, by where the centre sits against the price
    sources: tuple[str, ...]
    touches: int  # separate visits: runs of bars in the touch window whose high or low fell inside the zone
    volume: float  # total volume of those bars: how much traded at the zone
    score: float  # 0-1: touches and volume ranked among this symbol's zones

    @property
    def label(self) -> str:
        return f"zone {self.low:.2f}-{self.high:.2f}: " + " + ".join(self.sources)


# ---------------------------------------------------------------- candidate levels
def fractal_levels(completed: pd.DataFrame, window: int, lookback: int) -> list[Level]:
    recent = completed.iloc[-lookback:]
    out = [Level(p, "swing high") for p in fractal_points(recent["high"], window, "high")]
    return out + [Level(p, "swing low") for p in fractal_points(recent["low"], window, "low")]


def moving_average_levels(closes: pd.Series) -> list[Level]:
    return [Level(float(sma(closes, n).iloc[-1]), f"SMA{n}") for n in (50, 100, 200) if len(closes) >= n]


def prior_month_levels(completed: pd.DataFrame, now: datetime) -> list[Level]:
    """High and low of the previous calendar month (empty when the bars do not reach back that far)."""
    today = now.astimezone(NY).date()
    first_this = date(today.year, today.month, 1)
    last_prev = first_this - pd.Timedelta(days=1)
    first_prev = date(last_prev.year, last_prev.month, 1)
    days = completed.index.date if hasattr(completed.index, "date") else completed.index
    month = completed[(days >= first_prev) & (days <= last_prev)]
    if month.empty:
        return []
    return [Level(float(month["high"].max()), "prior month high"), Level(float(month["low"].min()), "prior month low")]


def volume_node_levels(completed: pd.DataFrame, cfg: LevelConfig) -> list[Level]:
    """Up to `hvn_max_nodes` price bins where far more volume traded than in an average bin."""
    centers, volumes = volume_profile(completed.iloc[-cfg.profile_days:], cfg.profile_bins)
    if len(volumes) < 3 or volumes.sum() <= 0:
        return []
    mean = volumes.mean()
    peaks = [i for i in range(len(volumes))
             if volumes[i] >= cfg.hvn_min_ratio * mean
             and volumes[i] >= (volumes[i - 1] if i else -1) and volumes[i] >= (volumes[i + 1] if i + 1 < len(volumes) else -1)]
    peaks.sort(key=lambda i: -volumes[i])
    return [Level(float(centers[i]), "volume node") for i in peaks[: cfg.hvn_max_nodes]]


def oi_levels(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float, cfg: LevelConfig) -> list[Level]:
    """Largest open-interest strikes within `oi_band` of the price: call strikes above it are walls of supply, put strikes
    below it walls of demand."""
    out: list[Level] = []
    lo, hi = spot * (1 - cfg.oi_band), spot * (1 + cfg.oi_band)

    def top(chain: pd.DataFrame | None, above: bool) -> pd.DataFrame:
        if chain is None or chain.empty or "open_interest" not in chain:
            return pd.DataFrame()
        side = chain[(chain["strike"] > spot) if above else (chain["strike"] < spot)]
        side = side[(side["strike"] >= lo) & (side["strike"] <= hi)]
        oi = pd.to_numeric(side["open_interest"], errors="coerce").fillna(0)
        return side.assign(_oi=oi)[oi > 0].sort_values("_oi", ascending=False).head(cfg.oi_top_n)

    out += [Level(float(s), "call wall") for s in top(calls, True)["strike"]] if not top(calls, True).empty else []
    out += [Level(float(s), "put wall") for s in top(puts, False)["strike"]] if not top(puts, False).empty else []
    return out


# ---------------------------------------------------------------- zones
def merge_levels(levels: list[Level], tolerance: float) -> list[list[Level]]:
    """Group levels whose price is within `tolerance` of the group's running mean (sorted by price)."""
    groups: list[list[Level]] = []
    for level in sorted(levels, key=lambda l: l.price):
        if groups:
            mean = sum(l.price for l in groups[-1]) / len(groups[-1])
            if abs(level.price - mean) <= tolerance:
                groups[-1].append(level)
                continue
        groups.append([level])
    return groups


def _ratio(values: list[float]) -> list[float]:
    """Each value over the largest (0 when all are zero): a zone's strength relative to this symbol's strongest."""
    top = max(values)
    return [0.0] * len(values) if top <= 0 else [v / top for v in values]


def build_zones(
    groups: list[list[Level]], completed: pd.DataFrame, spot: float, atr10: float, cfg: LevelConfig
) -> list[Zone]:
    pad = cfg.zone_pad_atr * atr10
    window = completed.iloc[-cfg.touch_lookback:]
    highs, lows, volumes = window["high"].to_numpy(float), window["low"].to_numpy(float), window["volume"].to_numpy(float)
    raw = []
    for g in groups:
        prices = [l.price for l in g]
        low, high = min(prices) - pad, max(prices) + pad
        touched = ((highs >= low) & (highs <= high)) | ((lows >= low) & (lows <= high))
        # a visit is a run of consecutive touching bars: price sitting inside a zone for weeks is one test, not fifty
        visits = int((touched & ~np.r_[False, touched[:-1]]).sum())
        raw.append((low, high, sorted({l.source for l in g}), visits,
                    float(volumes[touched].sum()) if touched.any() else 0.0))
    if not raw:
        return []
    t_norm, v_norm = _ratio([r[3] for r in raw]), _ratio([r[4] for r in raw])
    zones = []
    for (low, high, sources, touches, volume), tn, vn in zip(raw, t_norm, v_norm):
        center = (low + high) / 2
        zones.append(Zone(
            low=low, high=high, center=center, kind="support" if center < spot else "resistance", sources=tuple(sources),
            touches=touches, volume=volume, score=cfg.score_touches * tn + cfg.score_volume * vn,
        ))
    return zones


def compute_zones(
    bars: pd.DataFrame, spot: float, now: datetime | None = None, oi: dict | None = None, cfg: LevelConfig | None = None
) -> list[Zone]:
    """Ranked zones (strongest first) for `bars`. `oi` = {"calls": chain, "puts": chain} adds the open-interest walls."""
    cfg = cfg or get_config().levels
    now = now or datetime.now(NY)
    completed = completed_daily_bars(bars, now)
    if len(completed) < 30:
        return []
    atr10 = float(atr(completed, 10).iloc[-1])
    if not atr10 or np.isnan(atr10):
        return []
    levels = fractal_levels(completed, cfg.fractal_window, cfg.fractal_lookback)
    levels += moving_average_levels(completed["close"]) + prior_month_levels(completed, now) + volume_node_levels(completed, cfg)
    if oi:
        levels += oi_levels(oi.get("calls"), oi.get("puts"), spot, cfg)
    zones = build_zones(merge_levels(levels, cfg.merge_atr * atr10), completed, spot, atr10, cfg)
    return sorted(zones, key=lambda z: (-z.score, -len(z.sources), abs(z.center - spot)))


def nearest_support(zones: list[Zone], price: float, min_score: float | None = None) -> Zone | None:
    """The strongest-enough support zone closest under `price`."""
    min_score = get_config().levels.min_zone_score if min_score is None else min_score
    below = [z for z in zones if z.center < price and z.score >= min_score]
    return max(below, key=lambda z: z.center) if below else None


def nearest_resistance(zones: list[Zone], price: float, min_score: float | None = None) -> Zone | None:
    """The strongest-enough zone whose bottom edge is above `price`, nearest first."""
    min_score = get_config().levels.min_zone_score if min_score is None else min_score
    above = [z for z in zones if z.low > price and z.score >= min_score]
    return min(above, key=lambda z: z.low) if above else None


def get_levels(symbol: str, source, now: datetime | None = None, with_oi: bool = False, cfg: LevelConfig | None = None) -> list[Zone]:
    """Zones for a symbol from cached daily bars and the live price; `with_oi` adds the nearest monthly's open-interest walls."""
    now = now or datetime.now(NY)
    bars = source.daily_bars(symbol, lookback_days=400)
    spot = source.spot(symbol)
    oi = None
    if with_oi:
        from tool.expiries import monthly_expiry

        monthly = monthly_expiry(now.date())
        try:
            oi = {
                "calls": source.option_chain(symbol, expiry=monthly, right="C", strike_count=40),
                "puts": source.option_chain(symbol, expiry=monthly, right="P", strike_count=40),
            }
        except Exception:
            oi = None  # no chain: bars-only zones
    return compute_zones(bars, spot, now, oi, cfg)


def chart_data(bars: pd.DataFrame, zones: list[Zone], days: int = 120, now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """Frames for the price-and-zones chart: candles for the last `days` completed bars, SMA50/100/200 lines, and the zones
    as bands spanning the chart's date range."""
    completed = completed_daily_bars(bars, now or datetime.now(NY))
    window = completed.iloc[-days:]
    dates = pd.to_datetime(window.index.date)
    candles = pd.DataFrame({
        "date": dates, "open": window["open"].to_numpy(float), "high": window["high"].to_numpy(float),
        "low": window["low"].to_numpy(float), "close": window["close"].to_numpy(float),
    })
    averages = pd.DataFrame({"date": dates})
    for n in (50, 100, 200):
        averages[f"SMA{n}"] = sma(completed["close"], n).iloc[-days:].to_numpy(float)
    first, last = dates.min(), dates.max()
    zone_frame = pd.DataFrame([
        {"low": z.low, "high": z.high, "kind": z.kind, "score": z.score, "touches": z.touches, "label": z.label,
         "start": first, "end": last}
        for z in zones if window["low"].min() * 0.9 <= z.center <= window["high"].max() * 1.1
    ], columns=["low", "high", "kind", "score", "touches", "label", "start", "end"])
    return {"candles": candles, "averages": averages, "zones": zone_frame}

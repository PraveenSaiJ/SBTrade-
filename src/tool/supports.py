"""Support levels under the price: where a cash-secured put's strike is allowed to sit.

These are the strong support zones from sr_levels (fractal lows, SMA50/100/200, the prior month's low, volume nodes); the
zone's centre is the level and its sources are the label. Pivot points are not sources here (the guardrail puts still use
weekly S1, and the Levels tab shows pivots). Recommend-only.
"""

from __future__ import annotations

import pandas as pd

from tool.config import PutConfig, get_config
from tool.levels import LevelsResult
from tool.sr_levels import compute_zones


def support_levels(
    bars: pd.DataFrame, spot: float, levels: LevelsResult | None = None, cfg: PutConfig | None = None, now=None
) -> list[tuple[float, str]]:
    """(price, label) for every strong support zone under `spot`, highest first. `levels` and `cfg` are kept for callers."""
    zones = [z for z in compute_zones(bars, spot, now) if z.kind == "support" and z.score >= get_config().levels.min_zone_score]
    return sorted(((z.center, z.label) for z in zones), reverse=True)


def nearest_support(levels: list[tuple[float, str]]) -> tuple[float, str] | None:
    """The support closest under the price (the list is sorted highest first)."""
    return levels[0] if levels else None

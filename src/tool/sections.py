"""Sort option recommendations into display sections by |delta| (a proxy for chance of profit when selling)."""

from __future__ import annotations

import math
from tool.config import get_config

HIGH_MAX = get_config().options.bucket_high_max
MODERATE_MAX = get_config().options.bucket_moderate_max


def bucket(delta: float | None) -> str | None:
    """'high' (|delta| <= 0.20), 'moderate' (<= 0.35), 'lotto' (above), or None when delta is missing."""
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return None
    d = abs(delta)
    if d <= HIGH_MAX:
        return "high"
    if d <= MODERATE_MAX:
        return "moderate"
    return "lotto"


def chance_of_profit(delta: float | None) -> float | None:
    """Rough probability the sold option expires worthless: 1 - |delta|."""
    if bucket(delta) is None:
        return None
    return 1 - abs(delta)

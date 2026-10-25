"""Company size from market capitalisation."""

from __future__ import annotations

import pandas as pd

BUCKETS = ((200e9, "Mega"), (10e9, "Large"), (2e9, "Mid"), (300e6, "Small"), (0.0, "Micro"))


def size_bucket(market_cap: float | None) -> str | None:
    if market_cap is None or pd.isna(market_cap) or market_cap <= 0:
        return None
    return next(name for floor, name in BUCKETS if market_cap >= floor)


def money_short(value: float) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= scale:
            digits = 2 if value / scale < 10 else 1 if value / scale < 100 else 0
            return f"${value / scale:.{digits}f}{suffix}"
    return f"${value:,.0f}"


def size_label(market_cap: float | None) -> str | None:
    """'Large · $150B' (None when the market cap is unknown, e.g. ETFs)."""
    bucket = size_bucket(market_cap)
    return None if bucket is None else f"{bucket} · {money_short(float(market_cap))}"

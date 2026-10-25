from __future__ import annotations

import pytest

from tool.sections import bucket, chance_of_profit


@pytest.mark.parametrize(
    "delta, expected",
    [(0.07, "high"), (-0.20, "high"), (0.21, "moderate"), (-0.35, "moderate"), (0.36, "lotto"), (-0.60, "lotto"),
     (None, None), (float("nan"), None)],
)
def test_bucket_by_absolute_delta(delta, expected):
    assert bucket(delta) == expected


def test_chance_of_profit_is_one_minus_abs_delta():
    assert chance_of_profit(-0.25) == pytest.approx(0.75)
    assert chance_of_profit(None) is None

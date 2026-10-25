"""Setup detectors beyond the breakout: a pullback to the 20/50-day average in an uptrend, and the bull flag.

Pure functions over completed daily bars (open/high/low/close/volume). Thresholds live in ScreenerConfig.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tool.config import ScreenerConfig
from tool.indicators import sma


@dataclass(frozen=True)
class MaPullback:
    entry: float  # the moving average that was touched
    average: str  # "SMA20" or "SMA50"


@dataclass(frozen=True)
class BullFlag:
    entry: float  # the flag high: a close above it is the breakout
    flag_low: float
    pole_gain: float
    flag_bars: int
    broken: bool  # the last completed close is already above the flag high


def _num(x: object) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def detect_ma_pullback(completed: pd.DataFrame, atr10: float, cfg: ScreenerConfig) -> MaPullback | None:
    """An uptrend that dipped to its SMA20 (or SMA50) and held.

    Uptrend: last close above SMA200, SMA50 above SMA200 and rising. Touch: a low in the last `ma_touch_bars` bars came
    within `ma_touch_atr` x ATR of the average and the last close is back at or above it. The stock must also sit at least
    `ma_min_pullback_pct` under its 20-day high, so a stock still at its highs is not called a pullback.
    """
    closes = completed["close"]
    if len(closes) < 211 or atr10 <= 0:
        return None
    s20, s50, s200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    close, a20, a50, a200 = float(closes.iloc[-1]), _num(s20.iloc[-1]), _num(s50.iloc[-1]), _num(s200.iloc[-1])
    if None in (a20, a50, a200) or not (close > a200 and a50 > a200 and a50 > float(s50.iloc[-11])):
        return None
    high20 = float(completed["high"].iloc[-20:].max())
    if (high20 - close) / high20 < cfg.ma_min_pullback_pct:
        return None
    recent_low = float(completed["low"].iloc[-cfg.ma_touch_bars:].min())
    for name, value in (("SMA20", a20), ("SMA50", a50)):
        if recent_low <= value + cfg.ma_touch_atr * atr10 and close >= value:
            return MaPullback(entry=value, average=name)
    return None


def _pole(pole_bars: pd.DataFrame, cfg: ScreenerConfig) -> tuple[float, float, float] | None:
    """(low, high, gain) of a pole: a rise of at least `flag_pole_pct` whose low comes before its high."""
    lows, highs = pole_bars["low"].to_numpy(dtype=float), pole_bars["high"].to_numpy(dtype=float)
    low_i, high_i = int(np.argmin(lows)), int(np.argmax(highs))
    if low_i >= high_i or lows[low_i] <= 0:
        return None
    gain = highs[high_i] / lows[low_i] - 1
    return (float(lows[low_i]), float(highs[high_i]), float(gain)) if gain >= cfg.flag_pole_pct else None


def detect_bull_flag(completed: pd.DataFrame, cfg: ScreenerConfig) -> BullFlag | None:
    """A sharp rise (the pole) followed by a short, tight, quiet pause (the flag), just under or just through its high.

    Pole: at least `flag_pole_pct` within `flag_pole_bars` bars. Flag: `flag_min_bars` to `flag_max_bars` bars whose range is
    at most `flag_max_range_pct`, giving back at most `flag_max_retrace` of the pole, on lower average volume than the pole.
    The flag is looked for first as the bars before the latest one (so a close above its high shows as `broken`), then
    including the latest bar (a flag still waiting to break).
    """
    n = len(completed)
    if n < cfg.flag_pole_bars + cfg.flag_max_bars + 2:
        return None
    for f in range(cfg.flag_min_bars, cfg.flag_max_bars + 1):
        for include_latest in (False, True):
            end = n if include_latest else n - 1
            flag = completed.iloc[end - f:end]
            pole = completed.iloc[max(0, end - f - cfg.flag_pole_bars):end - f]
            if len(flag) < f or len(pole) < 5:
                continue
            flag_high, flag_low = float(flag["high"].max()), float(flag["low"].min())
            if (flag_high - flag_low) / flag_high > cfg.flag_max_range_pct:
                continue
            found = _pole(pole, cfg)
            if found is None:
                continue
            pole_low, pole_high, gain = found
            if flag_high > pole_high * 1.02 or (pole_high - flag_low) / (pole_high - pole_low) > cfg.flag_max_retrace:
                continue
            if float(flag["volume"].mean()) >= float(pole["volume"].mean()):
                continue  # a flag pauses on lighter volume than the move that made it
            close = float(completed["close"].iloc[-1])
            broken = (not include_latest) and close > flag_high
            if not include_latest and not broken:
                continue  # prior-bars flag only counts when the latest bar broke out of it
            return BullFlag(entry=flag_high, flag_low=flag_low, pole_gain=gain, flag_bars=f, broken=broken)
    return None

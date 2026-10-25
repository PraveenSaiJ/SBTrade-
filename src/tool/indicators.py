"""Technical indicators in one place (pure pandas/numpy; no data access).

levels.py, screener.py, longterm.py and the engines import from here instead of re-deriving SMA/RSI/ATR.
Wilder smoothing (alpha = 1/period) is used for ATR and RSI, matching CLAUDE.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_IV_HISTORY = 60  # IV rank needs a real history; below this it returns None instead of a made-up number


def sma(closes: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of closes (NaN until `period` values exist)."""
    return closes.rolling(window=period).mean()


def ema(closes: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average with the standard span smoothing (alpha = 2 / (period + 1))."""
    return closes.ewm(span=period, min_periods=period, adjust=False).mean()


def true_range(daily: pd.DataFrame) -> pd.Series:
    prev_close = daily["close"].shift(1)
    return pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(daily: pd.DataFrame, period: int = 10) -> pd.Series:
    """Average True Range (Wilder smoothing) over `period` bars."""
    return true_range(daily).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI for every bar (NaN until `period` changes exist)."""
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna() & loss.notna())  # no losses at all -> 100


def rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Wilder RSI of the last close, None when there are not enough bars."""
    if len(closes) <= period:
        return None
    value = rsi_series(closes, period).iloc[-1]
    return None if pd.isna(value) else float(value)


def volume_ratio(volume: pd.Series, window: int = 20) -> float | None:
    """Last bar's volume divided by the average of the `window` bars before it (None when unavailable)."""
    if len(volume) < window + 1:
        return None
    average = float(volume.iloc[-window - 1:-1].mean())
    return float(volume.iloc[-1]) / average if average > 0 else None


def highest_swing_high(highs: pd.Series, lookback: int = 60, window: int = 2) -> float | None:
    """Highest swing high in the last `lookback` bars. A swing high beats `window` bars on each side, so the last
    `window` bars (not yet confirmed) are never swing highs."""
    recent = highs.iloc[-lookback:].to_numpy(dtype=float)
    best = None
    for i in range(window, len(recent) - window):
        around = np.concatenate([recent[i - window:i], recent[i + 1:i + 1 + window]])
        if recent[i] > around.max() and (best is None or recent[i] > best):
            best = float(recent[i])
    return best


def recent_swing_low(
    lows: pd.Series, below: float, lookback: int = 20, window: int = 2, fallback_bars: int = 10
) -> float | None:
    """The latest confirmed swing low under `below` in the last `lookback` bars (a low beating `window` bars on each side).
    With none, the lowest low of the last `fallback_bars` bars if that is under `below`; else None."""
    recent = lows.iloc[-lookback:].to_numpy(dtype=float)
    for i in range(len(recent) - window - 1, window - 1, -1):  # newest confirmed bar first
        around = np.concatenate([recent[i - window:i], recent[i + 1:i + 1 + window]])
        if recent[i] < around.min() and recent[i] < below:
            return float(recent[i])
    tail = float(lows.iloc[-fallback_bars:].min()) if len(lows) else float("nan")
    return tail if tail < below else None


def fractal_points(values: pd.Series, window: int, kind: str) -> list[float]:
    """Swing highs (`kind="high"`) or lows (`"low"`): values that beat `window` bars on each side. The last `window` bars
    are never swing points (not yet confirmed)."""
    arr = values.to_numpy(dtype=float)
    out = []
    for i in range(window, len(arr) - window):
        around = np.concatenate([arr[i - window:i], arr[i + 1:i + 1 + window]])
        if (kind == "high" and arr[i] > around.max()) or (kind == "low" and arr[i] < around.min()):
            out.append(float(arr[i]))
    return out


def volume_profile(bars: pd.DataFrame, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(bin centres, volume per bin): each bar's volume spread evenly over the part of its high-low range in each bin."""
    low, high = float(bars["low"].min()), float(bars["high"].max())
    if high <= low or bins < 1:
        return np.array([]), np.array([])
    edges = np.linspace(low, high, bins + 1)
    volume = np.zeros(bins)
    for h, l, v in zip(bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["volume"].to_numpy(float)):
        if h <= l:
            volume[min(np.searchsorted(edges, l, side="right") - 1, bins - 1)] += v
            continue
        overlap = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0, None)
        volume += v * overlap / overlap.sum()
    return (edges[:-1] + edges[1:]) / 2, volume


def nearest_resistance(
    fill: float, sma100: float | None, sma200: float | None, swing_high: float | None
) -> float | None:
    """The lowest of SMA100, SMA200 and the highest recent swing high that sits above the fill."""
    above = [x for x in (sma100, sma200, swing_high) if x is not None and not np.isnan(x) and x > fill]
    return min(above) if above else None


def iv_rank(current_iv: float | None, history: list[float] | pd.Series) -> float | None:
    """Where today's IV sits between its low and high over the history, 0-100. None without a real history."""
    values = [float(v) for v in history if v is not None and not pd.isna(v)]
    if current_iv is None or pd.isna(current_iv) or len(values) < MIN_IV_HISTORY:
        return None
    low, high = min(values), max(values)
    if high <= low:
        return None
    return float(max(0.0, min(100.0, (current_iv - low) / (high - low) * 100)))


def hv_series(closes: pd.Series, window: int = 20) -> pd.Series:
    """Annualised realized volatility of daily log returns over a rolling `window` (NaN until it fills)."""
    returns = np.log(closes / closes.shift(1))
    return returns.rolling(window=window).std() * math.sqrt(252)


def hv_rank(closes: pd.Series, window: int = 20, lookback: int = 252, min_samples: int = 60) -> float | None:
    """Where today's realized volatility sits in its own trailing range, 0-100. A stand-in for IV rank until enough
    implied-vol history exists; None when the history is too short or perfectly flat."""
    series = hv_series(closes, window).dropna().iloc[-lookback:]
    if len(series) < min_samples:
        return None
    low, high = float(series.min()), float(series.max())
    if high <= low:
        return None
    return float((series.iloc[-1] - low) / (high - low) * 100)


def expected_move(spot: float, iv: float, days: float) -> float:
    """One-standard-deviation move to expiry from implied vol: spot * iv * sqrt(days / 365)."""
    return spot * iv * math.sqrt(max(days, 0) / 365)


def expected_move_from_straddle(call_mid: float, put_mid: float) -> float:
    """The market's expected move is the at-the-money straddle price (call mid + put mid)."""
    return call_mid + put_mid

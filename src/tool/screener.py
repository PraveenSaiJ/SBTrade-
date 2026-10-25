"""Swing screener — setup classification, targets, and sell signals.

See CLAUDE.md "Swing screener rules". Recommend-only; nothing here places orders.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tool.config import ScreenerConfig, get_config
from tool.indicators import (  # noqa: F401
    highest_swing_high, nearest_resistance, recent_swing_low, rsi, volume_ratio as _volume_ratio,
)
from tool.setups import detect_bull_flag, detect_ma_pullback
from tool.sr_levels import compute_zones, nearest_resistance as nearest_resistance_zone
from tool.levels import LevelsResult, atr, compute_levels, completed_daily_bars, prior_n_day_high, sma

CONSTITUENTS_PATH = Path("config/constituents.csv")
ACTIONABLE = ("breakout", "bull_flag", "ma_pullback", "pullback", "extended")
RANK = {"breakout": 0, "bull_flag": 1, "ma_pullback": 1, "pullback": 2, "extended": 3, "downtrend": 4, "range": 4,
        "weak_breakout": 4, "failed": 4}
LABELS = {"breakout": "Breakout", "bull_flag": "Bull flag", "ma_pullback": "MA pullback", "pullback": "Pullback (pivot)",
          "extended": "Extended", "failed": "Failed breakout", "weak_breakout": "Weak breakout"}
_S = get_config().screener
STOP_ATR = _S.stop_atr  # stop = entry - STOP_ATR x ATR(10)
EARNINGS_WINDOW_DAYS = get_config().screener.earnings_window_days  # the upcoming report is looked up this far ahead


@dataclass(frozen=True)
class Setup:
    symbol: str
    kind: str
    spot: float
    atr: float
    sma20: float
    prior_high: float
    change_20d: float
    entry: float | None = None
    stop: float | None = None
    t1: float | None = None
    t2: float | None = None
    rr: float | None = None
    volume_ratio: float | None = None  # breakout-day volume / average of the prior 20 sessions
    earnings: date | None = None
    notes: list[str] = field(default_factory=list)
    last_close: float | None = None  # last COMPLETED daily close
    data_as_of: date | None = None  # date of the last completed bar the setup is built on
    rsi: float | None = None  # RSI(14)
    atr14: float | None = None
    sma50: float | None = None
    sma50_rising: bool | None = None
    sma200: float | None = None
    resistance: float | None = None  # nearest overhead resistance above the fill
    score: float | None = None
    excluded: str | None = None  # why the scanner dropped it (None = recommended)
    excluded_code: str | None = None  # failed | low_volume | earnings_hold | expected_move | trend | overextended
    flag_broken: bool | None = None  # bull flag only: the last close is already above the flag high
    shares: int | None = None  # position size at risk_pct of the account (None until account_size is set)
    position_value: float | None = None
    risk_dollars: float | None = None
    last_reported: date | None = None
    earnings_hour: str | None = None  # bmo | amc | dmh for the upcoming report
    last_hour: str | None = None  # the same for the last report
    expected_move: float | None = None  # ATM straddle ($) to the first expiry after the next report
    expected_move_pct: float | None = None  # expected_move / spot
    em_expiry: date | None = None
    post_earnings: bool = False  # reported within the last `post_earnings_days` trading days
    earnings_known: bool = True  # False when the earnings calendar could not be loaded


@dataclass(frozen=True)
class SellSignal:
    symbol: str
    kind: str
    detail: str


def pivot_levels(levels: LevelsResult) -> list[float]:
    out: list[float] = []
    for p in (levels.daily.pivots, levels.weekly.pivots):
        out += [p.p, p.r1, p.r2, p.r3, p.s1, p.s2, p.s3]
    return sorted(out)


def classify_setup(
    spot: float,
    change_20d: float,
    sma20: float,
    atr: float,
    prior_high: float,
    pivots: list[float],
    last_close: float | None = None,
    volume_ratio: float | None = None,
) -> tuple[str, float | None]:
    """Return (kind, entry). Order matches CLAUDE.md: downtrend, breakout, extended, pullback, range.

    A breakout needs the last COMPLETED daily close above the prior 20-day high (not just a last trade), and
    volume above the 20-day average on that day. A close without the volume is a "weak_breakout" (not actionable).
    """
    if change_20d < -_S.downtrend_pct:
        return "downtrend", None
    confirmed_close = spot if last_close is None else last_close
    if confirmed_close > prior_high:
        if volume_ratio is not None and volume_ratio < _S.weak_breakout_volume:
            return "weak_breakout", prior_high
        return "breakout", prior_high
    if atr > 0 and (spot - sma20) / atr > _S.extended_atr:
        below = [l for l in pivots if l <= spot - atr]
        return "extended", (max(below) if below else None)
    supports = [l for l in pivots if l < spot and spot - l <= atr]
    if spot > sma20 and change_20d > _S.pullback_min_change and supports:
        return "pullback", max(supports)
    return "range", None


def reward_to_risk(t1: float, fill: float, stop: float) -> float | None:
    """(T1 - fill) / (fill - stop), where fill is where you would actually buy: max(spot, entry)."""
    risk = fill - stop
    return (t1 - fill) / risk if risk > 0 else None


def structural_stop(entry: float, atr: float, swing_low: float | None, cfg: ScreenerConfig | None = None) -> float:
    """Below the structure with a buffer, but never wider than `stop_max_atr` ATR under the entry.

    stop = max( min(entry - buffer x ATR, swing low), entry - max x ATR ). With no swing low to lean on it falls back
    to entry - stop_atr x ATR.
    """
    cfg = cfg or _S
    if swing_low is None:
        return entry - cfg.stop_atr * atr
    return max(min(entry - cfg.stop_buffer_atr * atr, swing_low), entry - cfg.stop_max_atr * atr)


def position_size(fill: float, stop: float, account_size: float | None, risk_pct: float) -> tuple[int, float, float] | None:
    """(shares, position value, dollars at risk): risk_pct of the account over the stop distance, rounded down and
    capped so the position fits in the account. None until an account size is configured."""
    if account_size is None or account_size <= 0 or fill <= stop:
        return None
    shares = min(int(risk_pct * account_size // (fill - stop)), int(account_size // fill))
    return shares, shares * fill, shares * (fill - stop)


def targets(
    entry: float, atr: float, pivots: list[float], fill: float | None = None, stop: float | None = None
) -> tuple[float, float, float, float]:
    """Return (stop, t1, t2, reward_to_risk). `fill` defaults to the entry (a limit order at the entry); `stop`
    defaults to entry - STOP_ATR x ATR."""
    stop = entry - STOP_ATR * atr if stop is None else stop
    t1_candidates = [l for l in pivots if l >= entry + _S.t1_atr * atr]
    t1 = min(t1_candidates) if t1_candidates else entry + _S.t1_fallback_atr * atr
    t2_candidates = [l for l in pivots if l >= t1 + _S.t2_atr * atr]
    t2 = min(t2_candidates) if t2_candidates else t1 + _S.t2_fallback_atr * atr
    rr = reward_to_risk(t1, entry if fill is None else fill, stop)
    return stop, t1, t2, rr


def cap_t1(t1: float, resistance: float | None, fill: float, stop: float) -> tuple[float, float | None]:
    """Cap T1 at the nearest overhead resistance and recompute R:R. Returns (t1, rr)."""
    if resistance is not None and resistance > fill and t1 > resistance:
        t1 = resistance
    return t1, reward_to_risk(t1, fill, stop)


def _last(series: pd.Series) -> float | None:
    value = series.iloc[-1] if len(series) else np.nan
    return None if pd.isna(value) else float(value)


def evaluate(
    symbol: str, bars: pd.DataFrame, spot: float, now: datetime | None = None, cfg: ScreenerConfig | None = None,
    account=None,
) -> Setup:
    cfg = cfg or ScreenerConfig()
    account = account if account is not None else get_config().account
    levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
    completed = completed_daily_bars(bars, now)
    closes = completed["close"]
    if len(closes) < 21:
        raise ValueError("need at least 21 completed daily bars for the 20-day change")
    change_20d = spot / float(closes.iloc[-21]) - 1
    last_close = float(closes.iloc[-1])
    volume_ratio = _volume_ratio(completed["volume"], 20) if "volume" in completed else None
    pivots = pivot_levels(levels)
    kind, entry = classify_setup(
        spot, change_20d, levels.sma20, levels.atr10, levels.prior_20d_high, pivots, last_close, volume_ratio
    )
    notes: list[str] = []
    flag = None
    flag_broken = None
    swing_low_override = None
    if kind not in ("downtrend", "breakout", "weak_breakout") and levels.atr10 > 0:
        flag = detect_bull_flag(completed, cfg)
        if flag is not None:
            kind, entry, flag_broken, swing_low_override = "bull_flag", flag.entry, flag.broken, flag.flag_low
            notes.append(
                f"bull flag: a {flag.pole_gain:.0%} pole, then {flag.flag_bars} quiet bars below ${flag.entry:.2f}"
                + ("; closed above the flag high" if flag.broken else "; waiting to break out")
            )
        else:
            pullback = detect_ma_pullback(completed, levels.atr10, cfg)
            if pullback is not None:
                kind, entry = "ma_pullback", pullback.entry
                notes.append(f"pulled back to the {pullback.average} (${pullback.entry:.2f}) in an uptrend and held")
    if kind == "breakout" and entry is not None and spot < entry:
        # the close was above the high, but the live price has already fallen back under it
        notes.append(f"failed breakout: spot ${spot:.2f} is back under the breakout level ${entry:.2f}")
        kind = "failed"
    elif kind == "bull_flag" and flag_broken and entry is not None and spot < entry:
        notes.append(f"failed breakout: spot ${spot:.2f} is back under the flag high ${entry:.2f}")
        kind = "failed"
    if kind != "breakout" and spot > levels.prior_20d_high >= last_close:
        notes.append(
            f"trading above the 20-day high ${levels.prior_20d_high:.2f}, but not confirmed: needs a daily close above it"
        )
    if kind == "weak_breakout":
        notes.append(
            f"closed above ${levels.prior_20d_high:.2f} on only {volume_ratio:.1f}x average volume; needs more than 1.0x"
        )
    setup = Setup(
        symbol=symbol,
        kind=kind,
        spot=spot,
        atr=levels.atr10,
        sma20=levels.sma20,
        prior_high=levels.prior_20d_high,
        change_20d=change_20d,
        volume_ratio=volume_ratio,
        notes=notes,
        last_close=last_close,
        data_as_of=completed.index[-1].date(),
        rsi=rsi(closes, 14),
        atr14=_last(atr(completed, 14)),
        sma50=_last(sma(closes, 50)),
        sma50_rising=(
            bool(sma(closes, 50).iloc[-1] > sma(closes, 50).iloc[-11]) if len(closes) >= 61 else None
        ),
        sma200=_last(sma(closes, 200)),
        flag_broken=flag_broken,
    )
    if entry is None or levels.atr10 <= 0:
        return setup
    fill = max(spot, entry)
    swing_low = swing_low_override if swing_low_override is not None else recent_swing_low(
        completed["low"], entry, cfg.swing_low_lookback, cfg.swing_window, cfg.swing_low_fallback_bars
    )
    stop, t1, t2, _ = targets(entry, levels.atr10, pivots, fill, stop=structural_stop(entry, levels.atr10, swing_low, cfg))
    zone = nearest_resistance_zone(compute_zones(bars, spot, now), fill)
    resistance = zone.low if zone is not None else None
    if zone is not None and t1 > zone.low:
        setup = replace(setup, notes=[*setup.notes, f"T1 capped at the resistance {zone.label}"])
    t1, rr = cap_t1(t1, resistance, fill, stop)
    t2 = max(t2, t1)
    sized = position_size(fill, stop, account.account_size, account.risk_pct)
    return replace(
        setup, entry=entry, stop=stop, t1=t1, t2=t2, rr=rr, resistance=resistance,
        shares=None if sized is None else sized[0], position_value=None if sized is None else sized[1],
        risk_dollars=None if sized is None else sized[2],
    )


def trading_days_between(start: date, end: date) -> int:
    """Weekdays from `start` up to (not including) `end`. Holidays are not modelled."""
    return int(np.busday_count(start, end))


def _date_hour(value: object) -> tuple[date | None, str | None]:
    """Calendar values are a date or a (date, hour) pair."""
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, None


def flag_earnings(
    setup: Setup,
    earnings_by_symbol: dict,
    today: date,
    reported_by_symbol: dict | None = None,
    calendar_ok: bool = True,
    post_earnings_days: int = 10,
) -> Setup:
    """Record the next report (within the lookup window) and a report in the last `post_earnings_days` trading days.

    Maps hold a date or a (date, bmo/amc/dmh) pair per symbol. `calendar_ok` is False when the earnings calendar
    could not be loaded, so a missing date means "unknown" rather than "none".
    """
    setup = replace(setup, earnings_known=calendar_ok)
    sym = setup.symbol
    alt = sym.replace("/", ".")
    reported, last_hour = _date_hour((reported_by_symbol or {}).get(sym) or (reported_by_symbol or {}).get(alt))
    if reported is not None and reported <= today and trading_days_between(reported, today) <= post_earnings_days:
        setup = replace(
            setup, last_reported=reported, last_hour=last_hour, post_earnings=True,
            notes=[*setup.notes, f"reported earnings {reported.isoformat()}: post-earnings, watch for drift or gap"],
        )
    elif reported is not None and reported <= today:
        setup = replace(setup, last_reported=reported, last_hour=last_hour)
    when, hour = _date_hour(earnings_by_symbol.get(sym) or earnings_by_symbol.get(alt))
    if when is None or not 0 <= (when - today).days <= EARNINGS_WINDOW_DAYS:
        return setup
    days = (when - today).days
    return replace(
        setup, earnings=when, earnings_hour=hour,
        notes=[*setup.notes, f"EARNINGS {when.isoformat()} in {days} days: inside the trade window"],
    )


def earnings_label(
    next_date: date | None, next_hour: str | None, last_date: date | None, last_hour: str | None,
    post_earnings: bool, known: bool, today: date,
) -> str:
    """'Reported Oct 15 (amc)' within 10 trading days of a report, else 'Oct 28 (in 6d, bmo)', else 'None'.
    'Unknown' means the calendar could not be loaded, which is not the same as no report."""
    if post_earnings and last_date is not None:
        return f"Reported {last_date:%b %d}" + (f" ({last_hour})" if last_hour else "")
    if next_date is not None:
        return f"{next_date:%b %d} (in {(next_date - today).days}d" + (f", {next_hour})" if next_hour else ")")
    return "None" if known else "Unknown"


def earnings_risk(setup: Setup, today: date, cfg: ScreenerConfig) -> float:
    """0 (none) to 1 (earnings inside the hold window). Unknown (calendar not loaded) counts as a quarter."""
    if setup.earnings is not None:
        if trading_days_between(today, setup.earnings) < cfg.hold_days:
            return 1.0
        return 0.5 if (setup.earnings - today).days <= cfg.earnings_flag_days else 0.25
    if not setup.earnings_known:
        return 0.25
    return 0.0


def trend_score(setup: Setup) -> float:
    """Share of four trend checks that pass: above SMA50, above SMA200, SMA50 above SMA200, SMA50 rising."""
    c = setup.last_close
    checks = [
        c is not None and setup.sma50 is not None and c > setup.sma50,
        c is not None and setup.sma200 is not None and c > setup.sma200,
        setup.sma50 is not None and setup.sma200 is not None and setup.sma50 > setup.sma200,
        bool(setup.sma50_rising),
    ]
    return sum(checks) / len(checks)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_setup(setup: Setup, today: date, cfg: ScreenerConfig) -> float:
    """0.35*norm(volume) + 0.25*norm(R:R) + 0.20*trend + 0.20*(1 - earnings risk)."""
    vol = _clip01((setup.volume_ratio or 0.0) / cfg.volume_norm_cap)
    rr = _clip01((setup.rr or 0.0) / cfg.rr_norm_cap)
    return (
        cfg.w_volume * vol + cfg.w_rr * rr + cfg.w_trend * trend_score(setup)
        + cfg.w_earnings * (1 - earnings_risk(setup, today, cfg))
    )


def exclusion(
    setup: Setup,
    cfg: ScreenerConfig,
    today: date,
    allow_earnings: bool = False,
    show_failed: bool = False,
    expected_move_pct: float | None = None,
) -> tuple[str, str] | None:
    """First reason to drop a setup as (code, text), in this order: failed, low volume, earnings, trend, overextended.

    Codes: failed, stopped, low_volume, earnings_hold, expected_move, trend, overextended."""
    if setup.kind == "failed" and not show_failed:
        return "failed", f"failed: spot ${setup.spot:.2f} is back under the breakout level ${setup.entry:.2f}"
    if setup.stop is not None and setup.spot <= setup.stop:
        return "stopped", f"price ${setup.spot:.2f} is already at or under the stop ${setup.stop:.2f}"
    if setup.kind == "breakout" or (setup.kind == "bull_flag" and setup.flag_broken):
        vr = setup.volume_ratio
        if vr is None or vr < cfg.min_volume_ratio:
            shown = "no data" if vr is None else f"{vr:.1f}x"
            return "low_volume", f"low volume ({shown} vs {cfg.min_volume_ratio:.1f}x needed)"
    if setup.earnings is not None and trading_days_between(today, setup.earnings) < cfg.hold_days:
        if not allow_earnings:
            return "earnings_hold", f"earnings {setup.earnings.isoformat()} within the {cfg.hold_days}-day hold"
        # earnings allowed: still drop it when the market expects a bigger move than the stop can absorb
        stop_pct = (setup.entry - setup.stop) / setup.entry if setup.entry and setup.stop else None
        if expected_move_pct is not None and stop_pct is not None and expected_move_pct > stop_pct:
            return "expected_move", f"earnings expected move {expected_move_pct:.1%} is larger than the {stop_pct:.1%} stop"
    if cfg.trend_filter == "sma200":
        if setup.sma200 is None or setup.last_close is None:
            return "trend", "below SMA200 (not enough history to know)"
        if setup.last_close <= setup.sma200:
            return "trend", f"below SMA200 (${setup.last_close:.2f} vs ${setup.sma200:.2f})"
    elif cfg.trend_filter == "sma50_rising":
        if setup.sma50 is None or setup.last_close is None or not setup.sma50_rising or setup.last_close <= setup.sma50:
            return "trend", "trend: not above a rising SMA50"
    if setup.kind in ("breakout", "pullback", "ma_pullback", "bull_flag") and setup.entry is not None:
        if setup.rsi is not None and setup.rsi > cfg.rsi_max:
            return "overextended", f"overextended: RSI(14) {setup.rsi:.0f} above {cfg.rsi_max:.0f}"
        if setup.atr14 and setup.spot - setup.entry > cfg.overextension_atr * setup.atr14:
            return "overextended", (
                f"overextended: ${setup.spot - setup.entry:.2f} above the entry, more than "
                f"{cfg.overextension_atr:g} ATR(14) (${setup.atr14:.2f})"
            )
    return None


def exclusion_reason(*args, **kwargs) -> str | None:
    """The text of `exclusion(...)`, or None when the setup is recommended."""
    found = exclusion(*args, **kwargs)
    return None if found is None else found[1]


def apply_filters(
    setup: Setup,
    cfg: ScreenerConfig,
    today: date,
    allow_earnings: bool = False,
    show_failed: bool = False,
    expected_move_pct: float | None = None,
) -> Setup:
    """Score a tradable setup and record why it was dropped, if it was. Non-tradable kinds pass through."""
    if setup.kind not in ACTIONABLE and setup.kind != "failed":
        return setup
    found = exclusion(setup, cfg, today, allow_earnings, show_failed, expected_move_pct)
    return replace(
        setup, excluded=None if found is None else found[1], excluded_code=None if found is None else found[0],
        score=score_setup(setup, today, cfg),
    )


def sell_signals(symbol: str, bars: pd.DataFrame, spot: float, now: datetime | None = None) -> list[SellSignal]:
    levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
    completed = completed_daily_bars(bars, now)
    closes = completed["close"]
    signals: list[SellSignal] = []

    weekly_s1 = levels.weekly.pivots.s1
    if spot < weekly_s1:
        signals.append(SellSignal(symbol, "below_weekly_s1", f"spot {spot:.2f} below weekly S1 {weekly_s1:.2f}"))

    level_series = prior_n_day_high(completed, 20)
    if len(level_series) >= 6:
        level = level_series.iloc[-6]  # 20-day high as it stood 5 bars ago
        if pd.notna(level) and closes.iloc[-5:].max() > level and spot < level:
            signals.append(
                SellSignal(symbol, "failed_breakout", f"spot {spot:.2f} back under breakout level {level:.2f}")
            )
    return signals


def load_universe(path: Path = CONSTITUENTS_PATH, extra: list[str] | None = None) -> list[str]:
    """S&P 500 symbols (BRK.B -> BRK/B for Schwab) followed by any `extra` symbols, de-duplicated."""
    with path.open(newline="", encoding="utf-8") as f:
        symbols = [row["Symbol"].strip().replace(".", "/") for row in csv.DictReader(f) if row["Symbol"].strip()]
    seen = set(symbols)
    for sym in extra or []:
        sym = sym.strip().upper().replace(".", "/")
        if sym and sym not in seen:
            symbols.append(sym)
            seen.add(sym)
    return symbols

"""Naked (uncovered) call recommendations.

Risk: a naked call has UNLIMITED loss if the stock keeps rising. Recommend-only.
Margin is the Reg T approximation for uncovered stock calls; the broker's figure governs.

Two tables:
- standard: names below their 20-day high, not in a breakout or extended setup.
- breakout-ready (HIGH RISK): names at or near their breakout level. Strikes sit
  at least 5% above the 20-day high, and every row is labelled high risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from tool.covered_calls import earnings_guard
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.expiries import expiry_candidates
from tool.holdings import LOT_SIZE, Holdings
from tool.levels import NY, compute_levels
from tool.screener import evaluate
from tool.config import get_config

_OPT = get_config().options
DELTA_LOW = _OPT.naked_delta_low
DELTA_HIGH = _OPT.naked_delta_high
MAX_SPREAD_PCT = _OPT.max_spread_pct
SKIP_SETUPS = ("breakout", "extended")
BREAKOUT_NEAR = _OPT.naked_breakout_near
HIGH_RISK_STRIKE_ABOVE_BREAKOUT = _OPT.naked_high_risk_strike_above_breakout


@dataclass(frozen=True)
class NakedCallRec:
    symbol: str
    status: str  # "recommend" | "skip"
    reason: str
    expiry_type: str | None = None
    expiry: date | None = None
    strike: float | None = None
    bid: float | None = None
    delta: float | None = None
    spot: float | None = None
    premium_per_contract: float | None = None
    margin_per_contract: float | None = None
    return_on_margin: float | None = None
    exit_trigger: float | None = None
    high_risk: bool = False
    entry_rule: str | None = None
    stop_rule: str | None = None
    target_rule: str | None = None
    entry_level: float | None = None
    stop_level: float | None = None
    stop_option: float | None = None
    t1: float | None = None
    t2: float | None = None
    notes: list[str] = field(default_factory=list)


def margin_per_contract(spot: float, strike: float, premium: float) -> float:
    """Reg T approximation for an uncovered stock call, per 100-share contract."""
    otm = max(strike - spot, 0.0)
    option_1 = 0.20 * spot * LOT_SIZE - otm * LOT_SIZE + premium * LOT_SIZE
    option_2 = 0.10 * spot * LOT_SIZE + premium * LOT_SIZE
    return max(option_1, option_2)


def _pick_strike(chain: pd.DataFrame, floor: float) -> tuple[pd.Series | None, bool]:
    """Strike at or above `floor`: delta in [0.15, 0.25] if one exists, else the lowest above the floor."""
    candidates = chain[chain["strike"] >= floor].sort_values("strike")
    if candidates.empty:
        return None, False
    in_band = candidates[candidates["delta"].between(DELTA_LOW, DELTA_HIGH)]
    if not in_band.empty:
        return in_band.iloc[0], True
    return candidates.iloc[0], False


def recommend_one(
    symbol: str,
    source: DataSource,
    now: datetime,
    events: FinnhubEvents | None,
    expiry_type: str,
    expiry: date,
    breakout_ready: bool = False,
) -> NakedCallRec:
    try:
        bars = source.daily_bars(symbol, lookback_days=120)
        spot = source.spot(symbol)
        levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
        setup = evaluate(symbol, bars, spot, now=now)
    except Exception as exc:
        return NakedCallRec(symbol, "skip", f"data error: {exc}", expiry_type, expiry)

    near_breakout = setup.kind == "breakout" or spot >= BREAKOUT_NEAR * levels.prior_20d_high
    if breakout_ready:
        if not near_breakout:
            return NakedCallRec(symbol, "skip", "not near breakout (more than 2% below 20D high)", expiry_type, expiry, spot=spot)
    elif setup.kind in SKIP_SETUPS or spot > levels.prior_20d_high:
        return NakedCallRec(
            symbol, "skip",
            f"setup is {setup.kind} / spot above 20D high — no naked calls into strength",
            expiry_type, expiry, spot=spot,
        )

    notes: list[str] = []
    if expiry_type != "earnings":
        hits, earnings_note = earnings_guard(events, symbol, now.date(), expiry)
        if hits:
            return NakedCallRec(
                symbol, "skip",
                f"earnings {hits[0].date.isoformat()} before expiry {expiry.isoformat()}",
                expiry_type, expiry, spot=spot,
            )
        notes.append(earnings_note)
    else:
        notes.append("expiry is after an earnings date by design — gap risk through the print")

    if breakout_ready:
        floor = max(levels.weekly.pivots.r2, levels.prior_20d_high * HIGH_RISK_STRIKE_ABOVE_BREAKOUT)
    else:
        floor = max(levels.weekly.pivots.r2, levels.prior_20d_high)

    try:
        chain = source.option_chain(symbol, expiry=expiry, right="C")
    except Exception as exc:
        return NakedCallRec(symbol, "skip", f"chain error: {exc}", expiry_type, expiry, spot=spot)

    pick, in_band = _pick_strike(chain, floor)
    if pick is None:
        return NakedCallRec(symbol, "skip", "no strike above resistance floor", expiry_type, expiry, spot=spot)

    strike = float(pick["strike"])
    bid = float(pick["bid"]) if pick.get("bid") is not None else None
    ask = float(pick["ask"]) if pick.get("ask") is not None else None
    if bid is None or bid <= 0:
        return NakedCallRec(symbol, "skip", "no bid on chosen strike", expiry_type, expiry, strike, spot=spot)

    mid = (bid + ask) / 2 if ask is not None else bid
    premium = bid * LOT_SIZE
    margin = margin_per_contract(spot, strike, bid)
    if ask is not None and mid and (ask - bid) / mid > MAX_SPREAD_PCT:
        notes.append(f"wide spread ({(ask - bid) / mid:.0%} of mid) — use a limit near mid")
    if not in_band:
        notes.append(f"no strike with delta {DELTA_LOW:.2f}–{DELTA_HIGH:.2f} above the floor; delta shown is outside the band")
    if breakout_ready:
        notes.insert(0, "HIGH RISK: stock is at or near its breakout level; a breakout keeps losses running")
    notes.append(
        f"UNLIMITED RISK: close if the stock closes above {levels.prior_20d_high:.2f} "
        f"(20-day high) or above {strike:.2f} (strike)"
    )
    notes.append("margin is a Reg T approximation; check your broker's requirement")

    trigger = levels.prior_20d_high if breakout_ready else floor
    entry_level = trigger - 0.5 * levels.atr10  # leave half an ATR of room below resistance
    t1_price, t2_price, stop_price = 0.5 * bid, 0.2 * bid, 2 * bid
    entry_rule = (
        f"Sell only while {symbol} is below ${entry_level:.2f} (half an ATR under resistance ${trigger:.2f}); "
        "do not sell into a breakout"
    )
    stop_rule = (
        f"Buy back if {symbol} closes above ${trigger:.2f}, or if the option reaches ${stop_price:.2f} (2x the premium)"
    )
    target_rule = (
        f"T1: buy back at ${t1_price:.2f} (keep 50%). T2: buy back at ${t2_price:.2f} (keep 80%), "
        f"or let it expire if {symbol} stays below ${strike:.2f}"
    )

    return NakedCallRec(
        symbol=symbol,
        status="recommend",
        reason="ok",
        expiry_type=expiry_type,
        expiry=expiry,
        strike=strike,
        bid=bid,
        delta=float(pick["delta"]) if pick.get("delta") is not None else None,
        spot=spot,
        premium_per_contract=premium,
        margin_per_contract=margin,
        return_on_margin=premium / margin if margin > 0 else None,
        exit_trigger=levels.prior_20d_high,
        high_risk=breakout_ready,
        entry_rule=entry_rule,
        stop_rule=stop_rule,
        target_rule=target_rule,
        entry_level=entry_level,
        stop_level=trigger,
        stop_option=stop_price,
        t1=t1_price,
        t2=t2_price,
        notes=notes,
    )


def recommend_naked_calls(
    symbols: list[str],
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
    breakout_ready: bool = False,
) -> list[NakedCallRec]:
    now = now or datetime.now(NY)
    out: list[NakedCallRec] = []
    for sym in symbols:
        candidates = expiry_candidates(sym, now, events)
        if hasattr(source, "hint_chains"):  # one request for all of the symbol's expiries instead of one each
            source.hint_chains(sym, "C", [d for _, d in candidates])
        for expiry_type, expiry in candidates:
            out.append(recommend_one(sym, source, now, events, expiry_type, expiry, breakout_ready))
    return out


def default_symbols(holdings: Holdings) -> list[str]:
    """Watchlist names you don't already hold (held names are covered-call territory)."""
    held = {sym for sym, _ in holdings.positions()}
    return [sym for sym in holdings.watchlist_scan if sym not in held]

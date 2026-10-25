"""Put recommendations: cash-secured puts (CSP), naked puts, and wheel put legs.

See CLAUDE.md "Cash-secured put rules": strike at or below weekly S1, showing
return, collateral, and effective cost (strike minus premium).

Recommend-only. Naked puts are not cash-secured: losses run to strike minus
premium per share if the stock goes to zero, and margin is required. The margin
figure is a Reg T approximation; the broker's figure governs.
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
DELTA_LOW = _OPT.put_delta_low
DELTA_HIGH = _OPT.put_delta_high
MAX_SPREAD_PCT = _OPT.max_spread_pct
SUSPECT_ANNUALIZED = _OPT.put_suspect_annualized  # above this, flag the quote as suspect (note only)
LOW_RETURN_NOTE = _OPT.put_low_return_note  # below this annualized return a note is added; nothing is filtered out


@dataclass(frozen=True)
class PutRec:
    symbol: str
    mode: str  # "csp" | "naked_put" | "wheel"
    status: str  # "recommend" | "skip"
    reason: str
    expiry_type: str | None = None
    expiry: date | None = None
    strike: float | None = None
    bid: float | None = None
    delta: float | None = None
    spot: float | None = None
    premium_per_contract: float | None = None
    collateral_per_contract: float | None = None
    margin_per_contract: float | None = None
    return_on_collateral: float | None = None
    annualized_return: float | None = None
    effective_cost: float | None = None
    max_loss_per_contract: float | None = None
    entry_rule: str | None = None
    stop_rule: str | None = None
    target_rule: str | None = None
    entry_level: float | None = None
    stop_level: float | None = None
    stop_option: float | None = None
    t1: float | None = None
    t2: float | None = None
    notes: list[str] = field(default_factory=list)
    # fields used by the support engine (csp_support); empty on the guardrail families
    dte: int | None = None
    support: float | None = None  # the nearest support the strike sits at or under
    support_label: str | None = None
    rank: float | None = None  # IV rank, or the HV stand-in (see rank_source)
    rank_source: str | None = None  # "IV" | "HV"
    break_even: float | None = None  # strike - premium
    prob_otm: float | None = None  # 1 - |delta|


def naked_put_margin(spot: float, strike: float, premium: float) -> float:
    """Reg T approximation for an uncovered stock put, per 100-share contract."""
    otm = max(spot - strike, 0.0)
    option_1 = (0.20 * spot - otm + premium) * LOT_SIZE
    option_2 = (0.10 * strike + premium) * LOT_SIZE
    return max(option_1, option_2)


def _pick_put(chain: pd.DataFrame, cap: float, dte: int) -> tuple[pd.Series | None, bool]:
    """Put at or below `cap` (weekly S1) with the best annualized return on collateral.

    Prefers |delta| in the target band; if none fits, falls back to the best-return
    strike at or below `cap`. Returns (row, in_band).
    """
    eligible = chain[(chain["strike"] <= cap) & (chain["bid"] > 0)].copy()
    if eligible.empty or dte <= 0:
        return None, False
    eligible["annualized"] = eligible["bid"] / eligible["strike"] * 365 / dte
    in_band = eligible[eligible["abs_delta"].between(DELTA_LOW, DELTA_HIGH)]
    if not in_band.empty:
        return in_band.sort_values("annualized", ascending=False).iloc[0], True
    return eligible.sort_values("annualized", ascending=False).iloc[0], False


def recommend_one(
    symbol: str,
    mode: str,
    source: DataSource,
    now: datetime,
    events: FinnhubEvents | None,
    expiry_type: str,
    expiry: date,
) -> PutRec:
    try:
        bars = source.daily_bars(symbol, lookback_days=120)
        spot = source.spot(symbol)
        levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
        setup = evaluate(symbol, bars, spot, now=now)
    except Exception as exc:
        return PutRec(symbol, mode, "skip", f"data error: {exc}", expiry_type, expiry)

    if setup.kind == "downtrend":
        return PutRec(symbol, mode, "skip", "downtrend (20-day change below -3%) — no puts on falling names",
                      expiry_type, expiry, spot=spot)

    notes: list[str] = []
    if expiry_type != "earnings":
        hits, earnings_note = earnings_guard(events, symbol, now.date(), expiry)
        if hits:
            return PutRec(symbol, mode, "skip",
                          f"earnings {hits[0].date.isoformat()} before expiry {expiry.isoformat()}",
                          expiry_type, expiry, spot=spot)
        notes.append(earnings_note)
    else:
        notes.append("expiry is after an earnings date by design — gap risk through the print")

    try:
        chain = source.option_chain(symbol, expiry=expiry, right="P")
    except Exception as exc:
        return PutRec(symbol, mode, "skip", f"chain error: {exc}", expiry_type, expiry, spot=spot)

    chain = chain.assign(abs_delta=pd.to_numeric(chain["delta"], errors="coerce").abs())
    dte = (expiry - now.date()).days
    cap = levels.weekly.pivots.s1
    pick, in_band = _pick_put(chain, cap, dte)
    if pick is None:
        return PutRec(symbol, mode, "skip", "no put at or below weekly S1 with a bid",
                      expiry_type, expiry, spot=spot)

    strike = float(pick["strike"])
    bid = float(pick["bid"])
    roc = bid / strike
    annualized = roc * 365 / dte
    if not in_band:
        notes.append(f"no put with |delta| {DELTA_LOW:.2f}–{DELTA_HIGH:.2f} at or below S1; delta shown is outside the band")
    if annualized < LOW_RETURN_NOTE:
        notes.append(
            f"LOW RETURN: {annualized:.1%} annualized on ${strike * LOT_SIZE:,.0f} of "
            f"{'collateral' if mode != 'naked_put' else 'notional'} — premium ${bid * LOT_SIZE:,.0f}"
        )
    if annualized > SUSPECT_ANNUALIZED:
        notes.append(
            f"SUSPECT QUOTE: {annualized:.0%} annualized is implausible — verify the bid before trading "
            "(stale, crossed, or bad data)"
        )
    ask = float(pick["ask"]) if pick.get("ask") is not None else None
    if ask is not None and (ask - bid) / ((ask + bid) / 2) > MAX_SPREAD_PCT:
        notes.append(f"wide spread ({(ask - bid) / ((ask + bid) / 2):.0%} of mid) — use a limit near mid")
    premium = bid * LOT_SIZE

    if mode == "naked_put":
        margin = naked_put_margin(spot, strike, bid)
        collateral = None
        notes.append(f"NAKED PUT: loss to strike minus premium (${(strike - bid) * LOT_SIZE:,.0f}/contract) if the stock goes to zero")
        notes.append("margin is a Reg T approximation; check your broker's requirement")
    else:
        margin = None
        collateral = strike * LOT_SIZE
        notes.append("cash-secured: keep the collateral in cash")

    entry_level = cap + 0.5 * levels.atr10  # leave half an ATR of cushion above weekly S1
    t1_price, t2_price, stop_price = 0.5 * bid, 0.2 * bid, 2 * bid
    entry_rule = (
        f"Sell only while {symbol} is above ${entry_level:.2f} (half an ATR over weekly S1 support ${cap:.2f})"
    )
    stop_rule = (
        f"Buy back if {symbol} closes below ${cap:.2f}, or if the option reaches ${stop_price:.2f} (2x the premium)"
    )
    target_rule = (
        f"T1: buy back at ${t1_price:.2f} (keep 50%). T2: buy back at ${t2_price:.2f} (keep 80%), or hold to "
        f"expiry; assignment at ${strike:.2f} costs ${strike - bid:.2f} per share"
    )

    return PutRec(
        symbol=symbol,
        mode=mode,
        status="recommend",
        reason="ok",
        expiry_type=expiry_type,
        expiry=expiry,
        strike=strike,
        bid=bid,
        delta=float(pick["delta"]) if pick.get("delta") is not None else None,
        spot=spot,
        premium_per_contract=premium,
        collateral_per_contract=collateral,
        margin_per_contract=margin,
        return_on_collateral=roc,
        annualized_return=roc * 365 / dte if dte > 0 else None,
        effective_cost=strike - bid,
        max_loss_per_contract=(strike - bid) * LOT_SIZE,
        entry_rule=entry_rule,
        stop_rule=stop_rule,
        target_rule=target_rule,
        entry_level=entry_level,
        stop_level=cap,
        stop_option=stop_price,
        t1=t1_price,
        t2=t2_price,
        notes=notes,
    )


def recommend_puts(
    symbols: list[str],
    mode: str,
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
) -> list[PutRec]:
    now = now or datetime.now(NY)
    out: list[PutRec] = []
    for sym in symbols:
        for expiry_type, expiry in expiry_candidates(sym, now, events):
            out.append(recommend_one(sym, mode, source, now, events, expiry_type, expiry))
    return out


def wheel_symbols(holdings: Holdings) -> list[str]:
    return sorted({l.symbol for l in holdings.lots if l.strategy == "wheel"})


def csp_symbols(holdings: Holdings) -> list[str]:
    """Watchlist names you don't hold (same universe as naked calls)."""
    held = {sym for sym, _ in holdings.positions()}
    return [sym for sym in holdings.watchlist_scan if sym not in held]

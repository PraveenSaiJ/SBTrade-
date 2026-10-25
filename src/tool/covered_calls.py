"""Covered-call strike recommendations — see CLAUDE.md "Covered-call engine rules".

Never places orders; this only produces recommendations for manual execution.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from tool.cc_flags import call_flags
from tool.data.base import DataSource
from tool.data.earnings import EarningsEvent, EventsUnavailable, FinnhubEvents
from tool.holdings import LOT_SIZE, Holdings
from tool.levels import NY, compute_levels
from tool.config import get_config

_OPT = get_config().options
TARGET_DELTA_LOW = _OPT.covered_delta_low
TARGET_DELTA_HIGH = _OPT.covered_delta_high
MAX_SPREAD_PCT = _OPT.max_spread_pct


@dataclass(frozen=True)
class CoveredCallRecommendation:
    symbol: str
    account: str
    status: str  # "recommend" | "wait" | "skip"
    reason: str
    free_lots: int = 0
    expiry: date | None = None
    strike: float | None = None
    bid: float | None = None
    mid: float | None = None
    delta: float | None = None
    return_pct: float | None = None  # bid / spot
    premium_per_lot: float | None = None  # bid * 100
    total_premium: float | None = None  # premium_per_lot * free_lots
    distance_otm_pct: float | None = None
    liquidity_ok: bool | None = None
    tax_note: str | None = None
    notes: list[str] = field(default_factory=list)
    mode: str = "guardrail"  # "guardrail" (CLAUDE.md rules) | "income" (nearest-ATM at the tier target)
    tier: str | None = None  # safe | proven | aggressive | other
    net_pct: float | None = None  # (bid - fee) / spot
    target_pct: float | None = None
    meets_target: bool | None = None
    prob_itm: float | None = None  # approximated by |delta|
    flags: list[str] = field(default_factory=list)  # ex-dividend, early assignment, earnings, manual events


def next_weekly_expiry(now: datetime) -> date:
    """Next Friday on/after `now`'s date; if today IS Friday, roll to next week.

    Selling a same-day-expiry call isn't a useful "weekly" recommendation, so
    a Friday run (per CLAUDE.md's "Friday afternoon" run) targets next week.
    """
    today = now.date()
    days_to_friday = (4 - today.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    return today + timedelta(days=days_to_friday)


def _select_strike_row(
    chain: pd.DataFrame,
    min_strike: float,
    nearest_to: float | None = None,
    use_delta_preference: bool = True,
) -> pd.Series | None:
    """Pick a strike >= min_strike (the cost-basis/pivot floor).

    - `nearest_to` set: pick the eligible strike closest to that price (ALAB
      ATM-trigger case).
    - otherwise: prefer a strike with delta in [0.20, 0.30] if one exists in
      the eligible set, else the lowest eligible strike.
    """
    candidates = chain[chain["strike"] >= min_strike].sort_values("strike")
    if candidates.empty:
        return None
    if nearest_to is not None:
        order = (candidates["strike"] - nearest_to).abs().sort_values().index
        return candidates.loc[order[0]]
    if use_delta_preference and "delta" in candidates:
        in_target = candidates[candidates["delta"].between(TARGET_DELTA_LOW, TARGET_DELTA_HIGH)]
        if not in_target.empty:
            return in_target.iloc[0]
    return candidates.iloc[0]


def _build_recommendation(
    symbol: str,
    account: str,
    pick: pd.Series,
    free_lots: int,
    spot: float,
    avg_cost: float,
    notes: list[str],
) -> CoveredCallRecommendation:
    strike = float(pick["strike"])
    bid = float(pick["bid"]) if pick.get("bid") is not None else None
    ask = float(pick["ask"]) if pick.get("ask") is not None else None
    mid = float(pick["mid"]) if pick.get("mid") is not None else (
        (bid + ask) / 2 if bid is not None and ask is not None else None
    )
    delta = pick.get("delta")

    liquidity_ok = None
    if bid is not None and ask is not None and mid:
        spread_pct = (ask - bid) / mid
        liquidity_ok = spread_pct <= MAX_SPREAD_PCT
        if not liquidity_ok:
            notes.append(f"wide bid/ask spread ({spread_pct:.0%} of mid) — use a limit near mid")
    else:
        notes.append("no live bid/ask — confirm quote manually before placing a limit")

    return_pct = bid / spot if bid is not None and spot else None
    premium_per_lot = bid * LOT_SIZE if bid is not None else None
    total_premium = premium_per_lot * free_lots if premium_per_lot is not None else None
    distance_otm_pct = (strike - spot) / spot if spot else None

    if account == "rh_ira":
        tax_note = "IRA — no tax effect if assigned"
    else:
        realized_gain_per_lot = (strike - avg_cost) * LOT_SIZE
        tax_note = f"Taxable — if assigned: ~${realized_gain_per_lot:,.2f}/lot realized gain"

    return CoveredCallRecommendation(
        symbol=symbol,
        account=account,
        status="recommend",
        reason="ok",
        free_lots=free_lots,
        expiry=pick["expiry"],
        strike=strike,
        bid=bid,
        mid=mid,
        delta=float(delta) if delta is not None else None,
        return_pct=return_pct,
        premium_per_lot=premium_per_lot,
        total_premium=total_premium,
        distance_otm_pct=distance_otm_pct,
        liquidity_ok=liquidity_ok,
        tax_note=tax_note,
        notes=notes,
    )


def _skip(symbol: str, account: str, reason: str, free_lots: int = 0, expiry: date | None = None) -> CoveredCallRecommendation:
    return CoveredCallRecommendation(symbol=symbol, account=account, status="skip", reason=reason, free_lots=free_lots, expiry=expiry)


def _alab_wheel_recommendation(
    symbol: str,
    account: str,
    free_lots: int,
    avg_cost: float,
    spot: float,
    weekly_r2: float,
    alab_rules: dict,
    source: DataSource,
    expiry: date,
    now: datetime,
) -> CoveredCallRecommendation:
    """ALAB (wheel) override — see CLAUDE.md per-ticker overrides.

    Aggressive: sell ATM when price touches `atm_call_trigger` on Mon/Tue;
    otherwise (past Tue close) fall back to ~1 strike above weekly R2.
    """
    trigger = float(alab_rules.get("atm_call_trigger", spot))
    weekday = now.weekday()  # Monday=0 ... Tuesday=1
    touched = spot >= trigger

    if weekday in (0, 1) and not touched:
        return CoveredCallRecommendation(
            symbol=symbol,
            account=account,
            status="wait",
            reason=f"wheel: waiting for spot ({spot:.2f}) to touch trigger {trigger:.2f}; deadline Tue close",
            free_lots=free_lots,
            expiry=expiry,
        )

    try:
        chain = source.option_chain(symbol, expiry=expiry, right="C")
    except Exception as exc:
        return _skip(symbol, account, f"chain error: {exc}", free_lots, expiry)

    if weekday in (0, 1) and touched:
        pick = _select_strike_row(chain, min_strike=avg_cost, nearest_to=trigger)
        notes = [f"wheel: trigger {trigger:.2f} touched — sold ATM"]
    else:
        min_strike = max(weekly_r2, avg_cost)
        pick = _select_strike_row(chain, min_strike=min_strike, use_delta_preference=False)
        notes = ["wheel: Tue close passed without a trigger touch — fallback to ~1 strike above weekly R2"]

    if pick is None:
        return _skip(symbol, account, "wheel: no suitable strike found in chain", free_lots, expiry)

    return _build_recommendation(symbol, account, pick, free_lots, spot, avg_cost, notes)


def earnings_guard(
    events: FinnhubEvents | None,
    symbol: str,
    today: date,
    expiry: date,
) -> tuple[list[EarningsEvent], str]:
    """Earnings between today and `expiry` (CLAUDE.md: skip such expiries)."""
    if events is None:
        return [], "earnings unverified — no events source configured"
    try:
        hits = events.earnings(symbol, today, expiry)
    except (EventsUnavailable, httpx.HTTPError) as exc:
        detail = re.sub(r"token=[^&\s'\"]+", "token=***", str(exc))
        return [], f"earnings unverified — events lookup failed ({detail})"
    if hits:
        return hits, ""
    return [], f"no earnings before {expiry.isoformat()} (checked Finnhub)"


def recommend_covered_calls(
    holdings: Holdings,
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
    fundamentals: dict | None = None,
) -> list[CoveredCallRecommendation]:
    """One recommendation per (symbol, account) position in `holdings` (the guardrail rules)."""
    from tool.income_calls import is_uncapped, tier_of  # local import: income_calls imports this module

    now = now or datetime.now(NY)
    results: list[CoveredCallRecommendation] = []
    cc_cfg = get_config().covered_calls

    for symbol, account in holdings.positions():
        if holdings.no_covered_calls_for(symbol, account) or is_uncapped(symbol, account, cc_cfg):
            results.append(replace(_skip(symbol, account, "no_covered_calls override (held uncapped)"), tier=tier_of(symbol, cc_cfg)))
            continue

        free_lots = holdings.free_lots_for(symbol, account)
        if free_lots <= 0:
            results.append(_skip(symbol, account, "no free lots (fully covered or < 100 free shares)"))
            continue

        avg_cost = holdings.avg_cost_for(symbol, account)

        try:
            bars = source.daily_bars(symbol, lookback_days=90)
            spot = source.spot(symbol)
        except Exception as exc:
            results.append(_skip(symbol, account, f"data error: {exc}", free_lots))
            continue

        try:
            levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
        except Exception as exc:
            results.append(_skip(symbol, account, f"levels error: {exc}", free_lots))
            continue

        expiry = next_weekly_expiry(now)
        breaking_out = spot > levels.prior_20d_high

        hits, earnings_note = earnings_guard(events, symbol, now.date(), expiry)
        if hits:
            results.append(
                _skip(
                    symbol, account,
                    f"earnings {hits[0].date.isoformat()} falls before expiry {expiry.isoformat()} — expiry skipped",
                    free_lots, expiry,
                )
            )
            continue

        strategy = holdings.strategy_for(symbol, account)
        if strategy == "wheel":
            rec = _alab_wheel_recommendation(
                symbol, account, free_lots, avg_cost, spot,
                levels.weekly.pivots.r2, holdings.alab_rules, source, expiry, now,
            )
            rec.notes.append(earnings_note)
            results.append(replace(
                rec, tier=tier_of(symbol, cc_cfg),
                flags=call_flags(symbol, now.date(), expiry, fundamentals, spot, rec.strike, rec.mid, cc_cfg),
            ))
            continue

        candidate_strike = levels.weekly.pivots.r2 if breaking_out else levels.weekly.pivots.r1
        min_strike = max(candidate_strike, avg_cost)  # never strike below cost basis

        try:
            chain = source.option_chain(symbol, expiry=expiry, right="C")
        except Exception as exc:
            results.append(_skip(symbol, account, f"chain error: {exc}", free_lots, expiry))
            continue

        pick = _select_strike_row(chain, min_strike=min_strike)
        if pick is None:
            results.append(_skip(symbol, account, "no suitable strike found in chain", free_lots, expiry))
            continue

        notes = []
        if breaking_out:
            notes.append("breaking out (spot > prior 20D high) — used weekly R2 for strike floor")
        if min_strike > candidate_strike:
            notes.append(f"pivot strike was below cost basis (${avg_cost:.2f}) — floor raised to cost basis")

        notes.append(earnings_note)
        rec = _build_recommendation(symbol, account, pick, free_lots, spot, avg_cost, notes)
        results.append(replace(
            rec, tier=tier_of(symbol, cc_cfg),
            prob_itm=abs(rec.delta) if rec.delta is not None else None,
            flags=call_flags(symbol, now.date(), expiry, fundamentals, spot, rec.strike, rec.mid, cc_cfg),
        ))

    return [r if r.tier else replace(r, tier=tier_of(r.symbol, cc_cfg)) for r in results]

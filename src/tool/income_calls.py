"""Income covered calls: the wheel-style weekly call nearest the money that nets the tier's target return.

This sits next to the guardrail engine in covered_calls.py (strike at or above weekly R1/R2, delta band, never below
cost basis): the dashboard shows both for each holding so the choice is yours. The income row's rules:
- strikes at or above the stock price (ATM or OTM, never in the money) and at or above the cost basis;
- the one nearest the stock price whose net return (bid less fees, over the stock price) meets the tier target;
- if none does, the best available is shown and marked "below target".
Earnings before expiry are flagged, not skipped. Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd

from tool.cc_flags import call_flags
from tool.config import CoveredCallConfig, get_config
from tool.covered_calls import (
    CoveredCallRecommendation, _build_recommendation, _skip, earnings_guard, next_weekly_expiry,
)
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.holdings import Holdings
from tool.levels import NY

SHARES_PER_CONTRACT = 100


def tier_of(symbol: str, cfg: CoveredCallConfig | None = None) -> str:
    cfg = cfg or get_config().covered_calls
    for tier, symbols in cfg.tiers.items():
        if symbol in symbols:
            return tier
    return "other"


def tier_target(tier: str, cfg: CoveredCallConfig | None = None) -> float:
    cfg = cfg or get_config().covered_calls
    return cfg.tier_targets.get(tier, cfg.tier_targets.get("other", 0.015))


def is_uncapped(symbol: str, account: str, cfg: CoveredCallConfig | None = None) -> bool:
    """True when config lists the symbol (`CRWV`) or the symbol in that account (`SNOW:rh_ira`) as held without calls."""
    cfg = cfg or get_config().covered_calls
    return symbol in cfg.uncapped or f"{symbol}:{account}" in cfg.uncapped


def net_return(bid: float, spot: float, fee_per_contract: float = 0.0) -> float:
    """Premium kept per share after the opening fee, as a share of the stock price."""
    return (bid - fee_per_contract / SHARES_PER_CONTRACT) / spot


def pick_income_strike(
    chain: pd.DataFrame, spot: float, floor: float, target: float, fee_per_contract: float = 0.0
) -> tuple[pd.Series, float, bool] | None:
    """(row, net return, meets target). Strikes at or above max(spot, floor) with a bid; nearest the spot that meets the
    target, else the best net return."""
    eligible = chain[(chain["strike"] >= max(spot, floor)) & (pd.to_numeric(chain["bid"], errors="coerce") > 0)]
    if eligible.empty:
        return None
    eligible = eligible.sort_values("strike").assign(
        _net=lambda d: [net_return(float(b), spot, fee_per_contract) for b in d["bid"]]
    )
    meeting = eligible[eligible["_net"] >= target]
    if not meeting.empty:
        row = meeting.iloc[0]  # lowest strike = nearest the money
        return row, float(row["_net"]), True
    row = eligible.sort_values("_net", ascending=False).iloc[0]
    return row, float(row["_net"]), False


def recommend_income_calls(
    holdings: Holdings,
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
    fundamentals: dict | None = None,
    cfg: CoveredCallConfig | None = None,
) -> list[CoveredCallRecommendation]:
    """One income-mode row per covered-call-eligible position (uncapped and wheel positions are left to their own rules)."""
    cfg = cfg or get_config().covered_calls
    now = now or datetime.now(NY)
    out: list[CoveredCallRecommendation] = []
    for symbol, account in holdings.positions():
        tier = tier_of(symbol, cfg)
        if holdings.no_covered_calls_for(symbol, account) or is_uncapped(symbol, account, cfg):
            continue  # the guardrail engine reports these as skipped
        if holdings.strategy_for(symbol, account) == "wheel":
            continue  # the wheel ticker has its own rule
        free_lots = holdings.free_lots_for(symbol, account)
        if free_lots <= 0:
            continue
        avg_cost = holdings.avg_cost_for(symbol, account)
        expiry = next_weekly_expiry(now)
        try:
            spot = source.spot(symbol)
            chain = source.option_chain(symbol, expiry=expiry, right="C")
        except Exception as exc:
            out.append(replace(_skip(symbol, account, f"data error: {exc}", free_lots, expiry), mode="income", tier=tier))
            continue
        picked = pick_income_strike(chain, spot, avg_cost, tier_target(tier, cfg), cfg.fee_per_contract)
        if picked is None:
            out.append(replace(
                _skip(symbol, account, "no strike at or above the stock price and cost basis has a bid", free_lots, expiry),
                mode="income", tier=tier,
            ))
            continue
        row, net, meets = picked
        notes: list[str] = []
        if avg_cost > spot:
            notes.append(f"stock is below your cost basis (${avg_cost:.2f}): strike floored at the cost basis")
        hits, earnings_note = earnings_guard(events, symbol, now.date(), expiry)
        flags = [f"earnings {hits[0].date.isoformat()} before expiry: gap risk through the strike"] if hits else []
        rec = _build_recommendation(symbol, account, row, free_lots, spot, avg_cost, notes)
        flags += call_flags(symbol, now.date(), expiry, fundamentals, spot, rec.strike, rec.mid, cfg)
        if not hits:
            notes.append(earnings_note)
        target = tier_target(tier, cfg)
        if not meets:
            notes.append(f"below the {target:.1%} target for {tier} names: best net return is {net:.2%}")
        out.append(replace(
            rec, mode="income", tier=tier, net_pct=net, target_pct=target, meets_target=meets,
            prob_itm=abs(rec.delta) if rec.delta is not None else None, flags=flags,
        ))
    return out

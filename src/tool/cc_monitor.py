"""Covered-call monitor: what to do with the short calls you hold, plus new calls to open.

Rules (CLAUDE.md "Roll rule", plus the 1% upside warning agreed for this section):
- CLOSE/ROLL when the stock is within 1% of the strike (or above it): upside is about to cap or assign the shares.
- CLOSE/ROLL urgently when the call is in the money with less than $0.50 of time value: early-assignment risk.
- A roll must be one week out, at the same or a higher strike, and only for a net credit, and never into earnings.
- With no credit roll, accept assignment (or switch to selling puts) instead of paying to roll.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from tool.covered_calls import earnings_guard, recommend_covered_calls
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.holdings import Holdings
from tool.levels import MARKET_CLOSE, NY
from tool.config import get_config

NEAR_STRIKE = get_config().options.near_strike
MIN_EXTRINSIC = get_config().options.min_extrinsic


@dataclass(frozen=True)
class RollOption:
    """One way to roll a short call: a net credit at the same or a higher strike, same expiry or up to 3 weeks out."""

    expiry: date
    strike: float
    bid: float
    net_credit: float  # bid of the new call minus the cost to buy back the old one
    delta: float | None
    weeks_out: int  # 0 = same expiry, higher strike


@dataclass(frozen=True)
class MonitorRow:
    symbol: str
    action: str  # OPEN | CLOSE | ROLL | HOLD | EXPIRES
    strike: float | None
    expiry: date | None
    contracts: int | None
    spot: float | None
    mid: float | None
    delta: float | None
    extrinsic: float | None
    distance_pct: float | None  # (strike - spot) / spot; negative once the call is in the money
    dte: int | None
    reason: str
    detail: str | None = None
    rolls: list[RollOption] = field(default_factory=list)  # net-credit rolls, best first


def assess(
    symbol: str, strike: float, expiry: date, contracts: int, spot: float, quote: pd.Series | None, today: date
) -> MonitorRow:
    """Hold, or close/roll, one short call. `quote` is its row from the option chain (None if unavailable)."""
    mid = None
    delta = None
    if quote is not None:
        bid, ask = quote.get("bid"), quote.get("ask")
        if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask):
            mid = (float(bid) + float(ask)) / 2
        d = quote.get("delta")
        delta = None if d is None or pd.isna(d) else float(d)
    intrinsic = max(spot - strike, 0.0)
    extrinsic = None if mid is None else max(mid - intrinsic, 0.0)
    distance = (strike - spot) / spot
    dte = (expiry - today).days

    common = dict(symbol=symbol, strike=strike, expiry=expiry, contracts=contracts, spot=spot, mid=mid,
                  delta=delta, extrinsic=extrinsic, distance_pct=distance, dte=dte)
    if spot > strike and extrinsic is not None and extrinsic < MIN_EXTRINSIC:
        return MonitorRow(
            action="CLOSE",
            reason=f"in the money with only ${extrinsic:.2f} of time value left — early-assignment risk",
            **common,
        )
    if distance <= NEAR_STRIKE:
        where = "above the strike" if spot > strike else f"within {distance:.1%} of the strike"
        return MonitorRow(action="CLOSE", reason=f"upside: {symbol} is {where} (${spot:.2f} vs ${strike:.2f})", **common)
    return MonitorRow(
        action="HOLD", reason=f"{distance:.1%} below the strike with {dte} days left", **common
    )


def find_rolls(
    symbol: str, strike: float, expiry: date, cost_to_close: float, source: DataSource,
    events: FinnhubEvents | None, today: date, weeks: int,
) -> tuple[list[RollOption], RollOption | None, str | None]:
    """Net-credit rolls for a short call: the same expiry at higher strikes, then each of the next `weeks` weeks at the same
    or a higher strike. Expiries with earnings before them are dropped. Returns (credits best-first, the least-bad debit
    roll for the "no credit" message, the first earnings blocker)."""
    credits: list[RollOption] = []
    best_debit: RollOption | None = None
    blocked: str | None = None
    seen: set[date] = set()
    for w in range(0, weeks + 1):
        target = expiry + timedelta(days=7 * w)
        try:
            chain = source.option_chain(symbol, expiry=target, right="C")
        except Exception:
            continue
        if chain.empty:
            continue
        actual = chain["expiry"].iloc[0] if "expiry" in chain else target
        if actual in seen or (w > 0 and actual <= expiry):
            continue
        seen.add(actual)
        if w > 0:
            hits, _ = earnings_guard(events, symbol, today, actual)
            if hits:
                blocked = blocked or f"earnings {hits[0].date.isoformat()} before the roll expiry {actual}"
                continue
        bids = pd.to_numeric(chain["bid"], errors="coerce")
        higher = chain["strike"] > strike if w == 0 else chain["strike"] >= strike
        options = []
        for _, c in chain[higher & (bids > 0)].sort_values("strike").iterrows():
            delta = c.get("delta")
            options.append(RollOption(
                actual, float(c["strike"]), float(c["bid"]), float(c["bid"]) - cost_to_close,
                None if delta is None or pd.isna(delta) else float(delta), w,
            ))
        good = [o for o in options if o.net_credit >= 0]
        if good:
            credits.append(good[0])  # lowest strike: the biggest credit
            if good[-1] is not good[0]:
                credits.append(good[-1])  # highest strike that still pays a credit: the most upside kept
        elif options and w > 0:
            candidate = max(options, key=lambda o: o.net_credit)
            if best_debit is None or candidate.net_credit > best_debit.net_credit:
                best_debit = candidate
    credits.sort(key=lambda o: (o.weeks_out, -o.net_credit))
    return credits, best_debit, blocked


def _roll_row(row: MonitorRow, credits: list[RollOption], cost_to_close: float, estimated: bool) -> MonitorRow:
    note = " (buy-back price estimated at intrinsic value; no live quote)" if estimated else ""
    best = credits[0]
    return _with(
        row, action="ROLL", rolls=credits,
        detail=f"roll to {best.expiry} ${best.strike:.2f} for a net credit of ${best.net_credit:.2f} "
        f"(sell ${best.bid:.2f}, buy back ${cost_to_close:.2f})" + note,
    )


def _with(row: MonitorRow, **changes: object) -> MonitorRow:
    return MonitorRow(**{**row.__dict__, **changes})


def _expiry_row(so, source: DataSource) -> MonitorRow:
    """A call whose expiry has passed or whose last trading session is over: it settles at the close."""
    try:
        spot = source.spot(so.symbol)
    except Exception:
        spot = None
    if spot is not None and spot > so.strike:
        reason = f"expired in the money: shares are called away at ${so.strike:.2f} (stock ${spot:.2f})"
    elif spot is not None:
        reason = f"expires worthless (stock ${spot:.2f} is below ${so.strike:.2f}); keep the premium"
    else:
        reason = "expired; stock price unavailable, check the account for assignment"
    distance = None if spot is None else (so.strike - spot) / spot
    return MonitorRow(so.symbol, "EXPIRES", so.strike, so.expiry, so.qty, spot, None, None, None, distance, 0, reason)


def monitor_covered_calls(
    holdings: Holdings,
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
    recs_out: list | None = None,
    fundamentals: dict | None = None,
) -> list[MonitorRow]:
    """`recs_out`, if given, receives the full covered-call engine results (including skips and their reasons)."""
    now = now or datetime.now(NY)
    today = now.date()
    rows: list[MonitorRow] = []

    market_closed = now.time() >= MARKET_CLOSE
    for so in (s for s in holdings.short_options if s.type == "call"):
        if so.expiry < today or (so.expiry == today and market_closed):
            rows.append(_expiry_row(so, source))
            continue
        try:
            spot = source.spot(so.symbol)
            chain = source.option_chain(so.symbol, expiry=so.expiry, right="C")
        except Exception as exc:
            rows.append(MonitorRow(so.symbol, "HOLD", so.strike, so.expiry, so.qty, None, None, None, None, None,
                                   (so.expiry - today).days, f"data error: {exc}"))
            continue
        match = chain[(chain["strike"] == so.strike) & (chain["expiry"] == so.expiry)] if "expiry" in chain else chain[chain["strike"] == so.strike]
        quote = match.iloc[0] if len(match) else None
        row = assess(so.symbol, so.strike, so.expiry, so.qty, spot, quote, today)
        if quote is None:
            row = _with(row, detail="this strike is outside the quoted option-chain window; decision uses the stock price only")
        ask = None if quote is None or pd.isna(quote.get("ask")) else float(quote["ask"])
        estimated = ask is None and row.mid is None
        cost_to_close = ask if ask is not None else row.mid if row.mid is not None else max(spot - so.strike, 0.0)
        credits, best_debit, blocked = find_rolls(
            so.symbol, so.strike, so.expiry, cost_to_close, source, events, today, get_config().covered_calls.roll_weeks
        )
        if row.action == "CLOSE":
            note = " (buy-back price estimated at intrinsic value; no live quote)" if estimated else ""
            if credits:
                row = _roll_row(row, credits, cost_to_close, estimated)
            elif blocked:
                row = _with(row, detail=f"{blocked}; no roll — accept assignment")
            elif best_debit is not None:
                row = _with(
                    row, detail=f"no credit roll (best is a ${-best_debit.net_credit:.2f} debit to {best_debit.expiry} "
                    f"${best_debit.strike:.2f}); accept assignment or switch to selling puts{note}",
                )
            else:
                row = _with(row, detail=f"no roll strike at or above ${so.strike:.2f}; accept assignment or switch to puts")
        else:
            row = _with(row, rolls=credits)
            if not credits and blocked:
                row = _with(row, detail=f"{blocked}: no roll")  # say why the roll panel is empty
        rows.append(row)

    recs = recommend_covered_calls(holdings, source, now=now, events=events, fundamentals=fundamentals)
    if recs_out is not None:
        recs_out.extend(recs)
    for rec in recs:
        if rec.status != "recommend":
            continue
        spot = rec.strike / (1 + rec.distance_otm_pct) if rec.distance_otm_pct is not None else None
        rows.append(
            MonitorRow(
                rec.symbol, "OPEN", rec.strike, rec.expiry, rec.free_lots, spot, rec.mid, rec.delta, None,
                rec.distance_otm_pct, (rec.expiry - today).days if rec.expiry else None,
                f"sell {rec.free_lots} call(s) at ${rec.strike:.2f}, bid ${rec.bid:.2f}",
                " · ".join(n for n in rec.notes if n),
            )
        )
    return rows

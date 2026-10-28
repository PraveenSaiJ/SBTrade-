"""Directional and LEAPS calls to BUY: higher-delta calls with their break-even weighed against the expected move.

- Directional: 14-60 days out, delta 0.60-0.70, on the bullish swing setups.
- LEAPS: 270+ days out, delta 0.70-0.80, on the Long term picks (buy zone or wait) and the same swing setups.
Each row carries the break-even and expected move to expiry, theta per day as a share of the premium, IV rank (an HV
stand-in until 60 days of IV exist), max loss (the premium) and a position size from the configured risk %. An IV-crush flag
fires when earnings fall before expiry and the rank is above the threshold. Recommend-only. Nothing is ordered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from tool.buy_options import BuyRec, Plan, _rules, trigger_status
from tool.config import AccountConfig, CallConfig, get_config
from tool.covered_calls import earnings_guard
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.holdings import LOT_SIZE
from tool.indicators import expected_move
from tool.levels import NY
from tool.support_puts import default_rank, friday_window

BULLISH_SETUPS = ("breakout", "bull_flag", "ma_pullback", "pullback")
RankFn = Callable[[str, pd.Series], "tuple[float | None, str]"]
LONG_TERM_CHASE = 0.10  # a long-term pick up to 10% over its entry is still in the buy zone
THIN_OPEN_INTEREST = 100


@dataclass(frozen=True)
class Candidate:
    symbol: str
    plan: Plan
    source: str  # "swing" | "long-term"
    label: str  # the setup kind, or "long-term buy zone" / "long-term wait"
    score: float


def _num(value: object) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(out) else float(out)


def swing_candidates(setups: pd.DataFrame, limit: int | None = None) -> list[Candidate]:
    """Bullish swing setups not excluded, best score first, with the Swing tab's levels."""
    if setups.empty:
        return []
    live = setups[setups["kind"].isin(BULLISH_SETUPS) & setups["entry"].notna()]
    if "excluded" in live:
        live = live[live["excluded"].isna()]
    live = live.assign(_s=pd.to_numeric(live.get("score"), errors="coerce").fillna(0)).sort_values("_s", ascending=False)
    out = []
    for _, r in live.iterrows():
        entry, stop, t1, t2, atr = (_num(r.get(k)) for k in ("entry", "stop", "t1", "t2", "atr"))
        if None in (entry, stop, t1, t2):
            continue
        plan = Plan("call", entry, entry + 0.5 * (atr or 0.0), stop, t1, t2, r["kind"])
        out.append(Candidate(r["symbol"], plan, "swing", r["kind"], float(r["_s"])))
    return out[:limit] if limit else out


def longterm_candidates(longterm: pd.DataFrame) -> list[Candidate]:
    """Long term 'buy zone' and 'wait' names with that tab's entry, stop and targets, best quality first."""
    if longterm.empty:
        return []
    live = longterm[longterm["status"].isin(["recommend", "wait"]) & longterm["entry"].notna()]
    live = live.assign(_s=pd.to_numeric(live["quality"], errors="coerce").fillna(0) * 10
                       + pd.to_numeric(live["trend"], errors="coerce").fillna(0)).sort_values("_s", ascending=False)
    out = []
    for _, r in live.iterrows():
        entry, stop, t1, t2 = (_num(r.get(k)) for k in ("entry", "stop", "t1", "t2"))
        if None in (entry, stop, t1, t2):
            continue
        plan = Plan("call", entry, entry * (1 + LONG_TERM_CHASE), stop, t1, t2, "long-term")
        label = "long-term buy zone" if r["status"] == "recommend" else "long-term wait"
        out.append(Candidate(r["symbol"], plan, "long-term", label, float(r["_s"])))
    return out


def leaps_candidates(setups: pd.DataFrame, longterm: pd.DataFrame, limit: int | None = None) -> list[Candidate]:
    """Long-term picks (their levels win) followed by the other bullish swing setups, each symbol once."""
    seen: dict[str, Candidate] = {}
    for c in longterm_candidates(longterm) + swing_candidates(setups):
        seen.setdefault(c.symbol, c)
    out = list(seen.values())
    return out[:limit] if limit else out


def pick_delta_call(chain: pd.DataFrame, low: float, high: float) -> pd.Series | None:
    """The call with an ask whose delta is in [low, high], nearest the band's midpoint (open interest breaks ties)."""
    ask = pd.to_numeric(chain["ask"], errors="coerce")
    delta = pd.to_numeric(chain["delta"], errors="coerce")
    eligible = chain[(ask > 0) & delta.between(low, high)].copy()
    if eligible.empty:
        return None
    mid = (low + high) / 2
    eligible["_gap"] = (pd.to_numeric(eligible["delta"], errors="coerce") - mid).abs()
    eligible["_oi"] = -pd.to_numeric(eligible.get("open_interest"), errors="coerce").fillna(0)
    return eligible.sort_values(["_gap", "_oi"]).iloc[0]


def call_metrics(row: pd.Series, spot: float, dte: int) -> dict:
    """Break-even, expected move, theta and extrinsic for one contract (all per share unless named otherwise)."""
    strike, ask = float(row["strike"]), float(row["ask"])
    breakeven = strike + ask
    iv = _num(row.get("iv"))
    move = expected_move(spot, iv, dte) if iv else None
    theta = _num(row.get("theta"))
    extrinsic = ask - max(spot - strike, 0.0)
    return {
        "breakeven": breakeven,
        "breakeven_pct": breakeven / spot - 1,
        "iv": iv,
        "expected_move": move,
        "expected_move_pct": None if move is None else move / spot,
        "be_vs_move": None if not move else (breakeven - spot) / move,
        "theta_pct": None if theta is None else abs(theta) / ask,
        "extrinsic": extrinsic,
        "extrinsic_pct": extrinsic / ask,
    }


def position_contracts(ask: float, account: AccountConfig) -> int | None:
    """Contracts so that losing the whole premium costs `risk_pct` of the account; None until an account size is set."""
    if account.account_size is None or account.account_size <= 0 or ask <= 0:
        return None
    return int(account.risk_pct * account.account_size // (ask * LOT_SIZE))


def _spread_flag(row: pd.Series) -> str | None:
    bid, ask = _num(row.get("bid")), _num(row.get("ask"))
    if bid is None or ask is None or ask <= 0:
        return None
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid if mid > 0 else 0
    return f"wide spread ({spread:.0%} of mid): use a limit near mid" if spread > get_config().options.max_spread_pct else None


def _one(
    family: str, cand: Candidate, source: DataSource, now: datetime, events: FinnhubEvents | None, spot: float,
    closes: pd.Series, rank_fn: RankFn, target: date, low: float, high: float, cfg: CallConfig, account: AccountConfig,
    window_days: int = 21, strike_count: int = 20,
) -> BuyRec:
    base = dict(symbol=cand.symbol, side="call", style="swing", spot=spot)
    label = family.split("_")[0]
    try:
        chain = source.option_chain(cand.symbol, expiry=target, right="C", window_days=window_days, strike_count=strike_count)
    except Exception as exc:
        return BuyRec(status="skip", reason=f"chain error: {exc}", expiry_type=f"{label} {target.isoformat()}", **base)
    if chain.empty:
        return BuyRec(status="skip", reason="empty option chain", expiry_type=f"{label} {target.isoformat()}", **base)
    expiry = chain["expiry"].iloc[0] if "expiry" in chain else target
    dte = (expiry - now.date()).days
    expiry_type = f"{label} {expiry.isoformat()}"
    row = pick_delta_call(chain, low, high)
    if row is None:
        return BuyRec(status="skip", reason=f"no call with delta {low:.2f}-{high:.2f} and an ask on {expiry}",
                      expiry_type=expiry_type, expiry=expiry, **base)
    ask = float(row["ask"])
    m = call_metrics(row, spot, dte)
    rank, rank_source = rank_fn(cand.symbol, closes)
    flags: list[str] = []
    hits, _ = earnings_guard(events, cand.symbol, now.date(), expiry)
    if label == "leaps" and hits and (hits[0].date - now.date()).days > cfg.leaps_earnings_days:
        hits = []  # a LEAPS always spans several reports; only one close at hand moves the price that matters
    if hits and rank is not None and rank > cfg.iv_crush_rank:
        flags.append(f"IV CRUSH RISK: earnings {hits[0].date.isoformat()} before expiry and rank {rank:.0f} ({rank_source}) is high")
    elif hits:
        flags.append(f"earnings {hits[0].date.isoformat()} before expiry: expect a gap and an IV drop")
    if m["be_vs_move"] is not None and m["be_vs_move"] > 1.0:
        flags.append(f"break-even is {m['be_vs_move']:.1f}x the expected move: needs more than the market expects")
    spread = _spread_flag(row)
    if spread:
        flags.append(spread)
    oi = _num(row.get("open_interest"))
    if oi is not None and oi < THIN_OPEN_INTEREST:
        flags.append(f"thin open interest ({oi:.0f})")
    entry_rule, stop_rule, target_rule = _rules(cand.symbol, cand.plan, "swing", ask, None)
    delta = _num(row.get("delta"))
    return BuyRec(
        status="recommend", reason="ok", expiry_type=expiry_type, expiry=expiry, strike=float(row["strike"]), ask=ask,
        delta=delta, cost_per_contract=ask * LOT_SIZE, trigger=trigger_status(cand.plan, spot), entry_rule=entry_rule,
        stop_rule=stop_rule, target_rule=target_rule, entry_level=cand.plan.trigger, stop_level=cand.plan.stop,
        stop_option=0.5 * ask, t1=cand.plan.t1, t2=cand.plan.t2, dte=dte, rank=rank, rank_source=rank_source,
        expected_move=m["expected_move"], expected_move_pct=m["expected_move_pct"], breakeven=m["breakeven"],
        breakeven_pct=m["breakeven_pct"], be_vs_move=m["be_vs_move"], theta_pct=m["theta_pct"], iv=m["iv"],
        extrinsic_pct=m["extrinsic_pct"], contracts=position_contracts(ask, account),
        max_loss_per_contract=ask * LOT_SIZE, flags=flags, idea_source=cand.label,
        notes=[f"{cand.label}: {cand.symbol} is ${spot:.2f}; entry ${cand.plan.trigger:.2f}"], **base,
    )


def _closes_and_spot(symbol: str, source: DataSource) -> tuple[pd.Series, float]:
    bars = source.daily_bars(symbol, lookback_days=400)
    return bars["close"], source.spot(symbol)


def recommend_directional(
    setups: pd.DataFrame, source: DataSource, now: datetime | None = None, events: FinnhubEvents | None = None,
    rank_fn: RankFn | None = None, cfg: CallConfig | None = None, account: AccountConfig | None = None,
) -> list[BuyRec]:
    """Directional calls (14-60 DTE, delta 0.60-0.70) on the bullish swing setups."""
    cfg = cfg or get_config().calls
    account = account or get_config().account
    now = now or datetime.now(NY)
    rank_fn = rank_fn or default_rank
    expiries = friday_window(now, cfg.directional_min_dte, cfg.directional_max_dte, cfg.directional_max_expiries)
    out: list[BuyRec] = []
    for cand in swing_candidates(setups, cfg.max_directional):
        try:
            closes, spot = _closes_and_spot(cand.symbol, source)
        except Exception as exc:
            out.append(BuyRec(cand.symbol, "call", "swing", "skip", f"data error: {exc}"))
            continue
        if hasattr(source, "hint_chains"):
            source.hint_chains(cand.symbol, "C", expiries)
        seen: set[date] = set()  # names with only monthly expiries resolve several targets to the same contract
        for target in expiries:
            rec = _one("directional_call", cand, source, now, events, spot, closes, rank_fn, target,
                       cfg.directional_delta_low, cfg.directional_delta_high, cfg, account)
            if rec.expiry is not None and rec.expiry in seen:
                continue
            if rec.expiry is not None:
                seen.add(rec.expiry)
            out.append(rec)
    return out


def recommend_leaps(
    setups: pd.DataFrame, longterm: pd.DataFrame, source: DataSource, now: datetime | None = None,
    events: FinnhubEvents | None = None, rank_fn: RankFn | None = None, cfg: CallConfig | None = None,
    account: AccountConfig | None = None,
) -> list[BuyRec]:
    """LEAPS (270+ DTE, delta 0.70-0.80) on the Long term picks and the bullish swing setups."""
    cfg = cfg or get_config().calls
    account = account or get_config().account
    now = now or datetime.now(NY)
    rank_fn = rank_fn or default_rank
    target = now.date() + timedelta(days=cfg.leaps_min_dte)
    out: list[BuyRec] = []
    for cand in leaps_candidates(setups, longterm, cfg.max_leaps):
        try:
            closes, spot = _closes_and_spot(cand.symbol, source)
        except Exception as exc:
            out.append(BuyRec(cand.symbol, "call", "swing", "skip", f"data error: {exc}"))
            continue
        out.append(_one("leaps_call", cand, source, now, events, spot, closes, rank_fn, target,
                        cfg.leaps_delta_low, cfg.leaps_delta_high, cfg, account,
                        window_days=cfg.leaps_window_days, strike_count=cfg.leaps_strike_count))
    return out

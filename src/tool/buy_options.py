"""Long calls/puts (swing, monthly) and weekly lottos, built from the swing setups.

Each idea carries an entry trigger on the stock ("buy only above X"), a stop-loss exit, and target exits.
Recommend-only. A long option can lose its whole premium; lottos usually do, so size them small.

Directional plan (from the swing screener rules in CLAUDE.md):
- calls: breakout or pullback setups. Trigger = the setup entry; stop = entry - 1 ATR; targets = T1 / T2.
- puts: downtrend setups. Trigger = weekly S1 breakdown; stop = trigger + 1 ATR; targets mirror the call rules downward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from tool.covered_calls import earnings_guard, next_weekly_expiry
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.expiries import monthly_expiry
from tool.holdings import LOT_SIZE, Holdings
from tool.levels import NY, compute_levels
from tool.screener import evaluate, pivot_levels
from tool.config import get_config

_OPT = get_config().options
SWING_DELTA = (_OPT.swing_delta_low, _OPT.swing_delta_high)
LOTTO_DELTA = (_OPT.lotto_delta_low, _OPT.lotto_delta_high)
LOTTO_MIN_PRICE_CAP = _OPT.lotto_min_price_cap  # dollars per share; pricier stocks use lotto_price_cap_pct instead
LOTTO_PRICE_CAP_PCT = _OPT.lotto_price_cap_pct
ODTE_DELTA = (_OPT.odte_delta_low, _OPT.odte_delta_high)
ODTE_PRICE_CAP_PCT = _OPT.odte_price_cap_pct
MAG7 = list(_OPT.mag7)
INDEX_ETFS = list(_OPT.index_etfs)


@dataclass(frozen=True)
class Plan:
    side: str  # "call" | "put"
    trigger: float
    chase_limit: float
    stop: float
    t1: float
    t2: float
    basis: str  # the setup this came from


@dataclass(frozen=True)
class BuyRec:
    symbol: str
    side: str
    style: str  # "lotto" | "swing"
    status: str  # "recommend" | "skip"
    reason: str
    expiry_type: str | None = None
    expiry: date | None = None
    strike: float | None = None
    ask: float | None = None
    delta: float | None = None
    spot: float | None = None
    cost_per_contract: float | None = None
    payoff_multiple: float | None = None
    trigger: str | None = None  # TRIGGERED | WAITING | EXTENDED
    entry_rule: str | None = None
    stop_rule: str | None = None
    target_rule: str | None = None
    entry_level: float | None = None
    stop_level: float | None = None
    stop_option: float | None = None
    t1: float | None = None
    t2: float | None = None
    notes: list[str] = field(default_factory=list)
    # directional and LEAPS calls (directional_calls.py); empty on the other families
    dte: int | None = None
    rank: float | None = None  # IV rank, or the HV stand-in (see rank_source)
    rank_source: str | None = None
    expected_move: float | None = None  # $ to expiry: spot x IV x sqrt(dte / 365)
    expected_move_pct: float | None = None
    breakeven: float | None = None  # strike + premium
    breakeven_pct: float | None = None  # breakeven over the stock price
    be_vs_move: float | None = None  # (breakeven - spot) / expected move
    theta_pct: float | None = None  # |theta| per day as a share of the premium
    iv: float | None = None
    extrinsic_pct: float | None = None  # time value as a share of the premium
    contracts: int | None = None  # sized to risk_pct of the account (None until account_size is set)
    max_loss_per_contract: float | None = None  # the whole premium
    flags: list[str] = field(default_factory=list)
    idea_source: str | None = None  # the setup or long-term label the levels came from


def make_plan(symbol: str, bars: pd.DataFrame, spot: float, now: datetime) -> tuple[Plan | None, str]:
    """Directional plan for the stock, or (None, setup kind) when there is no directional setup."""
    levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
    setup = evaluate(symbol, bars, spot, now=now)
    atr = levels.atr10
    if setup.kind in ("breakout", "pullback") and setup.entry is not None:
        return Plan("call", setup.entry, setup.entry + 0.5 * atr, setup.stop, setup.t1, setup.t2, setup.kind), setup.kind
    if setup.kind == "downtrend":
        x = levels.weekly.pivots.s1
        pivots = pivot_levels(levels)
        t1_options = [l for l in pivots if l <= x - 1.5 * atr]
        t1 = max(t1_options) if t1_options else x - 2 * atr
        t2_options = [l for l in pivots if l <= t1 - 0.75 * atr]
        t2 = max(t2_options) if t2_options else t1 - 1.5 * atr
        return Plan("put", x, x - 0.5 * atr, x + atr, t1, t2, "downtrend"), "downtrend"
    return None, setup.kind


def daily_pivot_plan(levels) -> Plan:
    """Same-day call plan from today's daily pivots: buy above P, stop halfway to S1, targets R1 and R2."""
    d = levels.daily.pivots
    return Plan("call", d.p, d.r1, (d.p + d.s1) / 2, d.r1, d.r2, "daily pivots")


def weekly_pivot_plan(levels) -> Plan:
    """Call plan from the weekly pivots, for names with no bullish swing setup: reclaim P, stop S1, targets R1 and R2."""
    w = levels.weekly.pivots
    return Plan("call", w.p, w.r1, w.s1, w.r1, w.r2, "weekly pivots")


def trigger_status(plan: Plan, spot: float) -> str:
    if plan.side == "call":
        if spot < plan.trigger:
            return "WAITING"
        return "TRIGGERED" if spot <= plan.chase_limit else "EXTENDED"
    if spot > plan.trigger:
        return "WAITING"
    return "TRIGGERED" if spot >= plan.chase_limit else "EXTENDED"


def _abs_delta(chain: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(chain["delta"], errors="coerce").abs()


def pick_swing(chain: pd.DataFrame) -> pd.Series | None:
    """The contract with delta closest to 0.50 (within 0.40-0.60) that has an ask."""
    c = chain.assign(abs_delta=_abs_delta(chain))
    c = c[(c["ask"] > 0) & c["abs_delta"].between(*SWING_DELTA)]
    if c.empty:
        return None
    return c.iloc[(c["abs_delta"] - 0.5).abs().argsort()].iloc[0]


def pick_lotto(
    chain: pd.DataFrame,
    side: str,
    spot: float,
    t1: float,
    delta_band: tuple[float, float] = LOTTO_DELTA,
    cap_pct: float = LOTTO_PRICE_CAP_PCT,
) -> pd.Series | None:
    """Cheap out-of-the-money contract with the best payoff multiple if the stock reaches its first target."""
    cap = max(LOTTO_MIN_PRICE_CAP, cap_pct * spot)
    c = chain.assign(abs_delta=_abs_delta(chain))
    c = c[(c["ask"] > 0) & (c["ask"] <= cap) & c["abs_delta"].between(*delta_band)].copy()
    if c.empty:
        return None
    reach = (t1 - c["strike"]) if side == "call" else (c["strike"] - t1)
    c["multiple"] = reach.clip(lower=0) / c["ask"]
    c = c[c["multiple"] > 0]
    if c.empty:
        return None
    return c.sort_values("multiple", ascending=False).iloc[0]


def _rules(symbol: str, plan: Plan, style: str, ask: float, multiple: float | None) -> tuple[str, str, str]:
    call = plan.side == "call"
    if call:
        if plan.basis == "breakout":
            entry = (
                f"Buy calls only if {symbol} trades above ${plan.trigger:.2f} (breakout level); "
                f"do not chase above ${plan.chase_limit:.2f}"
            )
        elif plan.basis.endswith("pivots"):
            entry = (
                f"Buy calls only if {symbol} trades above ${plan.trigger:.2f} ({plan.basis[:-7]} pivot); "
                f"do not chase above ${plan.chase_limit:.2f}"
            )
        else:
            entry = f"Buy calls only while {symbol} holds above ${plan.trigger:.2f} (support) and turns up"
    else:
        entry = (
            f"Buy puts only if {symbol} trades below ${plan.trigger:.2f} (weekly S1 breakdown); "
            f"do not chase below ${plan.chase_limit:.2f}"
        )
    way = "below" if call else "above"
    if style == "odte":
        stop = (
            f"Exit if {symbol} trades below ${plan.stop:.2f}, or if the option loses 50% (${0.5 * ask:.2f}); "
            "it expires today, so close it before the bell"
        )
        target = (
            f"Sell half at ${plan.t1:.2f} (R1) and the rest at ${plan.t2:.2f} (R2), "
            f"or when the option doubles (${2 * ask:.2f})"
        )
    elif style == "lotto":
        stop = (
            f"Risk is the whole premium (${ask * LOT_SIZE:,.0f}); exit if {symbol} closes back "
            f"{way} ${plan.trigger:.2f}"
        )
        target = (
            f"Sell half at 2x (${2 * ask:.2f}); sell the rest if {symbol} reaches ${plan.t1:.2f} "
            f"(T1; best case about {multiple:.1f}x at expiry, which is rare)"
        )
    else:
        stop = (
            f"Exit if {symbol} closes {way} ${plan.stop:.2f}, or if the option falls to ${0.5 * ask:.2f} (-50%)"
        )
        target = (
            f"Sell half at ${plan.t1:.2f} (T1) and the rest at ${plan.t2:.2f} (T2), "
            f"or when the option doubles (${2 * ask:.2f})"
        )
    return entry, stop, target


def _build(
    symbol: str, plan: Plan, style: str, pick: pd.Series, spot: float, expiry_type: str, expiry: date,
    notes: list[str],
) -> BuyRec:
    ask = float(pick["ask"])
    multiple = float(pick["multiple"]) if style in ("lotto", "odte") else None
    entry, stop, target = _rules(symbol, plan, style, ask, multiple)
    status = trigger_status(plan, spot)
    delta = pick.get("delta")
    return BuyRec(
        symbol=symbol,
        side=plan.side,
        style=style,
        status="recommend",
        reason="ok",
        expiry_type=expiry_type,
        expiry=expiry,
        strike=float(pick["strike"]),
        ask=ask,
        delta=float(delta) if delta is not None and not pd.isna(delta) else None,
        spot=spot,
        cost_per_contract=ask * LOT_SIZE,
        payoff_multiple=multiple,
        trigger=status,
        entry_rule=entry,
        stop_rule=stop,
        target_rule=target,
        entry_level=plan.trigger,
        stop_level=plan.trigger if style == "lotto" else plan.stop,
        stop_option=None if style == "lotto" else 0.5 * ask,
        t1=plan.t1,
        t2=plan.t2,
        notes=[f"{status}: {symbol} is ${spot:.2f}; trigger ${plan.trigger:.2f}", *notes],
    )


def _swing_and_lotto_ideas(
    symbol: str, plan: Plan, spot: float, source: DataSource, now: datetime, events: FinnhubEvents | None,
    weekly_swing: bool = True,
) -> list[BuyRec]:
    right = "C" if plan.side == "call" else "P"
    recs: list[BuyRec] = []
    chains: dict[date, pd.DataFrame] = {}  # the weekly lotto and the weekly call use the same expiry: fetch once
    for style, expiry_type, expiry in (
        ("lotto", "weekly", next_weekly_expiry(now)),
        ("swing", "weekly", next_weekly_expiry(now)),
        ("swing", "monthly", monthly_expiry(now.date())),
    ):
        if style == "swing" and expiry_type == "weekly" and not weekly_swing:
            continue
        notes: list[str] = []
        hits, _ = earnings_guard(events, symbol, now.date(), expiry)
        if hits:
            notes.append(f"earnings {hits[0].date.isoformat()} before expiry — expect an IV crush and gap risk")
        try:
            if expiry not in chains:
                chains[expiry] = source.option_chain(symbol, expiry=expiry, right=right)
            chain = chains[expiry]
        except Exception as exc:
            recs.append(BuyRec(symbol, plan.side, style, "skip", f"chain error: {exc}", expiry_type, expiry, spot=spot))
            continue
        actual = chain["expiry"].iloc[0] if "expiry" in chain and len(chain) else expiry
        pick = pick_lotto(chain, plan.side, spot, plan.t1) if style == "lotto" else pick_swing(chain)
        if pick is None:
            why = (
                "no cheap weekly strike the first target can reach" if style == "lotto"
                else "no strike with delta 0.40-0.60 and an ask"
            )
            recs.append(BuyRec(symbol, plan.side, style, "skip", why, expiry_type, actual, spot=spot))
            continue
        recs.append(_build(symbol, plan, style, pick, spot, expiry_type, actual, notes))
    return recs


def recommend_for_symbol(
    symbol: str, source: DataSource, now: datetime, events: FinnhubEvents | None
) -> list[BuyRec]:
    try:
        bars = source.daily_bars(symbol, lookback_days=120)
        spot = source.spot(symbol)
        plan, kind = make_plan(symbol, bars, spot, now)
    except Exception as exc:
        return [BuyRec(symbol, "?", "swing", "skip", f"data error: {exc}")]
    if plan is None:
        return [BuyRec(symbol, "?", "swing", "skip", f"setup is {kind} — no directional trade", spot=spot)]
    return _swing_and_lotto_ideas(symbol, plan, spot, source, now, events)


def recommend_odte(symbol: str, source: DataSource, now: datetime, events: FinnhubEvents | None) -> BuyRec:
    """A call that expires today (0DTE), planned from today's daily pivots. Run it during market hours."""
    try:
        bars = source.daily_bars(symbol, lookback_days=120)
        spot = source.spot(symbol)
        levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
        plan = daily_pivot_plan(levels)
    except Exception as exc:
        return BuyRec(symbol, "call", "odte", "skip", f"data error: {exc}", "0DTE")
    today = now.date()
    try:
        chain = source.option_chain(symbol, expiry=today, right="C")
    except Exception as exc:
        return BuyRec(symbol, "call", "odte", "skip", f"chain error: {exc}", "0DTE", today, spot=spot)
    actual = chain["expiry"].iloc[0] if "expiry" in chain and len(chain) else today
    notes: list[str] = []
    if actual != today:
        notes.append(f"no contract expires today; using the next expiry {actual} instead")
    hits, _ = earnings_guard(events, symbol, today, actual)
    if hits:
        notes.append(f"earnings {hits[0].date.isoformat()} before expiry — expect an IV crush and gap risk")
    expiry_type = "0DTE" if actual == today else "next expiry"
    pick = pick_lotto(chain, "call", spot, plan.t1, delta_band=ODTE_DELTA, cap_pct=ODTE_PRICE_CAP_PCT)
    if pick is None:
        return BuyRec(symbol, "call", "odte", "skip", "no call the first target (R1) can reach at a fair price",
                      expiry_type, actual, spot=spot)
    return _build(symbol, plan, "odte", pick, spot, expiry_type, actual, notes)


def recommend_mag7_calls(symbol: str, source: DataSource, now: datetime, events: FinnhubEvents | None) -> list[BuyRec]:
    """Weekly lotto and monthly swing calls on a MAG7 stock, whether or not it has a bullish swing setup."""
    try:
        bars = source.daily_bars(symbol, lookback_days=120)
        spot = source.spot(symbol)
        plan, kind = make_plan(symbol, bars, spot, now)
        if plan is None or plan.side != "call":
            plan = weekly_pivot_plan(compute_levels(bars, now=now, spot=spot, symbol=symbol))
    except Exception as exc:
        return [BuyRec(symbol, "call", "swing", "skip", f"data error: {exc}")]
    return _swing_and_lotto_ideas(symbol, plan, spot, source, now, events, weekly_swing=False)


def recommend_special(
    source: DataSource, now: datetime | None = None, events: FinnhubEvents | None = None
) -> dict[str, list[BuyRec]]:
    """The 0DTE and MAG7 call sections: families odte_call, mag7_lotto_call, mag7_swing_call."""
    now = now or datetime.now(NY)
    out: dict[str, list[BuyRec]] = {"odte_call": [], "mag7_lotto_call": [], "mag7_swing_call": []}
    for symbol in dict.fromkeys(INDEX_ETFS + MAG7):
        out["odte_call"].append(recommend_odte(symbol, source, now, events))
    for symbol in MAG7:
        for rec in recommend_mag7_calls(symbol, source, now, events):
            out["mag7_lotto_call" if rec.style == "lotto" else "mag7_swing_call"].append(rec)
    return out


def candidate_symbols(
    setups: pd.DataFrame, holdings: Holdings, max_calls: int = 40, max_puts: int = 25
) -> tuple[list[str], list[str]]:
    """Bullish and bearish names from the latest swing scan, limited to holdings plus the watchlist."""
    pool = {sym for sym, _ in holdings.positions()} | set(holdings.watchlist_scan)
    df = setups[setups["symbol"].isin(pool)]
    if "excluded" in df:
        df = df[df["excluded"].isna() | ~df["kind"].isin(["breakout", "bull_flag", "ma_pullback", "pullback", "extended", "failed"])]
    bull = df[df["kind"].isin(["breakout", "bull_flag", "ma_pullback", "pullback"]) & df["entry"].notna()]
    bull = bull.sort_values("rr", ascending=False)
    bear = df[df["kind"] == "downtrend"].sort_values("change_20d", ascending=True)
    return list(bull["symbol"].head(max_calls)), list(bear["symbol"].head(max_puts))


def recommend_buys(
    symbols: list[str], source: DataSource, now: datetime | None = None, events: FinnhubEvents | None = None
) -> dict[str, list[BuyRec]]:
    """Group results into the dashboard families: lotto, weekly, and long (monthly) for calls and puts."""
    now = now or datetime.now(NY)
    out: dict[str, list[BuyRec]] = {
        "lotto_call": [], "weekly_call": [], "long_call": [], "lotto_put": [], "weekly_put": [], "long_put": []
    }
    for symbol in symbols:
        for rec in recommend_for_symbol(symbol, source, now, events):
            side = rec.side if rec.side in ("call", "put") else "call"
            if rec.style == "lotto":
                kind = "lotto"
            else:
                kind = "weekly" if rec.expiry_type == "weekly" else "long"
            out[f"{kind}_{side}"].append(rec)
    return out

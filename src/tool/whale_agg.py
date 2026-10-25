"""Whale flow over days, and how it lines up with your own positions.

`whale_flow` holds a row per contract per 15-minute scan, and a contract's volume is cumulative over the day, so totals
must take each contract's LAST row of each day (summing scans would count the same trades many times). Only whale-size
trades are stored, so every total here is a whale-only figure, not all option volume. Recommend-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tool.config import get_config

CONTRACT = ["symbol", "right", "expiry", "strike"]


def burst_flags(df: pd.DataFrame) -> pd.Series:
    """True where the premium added since the previous scan reaches the burst threshold (index names scale up)."""
    from tool.whales import INDEX_SCALE, LARGE_PREMIUM, _W
    from tool.buy_options import INDEX_ETFS

    if "burst_premium" not in df:
        return pd.Series(False, index=df.index)
    base = _W.burst_premium or LARGE_PREMIUM
    need = df["symbol"].map(lambda s: base * (INDEX_SCALE if s in INDEX_ETFS else 1.0))
    return pd.to_numeric(df["burst_premium"], errors="coerce").fillna(0) >= need


def end_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Each contract's last row of each day, noise removed."""
    if df.empty:
        return df
    df = df[df["noise"].isna()] if "noise" in df else df
    df = df.assign(day=df["ts"].str[:10])
    last = df.groupby(["day", *CONTRACT])["ts"].transform("max") == df["ts"]
    return df[last]


def net_flow(history: pd.DataFrame, days: int) -> pd.DataFrame:
    """Per symbol over the last `days` trading days saved: bullish, bearish and net premium (neutral prints are left out),
    call and put premium, and the number of contracts. One row per symbol, biggest total first."""
    cols = ["symbol", "bullish", "bearish", "net", "calls", "puts", "contracts", "bias"]
    if history.empty:
        return pd.DataFrame(columns=cols)
    keep_days = sorted(history["ts"].str[:10].unique())[-days:]
    eod = end_of_day(history[history["ts"].str[:10].isin(keep_days)])
    if eod.empty:
        return pd.DataFrame(columns=cols)
    directional = eod[eod["lean"].isin(["bullish", "bearish"])]
    prem = directional["premium"]
    frame = pd.DataFrame({
        "symbol": directional["symbol"],
        "bullish": prem.where(directional["lean"] == "bullish", 0.0),
        "bearish": prem.where(directional["lean"] == "bearish", 0.0),
        "calls": prem.where(directional["right"] == "C", 0.0),
        "puts": prem.where(directional["right"] == "P", 0.0),
        "contracts": 1,
    })
    out = frame.groupby("symbol").sum().reset_index()
    out["net"] = out["bullish"] - out["bearish"]
    out["bias"] = [flow_bias(b, r) for b, r in zip(out["bullish"], out["bearish"])]
    return out[cols].sort_values("net", key=lambda s: s.abs(), ascending=False)


def flow_bias(bullish: float, bearish: float) -> str:
    """'bullish' / 'bearish' when one side is `flow_ratio` times the other, else 'mixed' ('none' with no flow)."""
    ratio = get_config().whales.flow_ratio
    if bullish <= 0 and bearish <= 0:
        return "none"
    if bullish > ratio * bearish:
        return "bullish"
    if bearish > ratio * bullish:
        return "bearish"
    return "mixed"


# ---------------------------------------------------------------- your positions
@dataclass(frozen=True)
class Position:
    kind: str  # shares | long call | long put | short call | short put
    exposure: str  # "long": hurt by a fall (shares, long call, short put); "short": hurt by a rally (long put, short call)
    detail: str


def position_map(holdings, tracked: pd.DataFrame | None = None, today: date | None = None) -> dict[str, list[Position]]:
    """symbol -> positions from holdings (shares, short and long options) and the open Bought-options tracker.
    Options past their expiry are skipped (a stale statement can still list them)."""
    today = today or date.today()
    out: dict[str, list[Position]] = {}

    def add(symbol: str, kind: str, exposure: str, detail: str) -> None:
        out.setdefault(symbol, []).append(Position(kind, exposure, detail))

    shares: dict[str, float] = {}
    for lot in holdings.lots:
        shares[lot.symbol] = shares.get(lot.symbol, 0.0) + lot.shares
    for symbol, n in shares.items():
        if n > 0:
            add(symbol, "shares", "long", f"{n:g} shares")
    for o in holdings.short_options:
        if o.expiry < today:
            continue
        add(o.symbol, f"short {o.type}", "long" if o.type == "put" else "short", f"short {o.qty} {o.type} {o.strike:g} {o.expiry}")
    for o in holdings.long_options:
        if o.expiry < today:
            continue
        add(o.symbol, f"long {o.type}", "long" if o.type == "call" else "short", f"long {o.qty} {o.type} {o.strike:g} {o.expiry}")
    if tracked is not None and not tracked.empty:
        for _, t in tracked[tracked["status"] == "open"].iterrows():
            if str(t["expiry"])[:10] < today.isoformat():
                continue
            add(t["symbol"], f"long {t['opt_type']}", "long" if t["opt_type"] == "call" else "short",
                f"tracked long {int(t['qty'])} {t['opt_type']} {t['strike']:g} {t['expiry']}")
    return out


def relation(position: Position, bias: str) -> str:
    """conflict | supports | neutral for one position against the flow's bias."""
    if bias in ("none", "mixed"):
        return "neutral"
    hurts = (position.exposure == "long" and bias == "bearish") or (position.exposure == "short" and bias == "bullish")
    return "conflict" if hurts else "supports"


def position_flow(
    positions: dict[str, list[Position]], flow_1d: pd.DataFrame, flow_5d: pd.DataFrame
) -> pd.DataFrame:
    """One row per held symbol that has whale flow today or in 5 days, conflicts first."""
    one = flow_1d.set_index("symbol") if not flow_1d.empty else pd.DataFrame()
    five = flow_5d.set_index("symbol") if not flow_5d.empty else pd.DataFrame()
    rows = []
    for symbol, held in positions.items():
        if symbol not in one.index and symbol not in five.index:
            continue
        net1 = float(one.loc[symbol, "net"]) if symbol in one.index else 0.0
        net5 = float(five.loc[symbol, "net"]) if symbol in five.index else 0.0
        bias1 = one.loc[symbol, "bias"] if symbol in one.index else "none"
        bias5 = five.loc[symbol, "bias"] if symbol in five.index else "none"
        # today's flow decides; with none today the 5-day flow does
        bias = bias1 if bias1 != "none" else bias5
        relations = [relation(p, bias) for p in held]
        status = (
            "mixed" if "conflict" in relations and "supports" in relations
            else "CONFLICT" if "conflict" in relations else "supports" if "supports" in relations else "neutral"
        )
        rows.append({
            "symbol": symbol, "status": status, "flow_today": bias1, "net_today": net1, "flow_5d": bias5, "net_5d": net5,
            "positions": "; ".join(f"{p.detail} ({r})" for p, r in zip(held, relations)),
        })
    order = {"CONFLICT": 0, "mixed": 1, "supports": 2, "neutral": 3}
    return pd.DataFrame(rows).assign(_o=lambda d: d["status"].map(order)).sort_values(["_o", "symbol"]).drop(columns="_o") if rows else pd.DataFrame(
        columns=["symbol", "status", "flow_today", "net_today", "flow_5d", "net_5d", "positions"])

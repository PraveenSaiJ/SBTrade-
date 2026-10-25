"""Portfolio risk across accounts: net greeks, beta-weighted delta to SPY, sector exposure, this-week risk, cash vs collateral.

Positions come from the holdings file (stocks and options, each option tagged with its account); greeks come from the live option
chain at each option's strike and expiry. Shares have delta 1. Short calls and long puts carry negative delta; theta is in dollars a
day and is positive when you are short options. Funds and leveraged ETFs are not scaled for leverage, and a missing beta counts as
1.0 (marked). Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import pandas as pd

from tool.cc_flags import ex_dividend
from tool.config import get_config
from tool.holdings import Holdings, option_account
from tool.themes import theme_of

LOT = 100
MIN_TIME_VALUE = 0.50  # an ITM short call with less time value than this can be assigned early


@dataclass(frozen=True)
class PositionRow:
    account: str
    symbol: str
    kind: str  # shares | long call | long put | short call | short put
    qty: float  # shares, or contracts
    strike: float | None
    expiry: str | None
    price: float | None  # the underlying
    option_mid: float | None
    delta_shares: float | None  # position delta in shares of the underlying
    theta_day: float | None  # dollars a day (positive when short options)
    beta: float | None
    beta_assumed: bool
    spy_equiv_shares: float | None  # beta-weighted delta in SPY shares
    delta_dollars: float | None
    value: float | None  # market value (short options negative)
    status: str  # live | expired | no quote | no price


def _num(x: object) -> float | None:
    out = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(out) else float(out)


def match_contract(chain: pd.DataFrame, strike: float, expiry: date) -> pd.Series | None:
    if chain is None or chain.empty:
        return None
    rows = chain[(chain["strike"] == strike) & (chain["expiry"] == expiry)] if "expiry" in chain else chain[chain["strike"] == strike]
    return rows.iloc[0] if len(rows) else None


def build_position_rows(
    holdings: Holdings, source, fundamentals: dict, spy_price: float, today: date | None = None
) -> list[PositionRow]:
    """One row per (account, stock) and per option position, with delta, theta, beta and the SPY-equivalent delta."""
    today = today or date.today()
    spots: dict[str, float | None] = {}

    def spot(sym: str) -> float | None:
        if sym not in spots:
            try:
                spots[sym] = source.spot(sym)
            except Exception:
                spots[sym] = None
        return spots[sym]

    def beta(sym: str) -> tuple[float, bool]:
        b = _num((fundamentals.get(sym) or {}).get("beta"))
        return (b, False) if b else (1.0, True)

    rows: list[PositionRow] = []
    by_stock: dict[tuple[str, str], float] = {}
    for lot in holdings.lots:
        by_stock[(lot.account, lot.symbol)] = by_stock.get((lot.account, lot.symbol), 0.0) + lot.shares
    for (account, sym), shares in sorted(by_stock.items()):
        px, (b, assumed) = spot(sym), beta(sym)
        rows.append(PositionRow(
            account, sym, "shares", shares, None, None, px, None, shares, 0.0, b, assumed,
            None if px is None else shares * px * b / spy_price, None if px is None else shares * px,
            None if px is None else shares * px, "live" if px is not None else "no price",
        ))
    for options, sign, side in ((holdings.long_options, 1, "long"), (holdings.short_options, -1, "short")):
        for o in options:
            px, (b, assumed) = spot(o.symbol), beta(o.symbol)
            kind = f"{side} {o.type}"
            base = dict(account=option_account(o), symbol=o.symbol, kind=kind, qty=float(o.qty), strike=o.strike,
                        expiry=o.expiry.isoformat(), price=px, beta=b, beta_assumed=assumed)
            if o.expiry < today:
                rows.append(PositionRow(**base, option_mid=None, delta_shares=0.0, theta_day=0.0, spy_equiv_shares=0.0,
                                        delta_dollars=0.0, value=0.0, status="expired"))
                continue
            quote = None
            try:
                chain = source.option_chain(o.symbol, expiry=o.expiry, right="C" if o.type == "call" else "P", strike_count=60)
                quote = match_contract(chain, o.strike, o.expiry)
            except Exception:
                pass
            if quote is None or px is None:
                rows.append(PositionRow(**base, option_mid=None, delta_shares=None, theta_day=None, spy_equiv_shares=None,
                                        delta_dollars=None, value=None, status="no quote" if px is not None else "no price"))
                continue
            delta, theta = _num(quote.get("delta")), _num(quote.get("theta"))
            bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
            mid = (bid + ask) / 2 if bid is not None and ask is not None and ask > 0 else _num(quote.get("last"))
            d_shares = None if delta is None else delta * LOT * o.qty * sign
            rows.append(PositionRow(
                **base, option_mid=mid, delta_shares=d_shares,
                theta_day=None if theta is None else theta * LOT * o.qty * sign,
                spy_equiv_shares=None if d_shares is None else d_shares * px * b / spy_price,
                delta_dollars=None if d_shares is None else d_shares * px,
                value=None if mid is None else mid * LOT * o.qty * sign, status="live",
            ))
    return rows


def positions_frame(rows: list[PositionRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


# ---------------------------------------------------------------- totals and exposure
def totals(positions: pd.DataFrame, spy_price: float | None = None) -> dict:
    """Portfolio totals: delta in dollars, theta per day, beta-weighted delta in SPY shares (and dollars), stock value."""
    live = positions[positions["status"] == "live"] if not positions.empty else positions
    spy_shares = float(pd.to_numeric(live["spy_equiv_shares"], errors="coerce").sum()) if not live.empty else 0.0
    return {
        "delta_dollars": float(pd.to_numeric(live["delta_dollars"], errors="coerce").sum()) if not live.empty else 0.0,
        "theta_day": float(pd.to_numeric(live["theta_day"], errors="coerce").sum()) if not live.empty else 0.0,
        "spy_equiv_shares": spy_shares,
        "spy_equiv_dollars": None if spy_price is None else spy_shares * spy_price,
        "stock_value": float(pd.to_numeric(live[live["kind"] == "shares"]["value"], errors="coerce").sum()) if not live.empty else 0.0,
        "unscaled_beta": int(positions["beta_assumed"].sum()) if not positions.empty else 0,
        "no_quote": int((positions["status"].isin(["no quote", "no price"])).sum()) if not positions.empty else 0,
    }


def greeks_by_symbol(positions: pd.DataFrame) -> pd.DataFrame:
    live = positions[positions["status"] == "live"]
    if live.empty:
        return pd.DataFrame(columns=["symbol", "delta_shares", "delta_dollars", "theta_day", "spy_equiv_shares"])
    g = live.assign(**{c: pd.to_numeric(live[c], errors="coerce") for c in ("delta_shares", "delta_dollars", "theta_day", "spy_equiv_shares")})
    out = g.groupby("symbol")[["delta_shares", "delta_dollars", "theta_day", "spy_equiv_shares"]].sum().reset_index()
    return out.sort_values("delta_dollars", key=lambda s: s.abs(), ascending=False)


def sector_exposure(positions: pd.DataFrame, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(by industry, by theme): share-position value and its % of the stock portfolio."""
    shares = positions[(positions["kind"] == "shares") & (positions["status"] == "live")]
    if shares.empty:
        empty = pd.DataFrame(columns=["group", "value", "weight", "symbols"])
        return empty, empty
    by_symbol = shares.groupby("symbol")["value"].sum()
    info = info.set_index("symbol") if "symbol" in info.columns else info
    total = float(by_symbol.sum())

    def table(label_of) -> pd.DataFrame:
        df = pd.DataFrame({"symbol": by_symbol.index, "value": by_symbol.values})
        df["group"] = df["symbol"].map(label_of)
        out = df.groupby("group").agg(value=("value", "sum"), symbols=("symbol", lambda s: ", ".join(sorted(s)))).reset_index()
        out["weight"] = out["value"] / total if total else 0.0
        return out.sort_values("value", ascending=False)[["group", "value", "weight", "symbols"]]

    industry = lambda s: (info["industry"].get(s) if "industry" in info and s in info.index else None) or "Unknown"  # noqa: E731
    return table(industry), table(theme_of)


# ---------------------------------------------------------------- this week
def week_risk(positions: pd.DataFrame, info: pd.DataFrame, today: date | None = None, days: int = 7) -> pd.DataFrame:
    """Short options expiring within `days` (or already expired): moneyness, time value, assignment and event flags."""
    today = today or date.today()
    if positions.empty:
        return pd.DataFrame()
    info = info.set_index("symbol") if "symbol" in info.columns else info
    cutoff = (today + timedelta(days=days)).isoformat()
    shorts = positions[positions["kind"].str.startswith("short") & (positions["expiry"] <= cutoff)]
    share_count = positions[positions["kind"] == "shares"].groupby(["account", "symbol"])["qty"].sum().to_dict()
    calls_by_acct = positions[positions["kind"] == "short call"].groupby(["account", "symbol"])["qty"].sum().to_dict()
    out = []
    for _, p in shorts.iterrows():
        flags: list[str] = []
        if p["status"] == "expired":
            flags.append(f"expired {p['expiry']}: check the account for assignment or expiry; the holdings file may be stale")
            out.append({**_week_base(p), "flags": "; ".join(flags)})
            continue
        spot, strike, mid = _num(p["price"]), float(p["strike"]), _num(p["option_mid"])
        call = p["kind"] == "short call"
        itm = spot is not None and (spot > strike if call else spot < strike)
        intrinsic = 0.0 if spot is None else max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
        tv = None if mid is None else max(mid - intrinsic, 0.0)
        row = {**_week_base(p), "itm": "ITM" if itm else "out", "away": None if not spot else (strike / spot - 1),
               "time_value": tv}
        sym_info = info.loc[p["symbol"]] if p["symbol"] in info.index else None
        if itm and tv is not None and tv < MIN_TIME_VALUE:
            flags.append(f"ASSIGNMENT RISK: in the money with ${tv:.2f} of time value")
        if sym_info is not None:
            ex, div = _date(sym_info.get("ex_div")), _num(sym_info.get("dividend")) or 0.0
            if call and ex is not None and today <= ex <= date.fromisoformat(p["expiry"]) and itm and tv is not None and tv < div:
                flags.append(f"ASSIGNMENT RISK: ex-dividend {ex} (${div:.2f}) and time value is below it")
            nxt = _date(sym_info.get("next_earnings"))
            if nxt is not None and today <= nxt <= date.fromisoformat(p["expiry"]):
                flags.append(f"earnings {nxt} before expiry")
        if call:
            held = share_count.get((p["account"], p["symbol"]), 0.0)
            need = calls_by_acct.get((p["account"], p["symbol"]), 0.0) * LOT
            if held < need:
                flags.append(f"NAKED: {held:g} shares in {p['account']} for {need:g} shares of calls")
        else:
            flags.append(f"if assigned: ${strike * LOT * p['qty']:,.0f} cash needed")
        row["flags"] = "; ".join(flags)
        out.append(row)
    return pd.DataFrame(out)


def _week_base(p: pd.Series) -> dict:
    return {"account": p["account"], "symbol": p["symbol"], "kind": p["kind"], "contracts": p["qty"], "strike": p["strike"],
            "expiry": p["expiry"], "price": p["price"]}


def _date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def held_events(positions: pd.DataFrame, info: pd.DataFrame, today: date | None = None, days: int = 7) -> pd.DataFrame:
    """Names you hold shares or options in that report earnings or go ex-dividend within `days`."""
    today = today or date.today()
    if positions.empty or info.empty:
        return pd.DataFrame(columns=["symbol", "event", "date"])
    held = set(positions["symbol"])
    out = []
    for _, r in info.iterrows():
        if r["symbol"] not in held:
            continue
        for field, label in (("next_earnings", "earnings"), ("ex_div", "ex-dividend")):
            when = _date(r.get(field))
            if when is not None and today <= when <= today + timedelta(days=days):
                out.append({"symbol": r["symbol"], "event": label, "date": when.isoformat()})
    return pd.DataFrame(out, columns=["symbol", "event", "date"]).sort_values("date") if out else pd.DataFrame(columns=["symbol", "event", "date"])


# ---------------------------------------------------------------- cash vs collateral
def collateral_by_account(positions: pd.DataFrame, cash: dict | None = None) -> pd.DataFrame:
    """Per account: cash, collateral committed by live short puts (strike x 100 x contracts), what is left and a state."""
    cash = cash if cash is not None else get_config().puts.cash
    puts = positions[(positions["kind"] == "short put") & (positions["status"] != "expired")] if not positions.empty else positions
    committed: dict[str, float] = {}
    for _, p in puts.iterrows():
        committed[p["account"]] = committed.get(p["account"], 0.0) + float(p["strike"]) * LOT * float(p["qty"])
    rows = []
    for account in sorted(set(committed) | set(cash)):
        have, need = cash.get(account), committed.get(account, 0.0)
        state = "unset" if have is None else "over" if need > have else "ok"
        rows.append({"account": account, "cash": have, "committed": need, "free": None if have is None else have - need, "state": state})
    return pd.DataFrame(rows, columns=["account", "cash", "committed", "free", "state"])

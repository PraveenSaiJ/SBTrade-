"""Track long options you bought from the buy ideas: current price, P&L, and exit signals against the idea's levels.

The levels (entry, stop, T1, T2) are stock prices saved with the idea. Calls exit when the stock falls to the stop and
take profit as it rises to T1/T2; puts are the mirror image. Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from tool.data.base import DataSource


def _num(value: object) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(out) else float(out)


def quote_option(source: DataSource, symbol: str, opt_type: str, strike: float, expiry: date) -> tuple[float | None, float | None]:
    """(option mid, stock price). The mid falls back to the last trade; None when the strike is outside the chain."""
    spot = source.spot(symbol)
    chain = source.option_chain(symbol, expiry=expiry, right="C" if opt_type == "call" else "P")
    match = chain[(chain["strike"] == strike) & (chain["expiry"] == expiry)] if len(chain) else chain
    if match.empty:
        return None, spot
    row = match.iloc[0]
    bid, ask = _num(row.get("bid")), _num(row.get("ask"))
    mid = (bid + ask) / 2 if bid is not None and ask is not None and ask > 0 else _num(row.get("last"))
    return mid, spot


def assess_position(pos: dict, spot: float | None, mid: float | None, today: date) -> tuple[str, str]:
    """(signal, why) for one open position. Signals: EXIT, TAKE PROFIT, EXPIRING, EXPIRED, HOLD."""
    expiry = date.fromisoformat(str(pos["expiry"])[:10])
    if expiry < today:
        return "EXPIRED", "past its expiry date: close it out in your log"
    sign = 1 if pos["opt_type"] == "call" else -1
    stop, t1, t2, stop_opt = (_num(pos.get(k)) for k in ("stop_level", "t1", "t2", "stop_option"))
    if spot is not None:
        if stop is not None and sign * (spot - stop) <= 0:
            return "EXIT", f"stock ${spot:.2f} is at or through the stop ${stop:.2f}"
        if t2 is not None and sign * (spot - t2) >= 0:
            return "TAKE PROFIT", f"stock ${spot:.2f} reached T2 ${t2:.2f}: sell the rest"
        if t1 is not None and sign * (spot - t1) >= 0:
            return "TAKE PROFIT", f"stock ${spot:.2f} reached T1 ${t1:.2f}: consider selling half"
    if mid is not None and stop_opt is not None and mid <= stop_opt:
        return "EXIT", f"option ${mid:.2f} is at or below its stop ${stop_opt:.2f}"
    dte = (expiry - today).days
    if dte <= 2:
        return "EXPIRING", f"{dte} day(s) left: time value is almost gone"
    return "HOLD", "between the stop and T1" if spot is not None else "no live price"


def refresh_marks(store, source: DataSource) -> tuple[int, list[str]]:
    """Reprice every open position and save the marks. Returns (updated, errors)."""
    updated, errors = 0, []
    positions = store.bought_options("open")
    for _, p in positions.iterrows():
        try:
            mid, spot = quote_option(source, p["symbol"], p["opt_type"], float(p["strike"]), date.fromisoformat(p["expiry"]))
        except Exception as exc:
            errors.append(f"{p['symbol']}: {exc}")
            continue
        store.mark_bought(int(p["id"]), mid, spot)
        updated += 1
    return updated, errors

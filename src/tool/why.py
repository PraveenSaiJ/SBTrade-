"""Plain-English comment for each recommendation: why the app is suggesting it, built from the saved numbers.

Pure functions over a row (dict or Series) so the dashboard can add a "Why recommended" column to every table.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

import pandas as pd


def _f(row: object, key: str) -> float | None:
    value = pd.to_numeric(row.get(key), errors="coerce") if hasattr(row, "get") else None
    return None if value is None or pd.isna(value) else float(value)


def _s(row: object, key: str) -> str:
    value = row.get(key) if hasattr(row, "get") else None
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def swing_why(row: object) -> str:
    kind, spot, atr, sma = _s(row, "kind"), _f(row, "spot"), _f(row, "atr"), _f(row, "sma20")
    chg, vol, entry, rr = _f(row, "change_20d"), _f(row, "volume_ratio"), _f(row, "entry"), _f(row, "rr")
    high = _f(row, "prior_high")
    trend = f"{chg:+.0%} over 20 days" if chg is not None else "trend unknown"
    if kind == "breakout":
        text = f"Closed above its prior 20-day high{f' (${high:.2f})' if high else ''}"
        text += f" on {vol:.1f}x average volume" if vol else ""
        text += f"; {trend}. Buy the retest of ${entry:.2f}." if entry else f"; {trend}."
    elif kind == "bull_flag":
        broke = _f(row, "flag_broken")
        text = "Sharp rise, then a short quiet pause under the flag high" + (
            f" ${entry:.2f}" if entry else "") + (
            "; it has closed above it" if broke else "; buy a break above it")
        text += f" ({trend})."
    elif kind == "ma_pullback":
        sma20 = _f(row, "sma20")
        avg = "20-day" if sma20 is not None and entry is not None and abs(entry - sma20) < 1e-6 else "50-day"
        text = f"Uptrend ({trend}) that dipped to its {avg} average" + (f" ${entry:.2f}" if entry else "") + " and held."
    elif kind == "pullback":
        text = f"Uptrend ({trend}, above its 20-day average)"
        text += f" pulling back to support near ${entry:.2f}, within one ATR." if entry else " pulling back to support."
    elif kind == "extended":
        stretch = (spot - sma) / atr if spot and sma and atr else None
        text = f"Strong ({trend}) but stretched{f', {stretch:.1f} ATR above the 20-day average' if stretch else ''}."
        text += f" Wait for a dip to ${entry:.2f}." if entry else ""
    elif kind == "failed":
        text = f"Closed above its 20-day high but the price is back under ${entry:.2f}: the breakout failed."
    elif kind == "weak_breakout":
        text = f"Closed above its 20-day high but on thin volume{f' ({vol:.1f}x average)' if vol else ''}; not actionable."
    elif kind == "downtrend":
        text = f"Downtrend ({trend}): bearish setup, avoid longs."
    elif kind == "range":
        text = f"Going sideways ({trend}); no edge."
    else:
        return ""
    res = _f(row, "resistance")
    if res:
        text += f" Overhead resistance ${res:.2f}."
    if rr is not None:
        text += f" Reward:risk {rr:.1f}."
    return text


def buy_why(row: object, setup: object | None, family: str) -> str:
    """Comment for a long call/put idea. `setup` is the stock's latest swing-setup row, if any."""
    put = family.endswith("put")
    entry, stop, t1, mult = _f(row, "entry_level"), _f(row, "stop_level"), _f(row, "t1"), _f(row, "multiple")
    delta, cost = _f(row, "delta"), _f(row, "cost")
    parts = []
    if family == "leaps_call":
        parts.append(f"Long-dated stock-replacement call ({_s(row, 'idea_source') or 'bullish setup'})")
    elif family == "earnings_call":
        parts.append(f"Event trade into earnings ({_s(row, 'idea_source') or 'reports soon'})")
    elif family == "directional_call":
        parts.append(f"Directional call, 14-60 days out ({_s(row, 'idea_source') or 'bullish setup'})")
    elif family == "odte_call":
        parts.append("Same-day call planned from today's pivot levels: momentum play, expires today")
    elif family.startswith("mag7"):
        parts.append("MAG7 name, always tracked" + (f"; {swing_why(setup)}" if setup is not None and swing_why(setup) else ""))
    elif setup is not None and swing_why(setup):
        parts.append(swing_why(setup).rstrip("."))
    else:
        parts.append("Bearish trend" if put else "Bullish setup")
    if family.startswith("lotto"):
        parts.append(f"cheap lottery ticket{f' (${cost:,.0f} per contract)' if cost else ''}, small size only")
    elif delta is not None:
        parts.append(f"delta {abs(delta):.2f} gives real exposure")
    if entry:
        parts.append(f"buy {'below' if put else 'above'} ${entry:.2f}" + (f", stop ${stop:.2f}" if stop else ""))
    if t1 and mult:
        parts.append(f"T1 ${t1:.2f} would pay about {mult:.1f}x")
    return "; ".join(parts) + "."


def sell_why(row: object, family: str) -> str:
    """Comment for a sold option (naked call/put, cash-secured put, wheel put)."""
    delta, ret, ann = _f(row, "delta"), _f(row, "return_pct"), _f(row, "annualized")
    prem, strike, eff = _f(row, "premium"), _f(row, "strike"), _f(row, "effective_cost")
    chance = 1 - abs(delta) if delta is not None else None
    keep = f"about {chance:.0%} chance it expires worthless and you keep the premium" if chance else "premium collected up front"
    pay = f"${prem:,.0f} per contract" if prem else "a premium"
    ret_txt = f" ({ret:.1%}{f', {ann:.0%} annualized' if ann else ''})" if ret else ""
    if family in ("csp", "wheel"):
        base = f"Get paid {pay}{ret_txt} to agree to buy at ${strike:.2f}" if strike else f"Get paid {pay}{ret_txt}"
        tail = f"; if assigned your cost is ${eff:.2f}" if eff else ""
        return f"{base}{tail}. Strike sits at or below weekly support; {keep}."
    if family == "naked_put":
        return f"Sell the ${strike:.2f} put for {pay}{ret_txt}: {keep}. Needs margin; loss grows if the stock falls through the strike."
    if family == "naked_call_breakout":
        return f"Stock is near a breakout, so this is high risk: {pay}{ret_txt}, {keep}. Close if the 20-day high breaks."
    if family == "naked_call":
        return f"Sell the ${strike:.2f} call above resistance for {pay}{ret_txt}: {keep}. Unlimited risk if the stock rallies."
    return ""

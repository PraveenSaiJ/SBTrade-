"""Review of what you already hold: fundamentals, valuation, weight, a conviction score, and overlapping names.

Everything is computed from data already fetched (Schwab fundamentals and quotes, stored daily bars, cached Finnhub metrics).
Funds and leveraged ETFs have no fundamentals; they keep their price-based columns and show no conviction score rather than a
made-up one. Weights are shares of the stock portfolio at the latest price (cash and options are not included).
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tool.config import LongTermConfig, get_config
from tool.indicators import sma
from tool.longterm import quality_checks, trend_checks


@dataclass(frozen=True)
class HoldingReview:
    symbol: str
    shares: float
    price: float | None
    value: float | None
    weight: float | None  # share of the stock portfolio
    gain_pct: float | None  # price over average cost, minus 1
    rev_growth: float | None  # TTM, as a fraction
    eps_growth: float | None
    forward_pe: float | None
    pe_5y: float | None  # mean of the last five annual P/E values
    pe_vs_avg: float | None  # forward P/E over its 5-year average: below 1 = cheaper than its own history
    fcf_margin: float | None
    drawdown: float | None  # price over the 52-week high, minus 1 (negative)
    dist_sma200: float | None  # price over the 200-day average, minus 1
    quality: int | None  # 0-6
    trend: int | None  # 0-3
    conviction: float | None  # 0-100, None for funds and names with too little data
    fund: bool  # no fundamentals (ETF, fund, leveraged product)
    concentrated: bool  # weight above the configured maximum


def _num(x: object) -> float | None:
    out = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(out) else float(out)


def _pct(x: object) -> float | None:
    """Schwab and Finnhub give growth as a percent number (17.79 = 17.79%); return a fraction."""
    v = _num(x)
    return None if v is None else v / 100


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def is_fund(fundamentals: dict | None) -> bool:
    """No usable fundamentals: an ETF, a fund or a leveraged product."""
    f = fundamentals or {}
    return not any((_num(f.get(k)) or 0) != 0 for k in ("revChangeTTM", "epsTTM", "peRatio", "operatingMarginTTM"))


def conviction_score(
    quality: int | None, trend: int | None, rev_growth: float | None, eps_growth: float | None,
    pe_vs_avg: float | None, fcf_margin: float | None, drawdown: float | None, dist_sma200: float | None,
    cfg: LongTermConfig | None = None,
) -> float | None:
    """0-100 from six parts, each scaled 0-1 (weights in config). A part without data is dropped and the rest re-weighted;
    with under `min_data_share` of the weight available the answer is None."""
    cfg = cfg or get_config().longterm
    parts: dict[str, float | None] = {
        "quality": None if quality is None else quality / 6,
        "trend": None if trend is None else trend / 3,
        "growth": None,
        "valuation": None if pe_vs_avg is None else _clip((cfg.pe_rich - pe_vs_avg) / (cfg.pe_rich - cfg.pe_cheap)),
        "fcf": None if fcf_margin is None else _clip(fcf_margin / cfg.fcf_cap),
        "resilience": None,
    }
    growth = [g for g in (rev_growth, eps_growth) if g is not None]
    if growth:
        parts["growth"] = sum(_clip(g / cfg.growth_cap) for g in growth) / len(growth)
    resilience = []
    if drawdown is not None:
        resilience.append(1 - _clip(-drawdown / cfg.drawdown_cap))
    if dist_sma200 is not None:
        resilience.append(_clip(dist_sma200 / cfg.sma200_cap))
    if resilience:
        parts["resilience"] = sum(resilience) / len(resilience)
    weights = cfg.conviction_weights
    available = {k: v for k, v in parts.items() if v is not None}
    have = sum(weights[k] for k in available)
    if have < cfg.min_data_share * sum(weights.values()):
        return None
    return 100 * sum(weights[k] * v for k, v in available.items()) / have


def review_holdings(
    holdings, prices: dict[str, float], fundamentals: dict[str, dict], metrics: dict[str, dict],
    bars: dict[str, pd.DataFrame], cfg: LongTermConfig | None = None,
) -> list[HoldingReview]:
    """One review per held symbol (shares and cost summed over accounts), biggest weight first."""
    cfg = cfg or get_config().longterm
    symbols = sorted({sym for sym, _ in holdings.positions()})
    rows = []
    for sym in symbols:
        lots = [l for l in holdings.lots if l.symbol == sym]
        shares = sum(l.shares for l in lots)
        cost = sum(l.shares * l.cost for l in lots)
        b = bars.get(sym)
        closes = b["close"] if b is not None and not b.empty else pd.Series(dtype=float)
        price = prices.get(sym) or (float(closes.iloc[-1]) if len(closes) else None)
        f, m = fundamentals.get(sym) or {}, metrics.get(sym) or {}
        fund = is_fund(f)
        high52 = _num(f.get("high52")) or (float(closes.tail(252).max()) if len(closes) else None)
        sma200 = float(sma(closes, 200).iloc[-1]) if len(closes) >= 200 else None
        sma50 = float(sma(closes, 50).iloc[-1]) if len(closes) >= 50 else None
        drawdown = price / high52 - 1 if price and high52 else None
        dist = price / sma200 - 1 if price and sma200 else None
        quality = trend = None
        if not fund:
            quality = sum(ok for _, ok in quality_checks(f))
        if price and sma50 and sma200 and high52:
            trend = sum(ok for _, ok in trend_checks(price, sma50, sma200, high52))
        rev = _pct(f.get("revChangeTTM")) if not fund else None
        rev = rev if rev is not None else _pct(m.get("rev_growth_ttm"))
        eps = _pct(f.get("epsChangePercentTTM")) if not fund else None
        eps = eps if eps is not None else _pct(m.get("eps_growth_ttm"))
        fwd, avg = _num(m.get("forward_pe")), _num(m.get("pe_5y"))
        ratio = fwd / avg if fwd and avg and fwd > 0 and avg > 0 else None
        fcf = _num(m.get("fcf_margin"))
        rows.append({
            "symbol": sym, "shares": shares, "price": price, "value": shares * price if price else None,
            "gain_pct": price / (cost / shares) - 1 if price and shares and cost else None, "rev": rev, "eps": eps,
            "fwd": fwd, "avg": avg, "ratio": ratio, "fcf": fcf, "dd": drawdown, "dist": dist, "q": quality, "t": trend,
            "fund": fund,
        })
    total = sum(r["value"] for r in rows if r["value"])
    out = []
    for r in rows:
        weight = r["value"] / total if r["value"] and total else None
        conviction = None if r["fund"] else conviction_score(r["q"], r["t"], r["rev"], r["eps"], r["ratio"], r["fcf"], r["dd"], r["dist"], cfg)
        out.append(HoldingReview(
            symbol=r["symbol"], shares=r["shares"], price=r["price"], value=r["value"], weight=weight, gain_pct=r["gain_pct"],
            rev_growth=r["rev"], eps_growth=r["eps"], forward_pe=r["fwd"], pe_5y=r["avg"], pe_vs_avg=r["ratio"],
            fcf_margin=r["fcf"], drawdown=r["dd"], dist_sma200=r["dist"], quality=r["q"], trend=r["t"], conviction=conviction,
            fund=r["fund"], concentrated=weight is not None and weight > cfg.max_weight,
        ))
    return sorted(out, key=lambda x: -(x.weight or 0))


def returns_frame(symbols: list[str], bars: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    """Daily percent returns for the last `window` days, one column per symbol that has enough bars."""
    cols = {}
    for s in symbols:
        b = bars.get(s)
        if b is not None and len(b) >= window + 1:
            cols[s] = b["close"].astype(float).pct_change().iloc[-window:].reset_index(drop=True)
    return pd.DataFrame(cols)


def correlated_pairs(returns: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs of columns whose return correlation is at least `threshold`, highest first (a, b, corr)."""
    if returns.shape[1] < 2:
        return pd.DataFrame(columns=["a", "b", "corr"])
    corr = returns.corr()
    names = list(corr.columns)
    pairs = [
        {"a": names[i], "b": names[j], "corr": float(corr.iloc[i, j])}
        for i in range(len(names)) for j in range(i + 1, len(names))
        if not np.isnan(corr.iloc[i, j]) and corr.iloc[i, j] >= threshold
    ]
    return pd.DataFrame(pairs, columns=["a", "b", "corr"]).sort_values("corr", ascending=False, ignore_index=True)


def keep_suggestion(a: str, b: str, reviews: dict[str, HoldingReview]) -> str:
    """Which of two overlapping names carries more conviction (the other is the one to look at trimming)."""
    ca, cb = reviews[a].conviction if a in reviews else None, reviews[b].conviction if b in reviews else None
    if ca is None or cb is None:
        return "no conviction score for one of them: compare by hand"
    if abs(ca - cb) < 3:
        return "similar conviction: pick the one you understand better"
    keep, trim = (a, b) if ca > cb else (b, a)
    return f"keep {keep} (conviction {max(ca, cb):.0f}), look at trimming {trim} ({min(ca, cb):.0f})"

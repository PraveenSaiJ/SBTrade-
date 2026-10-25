"""Cash-secured puts at support: the put whose strike sits at or under the nearest support, 7-45 days out.

Sits next to the guardrail engine in puts.py (strike at or under weekly S1, delta 0.15-0.25); the dashboard shows both.
Rules: strike at or below the nearest support and below the stock price; |delta| 0.20-0.30 preferred, best annualized
return on collateral among those; an earnings date before expiry gates the row (shown only with the "Earnings play"
switch and a rank above 60). Rank is the IV rank once 60 days of IV exist, otherwise a labelled HV rank.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd

from tool.config import PutConfig, get_config
from tool.covered_calls import earnings_guard
from tool.data.base import DataSource
from tool.data.earnings import FinnhubEvents
from tool.holdings import LOT_SIZE
from tool.indicators import hv_rank
from tool.levels import NY, compute_levels
from tool.puts import MAX_SPREAD_PCT, SUSPECT_ANNUALIZED, PutRec
from tool.supports import nearest_support, support_levels

RankFn = Callable[[str, pd.Series], "tuple[float | None, str]"]
BAND_MID = 0.25  # with nothing in the delta band, take the strike whose |delta| is nearest this


def window_expiries(now: datetime, cfg: PutConfig | None = None) -> list[date]:
    """Fridays from `min_dte` to `max_dte` days out, at most `max_expiries`: the first two, the monthly, the last."""
    cfg = cfg or get_config().puts
    return friday_window(now, cfg.min_dte, cfg.max_dte, cfg.max_expiries)


def friday_window(now: datetime, min_dte: int, max_dte: int, max_expiries: int) -> list[date]:
    """Fridays `min_dte` to `max_dte` days out: the first two, the monthly (15th-21st) and the last, at most `max_expiries`."""
    today = now.date()
    fridays = []
    day = today + timedelta(days=min_dte)
    while (day - today).days <= max_dte:
        if day.weekday() == 4:
            fridays.append(day)
        day += timedelta(days=1)
    if not fridays:
        return []
    monthly = next((d for d in fridays if 15 <= d.day <= 21), None)
    # in priority order: nearest, the monthly, the farthest, then the second Friday; the cap keeps the first few of these
    wanted = [fridays[0], monthly, fridays[-1], fridays[1] if len(fridays) > 1 else None]
    picked: list[date] = []
    for d in wanted:
        if d is not None and d not in picked:
            picked.append(d)
    return sorted(picked[:max_expiries])


def pick_support_put(
    chain: pd.DataFrame, support: float, spot: float, dte: int, cfg: PutConfig | None = None
) -> tuple[pd.Series, bool] | None:
    """(row, in the delta band) or None. Strikes at or under both the support and the stock price, with a bid."""
    cfg = cfg or get_config().puts
    if dte <= 0:
        return None
    cap = min(support, spot)
    bids = pd.to_numeric(chain["bid"], errors="coerce")
    eligible = chain[(chain["strike"] <= cap) & (bids > 0)].copy()
    if eligible.empty:
        return None
    eligible["abs_delta"] = pd.to_numeric(eligible["delta"], errors="coerce").abs()
    eligible["annualized"] = eligible["bid"].astype(float) / eligible["strike"] * 365 / dte
    in_band = eligible[eligible["abs_delta"].between(cfg.support_delta_low, cfg.support_delta_high)]
    if not in_band.empty:
        return in_band.sort_values("annualized", ascending=False).iloc[0], True
    known = eligible[eligible["abs_delta"].notna()]
    if not known.empty:
        return known.iloc[(known["abs_delta"] - BAND_MID).abs().argsort()].iloc[0], False
    return eligible.sort_values("annualized", ascending=False).iloc[0], False


def default_rank(symbol: str, closes: pd.Series) -> tuple[float | None, str]:
    cfg = get_config().puts
    return hv_rank(closes, cfg.hv_window, cfg.hv_lookback), "HV"


def recommend_support_puts(
    symbols: list[str],
    source: DataSource,
    now: datetime | None = None,
    events: FinnhubEvents | None = None,
    rank_fn: RankFn | None = None,
    setup_kinds: dict[str, str] | None = None,
    cfg: PutConfig | None = None,
) -> list[PutRec]:
    """One row per symbol and expiry in the window. `setup_kinds` maps symbol -> latest swing setup, to skip downtrends."""
    cfg = cfg or get_config().puts
    now = now or datetime.now(NY)
    rank_fn = rank_fn or default_rank
    out: list[PutRec] = []
    expiries = window_expiries(now, cfg)
    for symbol in symbols:
        if (setup_kinds or {}).get(symbol) == "downtrend":
            out.append(PutRec(symbol, "csp_support", "skip", "downtrend (20-day change below -3%): no puts on falling names"))
            continue
        try:
            bars = source.daily_bars(symbol, lookback_days=400)
            spot = source.spot(symbol)
            levels = compute_levels(bars, now=now, spot=spot, symbol=symbol)
        except Exception as exc:
            out.append(PutRec(symbol, "csp_support", "skip", f"data error: {exc}"))
            continue
        found = nearest_support(support_levels(bars, spot, levels, cfg, now))
        if found is None:
            out.append(PutRec(symbol, "csp_support", "skip", "no support level under the stock price", spot=spot))
            continue
        support, label = found
        rank, rank_source = rank_fn(symbol, bars["close"])
        for expiry in expiries:
            out.append(_one(symbol, source, now, events, expiry, spot, support, label, rank, rank_source, cfg))
    return out


def _one(
    symbol: str, source: DataSource, now: datetime, events: FinnhubEvents | None, expiry: date, spot: float,
    support: float, label: str, rank: float | None, rank_source: str, cfg: PutConfig,
) -> PutRec:
    # expiry_type is part of the option_recs key (run, family, symbol, expiry_type): one label per expiry keeps them all
    base = dict(symbol=symbol, mode="csp_support", expiry_type=f"support {expiry.isoformat()}", spot=spot,
                support=support, support_label=label, rank=rank, rank_source=rank_source)
    try:
        chain = source.option_chain(symbol, expiry=expiry, right="P")
    except Exception as exc:
        return PutRec(status="skip", reason=f"chain error: {exc}", expiry=expiry, **base)
    if chain.empty:
        return PutRec(status="skip", reason="empty option chain", expiry=expiry, **base)
    actual = chain["expiry"].iloc[0] if "expiry" in chain else expiry
    dte = (actual - now.date()).days
    if not cfg.min_dte <= dte <= cfg.max_dte:
        return PutRec(status="skip", reason=f"nearest expiry {actual} is {dte} days out, outside {cfg.min_dte}-{cfg.max_dte}",
                      expiry=actual, **base)
    picked = pick_support_put(chain, support, spot, dte, cfg)
    if picked is None:
        return PutRec(status="skip", reason=f"no put at or under the {label} (${support:.2f}) with a bid",
                      expiry=actual, **base)
    row, in_band = picked
    strike, bid = float(row["strike"]), float(row["bid"])
    delta = None if pd.isna(row.get("delta")) else float(row["delta"])
    roc = bid / strike
    annualized = roc * 365 / dte
    notes = [f"cash-secured: keep ${strike * LOT_SIZE:,.0f} in cash"]
    if not in_band:
        notes.append(
            f"no put at or under the {label} has |delta| {cfg.support_delta_low:.2f}-{cfg.support_delta_high:.2f}; "
            "the nearest to that band is shown"
        )
    ask = None if pd.isna(row.get("ask")) else float(row["ask"])
    if ask is not None and (ask + bid) > 0 and (ask - bid) / ((ask + bid) / 2) > MAX_SPREAD_PCT:
        notes.append(f"wide spread ({(ask - bid) / ((ask + bid) / 2):.0%} of mid): use a limit near mid")
    if annualized > SUSPECT_ANNUALIZED:
        notes.append(f"SUSPECT QUOTE: {annualized:.0%} annualized is implausible: verify the bid before trading")
    status, reason = "recommend", "ok"
    hits, earnings_note = earnings_guard(events, symbol, now.date(), actual)
    if hits:
        status, reason = "earnings_gated", f"earnings {hits[0].date.isoformat()} before expiry {actual.isoformat()}"
        notes.append("earnings play: gap risk through the print; shown only with the switch on and a high rank")
    else:
        notes.append(earnings_note)
    return PutRec(
        status=status, reason=reason, expiry=actual, strike=strike, bid=bid, delta=delta,
        premium_per_contract=bid * LOT_SIZE, collateral_per_contract=strike * LOT_SIZE, return_on_collateral=roc,
        annualized_return=annualized, effective_cost=strike - bid, max_loss_per_contract=(strike - bid) * LOT_SIZE,
        dte=dte, break_even=strike - bid, prob_otm=None if delta is None else 1 - abs(delta), notes=notes, **base,
    )


def visible_rows(rows: pd.DataFrame, earnings_play: bool, rank_min: float | None = None) -> pd.DataFrame:
    """Recommended rows, plus earnings-gated ones when the switch is on and their rank is above the minimum."""
    rank_min = get_config().puts.earnings_rank_min if rank_min is None else rank_min
    ok = rows["status"] == "recommend"
    if earnings_play:
        gated = (rows["status"] == "earnings_gated") & (pd.to_numeric(rows["rank"], errors="coerce") > rank_min)
        return rows[ok | gated]
    return rows[ok]


def collateral_check(selected: float, cash: float | None) -> tuple[str, str]:
    """('ok' | 'over' | 'unset', message) for the selected puts' collateral against the account's cash."""
    if cash is None:
        return "unset", (
            f"Collateral needed: ${selected:,.0f}. Set `puts: cash: {{account: dollars}}` in config/settings.yaml "
            "to compare it with your cash."
        )
    if selected > cash:
        return "over", f"Over cash: ${selected:,.0f} of collateral against ${cash:,.0f} available (short ${selected - cash:,.0f})."
    return "ok", f"Within cash: ${selected:,.0f} of ${cash:,.0f} available (${cash - selected:,.0f} left)."

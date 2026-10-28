"""Calls to BUY into an earnings report: near-the-money calls that expire just after the report.

An event trade, not a swing setup: the stock's entry/stop/targets do not apply. Each row weighs the break-even against the
market's expected move to expiry (which already prices the report), flags the IV crush, and shows the whole premium as the
max loss. Recommend-only. Nothing is ordered.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from tool.buy_options import BuyRec
from tool.config import AccountConfig, CallConfig, get_config
from tool.data.base import DataSource
from tool.data.earnings import EarningsEvent
from tool.directional_calls import (
    THIN_OPEN_INTEREST, RankFn, _num, _spread_flag, call_metrics, pick_delta_call, position_contracts,
)
from tool.holdings import LOT_SIZE
from tool.levels import NY
from tool.support_puts import default_rank

FAMILY = "earnings_call"


def reaction_day(report: date, hour: str | None) -> date:
    """First session whose close reflects the report: the same day before the open (bmo), otherwise the next day."""
    return report if hour == "bmo" else report + timedelta(days=1)


def expiry_target(report: date, hour: str | None) -> date:
    """The first Friday on or after the reaction day (a weekend reaction rolls to Monday first)."""
    day = reaction_day(report, hour)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day + timedelta(days=(4 - day.weekday()) % 7)


def upcoming_reports(
    events: list[EarningsEvent], today: date, symbols: set[str], within_days: int
) -> dict[str, tuple[date, str | None]]:
    """symbol -> (report date, bmo/amc) for the pool's names reporting in the next `within_days` days (today after the close counts)."""
    out: dict[str, tuple[date, str | None]] = {}
    for e in events:
        if e.symbol not in symbols:
            continue
        pending = e.date > today or (e.date == today and e.hour != "bmo")
        if pending and (e.date - today).days <= within_days and (e.symbol not in out or e.date < out[e.symbol][0]):
            out[e.symbol] = (e.date, e.hour)
    return out


def recommend_earnings_calls(
    pool: list[str], events: list[EarningsEvent], source: DataSource, now: datetime | None = None,
    rank_fn: RankFn | None = None, cfg: CallConfig | None = None, account: AccountConfig | None = None,
) -> list[BuyRec]:
    """One near-the-money call per name reporting soon, expiring on the first Friday after the reaction."""
    cfg = cfg or get_config().calls
    account = account or get_config().account
    now = now or datetime.now(NY)
    rank_fn = rank_fn or default_rank
    reports = upcoming_reports(events, now.date(), set(pool), cfg.earnings_days)
    ordered = sorted(reports.items(), key=lambda kv: (kv[1][0], kv[0]))[: cfg.max_earnings]
    out: list[BuyRec] = []
    for symbol, (report, hour) in ordered:
        base = dict(symbol=symbol, side="call", style="earnings")
        when = {"bmo": "before the open", "amc": "after the close"}.get(hour or "", "time not given")
        try:
            bars = source.daily_bars(symbol, lookback_days=400)
            spot = source.spot(symbol)
            target = expiry_target(report, hour)
            chain = source.option_chain(symbol, expiry=target, right="C", window_days=14, strike_count=20)
        except Exception as exc:
            out.append(BuyRec(status="skip", reason=f"data error: {exc}", expiry_type=f"earnings {report.isoformat()}", **base))
            continue
        if chain.empty:
            out.append(BuyRec(status="skip", reason="empty option chain", expiry_type=f"earnings {report.isoformat()}", spot=spot, **base))
            continue
        expiry = chain["expiry"].iloc[0] if "expiry" in chain else target
        expiry_type = f"earnings {report.isoformat()}"
        if expiry < reaction_day(report, hour):
            out.append(BuyRec(status="skip", reason=f"no expiry after the report ({expiry} is too early)",
                              expiry_type=expiry_type, expiry=expiry, spot=spot, **base))
            continue
        row = pick_delta_call(chain, cfg.earnings_delta_low, cfg.earnings_delta_high)
        if row is None:
            out.append(BuyRec(status="skip", reason=f"no call with delta {cfg.earnings_delta_low:.2f}-{cfg.earnings_delta_high:.2f} "
                              f"and an ask on {expiry}", expiry_type=expiry_type, expiry=expiry, spot=spot, **base))
            continue
        dte = (expiry - now.date()).days
        ask = float(row["ask"])
        m = call_metrics(row, spot, dte)
        rank, rank_source = rank_fn(symbol, bars["close"])
        flags = [f"IV CRUSH: reports {report.isoformat()} {when}; implied volatility usually falls 30-50% right after, "
                 "so the stock must move more than the market expects"
                 + (f" (rank {rank:.0f}, {rank_source})" if rank is not None else "")]
        if m["be_vs_move"] is not None and m["be_vs_move"] > 1.0:
            flags.append(f"break-even is {m['be_vs_move']:.1f}x the expected move: needs more than the market expects")
        spread = _spread_flag(row)
        if spread:
            flags.append(spread)
        oi = _num(row.get("open_interest"))
        if oi is not None and oi < THIN_OPEN_INTEREST:
            flags.append(f"thin open interest ({oi:.0f})")
        out.append(BuyRec(
            status="recommend", reason="ok", expiry_type=expiry_type, expiry=expiry, strike=float(row["strike"]), ask=ask,
            delta=_num(row.get("delta")), spot=spot, cost_per_contract=ask * LOT_SIZE, trigger="EVENT",
            entry_rule=f"Buy before the close on {report.isoformat()} (report {when}); a limit near mid",
            stop_rule="No price stop: the whole premium is at risk. Sell the morning after the report, whatever the move.",
            target_rule=f"Sell at the first open after the report; a gain needs {symbol} above ${m['breakeven']:.2f} at expiry or enough IV left",
            dte=dte, rank=rank, rank_source=rank_source, expected_move=m["expected_move"],
            expected_move_pct=m["expected_move_pct"], breakeven=m["breakeven"], breakeven_pct=m["breakeven_pct"],
            be_vs_move=m["be_vs_move"], theta_pct=m["theta_pct"], iv=m["iv"], extrinsic_pct=m["extrinsic_pct"],
            contracts=position_contracts(ask, account), max_loss_per_contract=ask * LOT_SIZE, flags=flags,
            idea_source=f"reports {report.isoformat()} {when}", notes=[f"{symbol} is ${spot:.2f}; reports {report.isoformat()} {when}"],
            **base,
        ))
    return out

"""Expected move into earnings: the at-the-money straddle for the first expiry after the next report.

`expected_move` is the ATM call mid plus put mid (indicators.expected_move_from_straddle); `expected_move_pct` divides
it by the stock price. It is the market's own estimate of how far the stock moves on the report, used to compare
against a trade's stop distance. Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

from tool.config import ScreenerConfig, get_config
from tool.indicators import expected_move_from_straddle
from tool.screener import ACTIONABLE, Setup, apply_filters

MORNING_HOURS = ("bmo", "dmh")  # the stock reacts on the report date itself


class ExpectedMoveUnavailable(Exception):
    """No expected move could be computed; the message says why (it goes in the gaps log)."""


@dataclass(frozen=True)
class EarningsMove:
    expiry: date
    strike: float
    call_mid: float
    put_mid: float
    straddle: float  # the expected move in dollars
    pct: float  # straddle / spot
    wide_spread: bool  # a leg quoted wider than max_spread_pct of its mid: the number is rougher than usual


def expiry_target(report: date, hour: str | None) -> date:
    """The earliest expiry that contains the reaction: the report date for a before-open or during-hours report,
    the next day for an after-close or unknown-hour one. The chain lookup then picks the first expiry on or after it."""
    return report if hour in MORNING_HOURS else report + timedelta(days=1)


def _mid(row: pd.Series) -> tuple[float | None, float | None]:
    """(mid, spread as a share of mid) from a two-sided quote, else (None, None)."""
    bid, ask = pd.to_numeric(row.get("bid"), errors="coerce"), pd.to_numeric(row.get("ask"), errors="coerce")
    if pd.isna(bid) or pd.isna(ask) or ask <= 0 or bid < 0 or ask < bid:
        return None, None
    mid = (float(bid) + float(ask)) / 2
    return (mid, (float(ask) - float(bid)) / mid) if mid > 0 else (None, None)


def expected_move_for(source, symbol: str, spot: float, report: date, hour: str | None) -> EarningsMove:
    """ATM straddle at the first expiry after the report. Raises ExpectedMoveUnavailable with the reason."""
    target = expiry_target(report, hour)
    try:
        calls = source.option_chain(symbol, expiry=target, right="C")
        puts = source.option_chain(symbol, expiry=target, right="P")
    except Exception as exc:
        raise ExpectedMoveUnavailable(f"no option chain on or after {target} ({exc})") from exc
    if calls.empty or puts.empty:
        raise ExpectedMoveUnavailable(f"empty option chain on or after {target}")
    expiry = calls["expiry"].iloc[0]
    if puts["expiry"].iloc[0] != expiry:
        raise ExpectedMoveUnavailable("call and put chains return different expiries")

    max_spread = get_config().options.max_spread_pct
    best = None
    for _, c in calls.iterrows():
        match = puts[puts["strike"] == c["strike"]]
        if match.empty:
            continue
        call_mid, call_spread = _mid(c)
        put_mid, put_spread = _mid(match.iloc[0])
        if call_mid is None or put_mid is None:
            continue
        distance = abs(float(c["strike"]) - spot)
        if best is None or distance < best[0]:
            best = (distance, float(c["strike"]), call_mid, put_mid, max(call_spread, put_spread))
    if best is None:
        raise ExpectedMoveUnavailable("no strike with a two-sided call and put quote")
    _, strike, call_mid, put_mid, spread = best
    straddle = expected_move_from_straddle(call_mid, put_mid)
    return EarningsMove(expiry, strike, call_mid, put_mid, straddle, straddle / spot, spread > max_spread)


def attach_expected_moves(
    setups: list[Setup], source, cfg: ScreenerConfig, today: date, allow_earnings: bool = False,
    show_failed: bool = False,
) -> tuple[list[Setup], dict[str, str]]:
    """Add the expected move to setups that survived the filters and report within `earnings_flag_days`, then re-run
    the filters with it. Returns (setups, {symbol: why it could not be computed})."""
    out: list[Setup] = []
    failures: dict[str, str] = {}
    for s in setups:
        near = s.earnings is not None and 0 <= (s.earnings - today).days <= cfg.earnings_flag_days
        # setups dropped only because earnings fall inside the hold still get a move, so the dashboard's
        # "Allow earnings" switch can judge them against their stop
        droppable = s.excluded is None or s.excluded_code == "earnings_hold"
        if not droppable or not near or s.kind not in (*ACTIONABLE, "failed"):
            out.append(s)
            continue
        try:
            move = expected_move_for(source, s.symbol, s.spot, s.earnings, s.earnings_hour)
        except ExpectedMoveUnavailable as exc:
            failures[s.symbol] = str(exc)
            out.append(s)
            continue
        except Exception as exc:  # a quote failure must not abort the scan
            failures[s.symbol] = f"{type(exc).__name__}: {exc}"
            out.append(s)
            continue
        with_move = replace(
            s, expected_move=move.straddle, expected_move_pct=move.pct, em_expiry=move.expiry,
            notes=[*s.notes, f"expected earnings move ${move.straddle:.2f} ({move.pct:.1%}) to {move.expiry}"
                   + (" (wide spread)" if move.wide_spread else "")],
        )
        out.append(apply_filters(with_move, cfg, today, allow_earnings, show_failed, expected_move_pct=move.pct))
    return out, failures

"""Rows for the `signals` table: every swing recommendation (and exclusion) with all the inputs that produced it."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime

from tool.config import ScreenerConfig
from tool.screener import ACTIONABLE, Setup

TRADABLE = (*ACTIONABLE, "failed")


def _iso(value: object) -> str | None:
    return value.isoformat() if isinstance(value, (date, datetime)) else None


def signal_rows(setups: list[Setup], signal_date: str, cfg: ScreenerConfig, generated_at: str, tab: str = "swing") -> list[dict]:
    """One dict per tradable setup, recommended or excluded, ready for Store.save_signals."""
    rows = []
    snapshot = asdict(cfg)
    for s in setups:
        if s.kind not in TRADABLE:
            continue
        inputs = {
            "spot": s.spot, "last_close": s.last_close, "atr10": s.atr, "atr14": s.atr14, "sma20": s.sma20,
            "sma50": s.sma50, "sma50_rising": s.sma50_rising, "sma200": s.sma200, "rsi14": s.rsi,
            "volume_ratio": s.volume_ratio, "change_20d": s.change_20d, "prior_20d_high": s.prior_high,
            "earnings_next": _iso(s.earnings), "earnings_hour": s.earnings_hour, "last_reported": _iso(s.last_reported),
            "post_earnings": s.post_earnings, "earnings_known": s.earnings_known, "flag_broken": s.flag_broken,
            "expected_move_expiry": _iso(s.em_expiry), "notes": s.notes, "config": snapshot,
        }
        rows.append({
            "signal_date": signal_date, "tab": tab, "symbol": s.symbol, "setup": s.kind,
            "status": "recommended" if s.excluded is None else "excluded", "excluded_code": s.excluded_code,
            "reason": s.excluded, "entry": s.entry, "stop": s.stop, "t1": s.t1, "t2": s.t2, "rr": s.rr, "score": s.score,
            "resistance": s.resistance, "shares": s.shares, "expected_move": s.expected_move,
            "expected_move_pct": s.expected_move_pct, "inputs": json.dumps(inputs, default=str),
            "generated_at": generated_at, "data_as_of": _iso(s.data_as_of), "source": "schwab,finnhub",
        })
    return rows

"""Re-check the saved swing setups against live prices (one batched quote request, no bars, no chains).

The scan runs once, before the open. A breakout can lose its level at 11:00 and the table would still show a clean
setup. This refreshes each recommended setup's spot, recomputes R:R from the new fill, flips a breakout that fell back
under its entry to `failed`, and drops a setup that has run too far above its entry. Setups already excluded are left alone.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from tool.config import ScreenerConfig
from tool.screener import reward_to_risk

RECHECKED_KINDS = ("breakout", "bull_flag", "ma_pullback", "pullback", "extended")


@dataclass
class RecheckResult:
    checked: int = 0
    failed: list[str] = field(default_factory=list)  # breakouts that fell back under the entry
    overextended: list[str] = field(default_factory=list)
    no_quote: list[str] = field(default_factory=list)


def _f(value: object) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(out) else float(out)


def recheck_row(row: pd.Series, spot: float, cfg: ScreenerConfig) -> dict:
    """The updates for one saved setup at a new `spot`: kind, excluded, rr, score, spot."""
    entry, stop, t1 = _f(row["entry"]), _f(row["stop"]), _f(row["t1"])
    kind, excluded, code = row["kind"], None, None
    broken_flag = row["kind"] == "bull_flag" and bool(_f(row.get("flag_broken")))
    rr = reward_to_risk(t1, max(spot, entry), stop) if None not in (entry, stop, t1) else _f(row.get("rr"))

    if stop is not None and spot <= stop and kind != "failed":
        code = "stopped"
        excluded = f"price ${spot:.2f} is already at or under the stop ${stop:.2f}"
    elif (kind == "breakout" or broken_flag) and entry is not None and spot < entry:
        kind, code = "failed", "failed"
        excluded = f"failed: spot ${spot:.2f} is back under the breakout level ${entry:.2f}"
    elif kind in ("breakout", "pullback", "ma_pullback", "bull_flag") and entry is not None:
        atr14 = _f(row.get("atr14"))  # rows saved before this column existed skip the check until the next scan
        if atr14 and spot - entry > cfg.overextension_atr * atr14:
            code = "overextended"
            excluded = (
                f"overextended: ${spot - entry:.2f} above the entry, more than "
                f"{cfg.overextension_atr:g} ATR(14) (${atr14:.2f})"
            )

    score = _f(row.get("score"))
    old_rr = _f(row.get("rr"))
    if score is not None and rr is not None and old_rr is not None:
        # only the R:R term of the score changes with price: swap its old contribution for the new one
        norm = lambda x: max(0.0, min(1.0, x / cfg.rr_norm_cap))  # noqa: E731
        score = score + cfg.w_rr * (norm(rr) - norm(old_rr))
    return {"kind": kind, "excluded": excluded, "excluded_code": code, "rr": rr, "score": score, "spot": spot}


def recheck_setups(store, source, cfg: ScreenerConfig, now: datetime | None = None) -> RecheckResult:
    """Refresh every still-recommended setup from the latest saved scan. One batched quote request per 100 symbols."""
    result = RecheckResult()
    setups = store.latest_setups()
    if setups.empty:
        return result
    live = setups[setups["kind"].isin(RECHECKED_KINDS) & setups["entry"].notna() & setups["excluded"].isna()]
    if live.empty:
        return result
    prices = source.quotes(list(live["symbol"]), ttl=0)
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    for _, row in live.iterrows():
        spot = prices.get(row["symbol"])
        if spot is None:
            result.no_quote.append(row["symbol"])
            continue
        update = recheck_row(row, spot, cfg)
        store.update_setup_recheck(row["scan_date"], row["symbol"], update, stamp)
        store.update_signal_status(
            row["scan_date"], "swing", row["symbol"], update["kind"],
            "excluded" if update["excluded"] else "recommended", update["excluded_code"], update["excluded"],
            update["rr"], update["score"],
        )
        result.checked += 1
        if update["kind"] == "failed":
            result.failed.append(row["symbol"])
        elif update["excluded"]:
            result.overextended.append(row["symbol"])
    return result

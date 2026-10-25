"""Event flags for a covered call: ex-dividend (with early-assignment risk) and manual events before the expiry.

Earnings are handled by covered_calls.earnings_guard. Schwab fundamentals give `dividendDate` (the ex-dividend date of
the latest or next dividend) and `dividendAmount`, an ANNUAL figure; the quarterly payment is that divided by
`quarters_per_year` (a quarterly payer is assumed). Investor days and product events have no free data source, so they
come from the `covered_calls.events` list in config/settings.yaml. Recommend-only.
"""

from __future__ import annotations

from datetime import date, datetime

from tool.config import CoveredCallConfig, get_config

NEAR_THE_MONEY = 1.02  # early-assignment warnings only matter for calls at or within 2% of the money


def _to_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace(" 00:00:00.0", "")).date()
    except ValueError:
        return None


def ex_dividend(symbol: str, fundamentals: dict | None, cfg: CoveredCallConfig) -> tuple[date | None, float]:
    """(ex-dividend date, quarterly dividend per share) or (None, 0.0) when the stock pays nothing."""
    f = (fundamentals or {}).get(symbol) or {}
    annual = f.get("dividendAmount")
    if not isinstance(annual, (int, float)) or annual <= 0:
        return None, 0.0
    return _to_date(f.get("dividendDate")), float(annual) / cfg.quarters_per_year


def call_flags(
    symbol: str, today: date, expiry: date, fundamentals: dict | None = None,
    spot: float | None = None, strike: float | None = None, mid: float | None = None, cfg: CoveredCallConfig | None = None,
) -> list[str]:
    """Flags for selling (or holding) a call that expires on `expiry`: ex-dividend, early-assignment risk, manual events."""
    cfg = cfg or get_config().covered_calls
    flags: list[str] = []
    ex, quarterly = ex_dividend(symbol, fundamentals, cfg)
    if ex is not None and today <= ex <= expiry:
        flags.append(f"ex-dividend {ex.isoformat()} (~${quarterly:.2f}/share) before expiry")
        if None not in (spot, strike, mid) and strike <= spot * NEAR_THE_MONEY:
            extrinsic = max(mid - max(spot - strike, 0.0), 0.0)
            if extrinsic < quarterly:
                flags.append(
                    f"early-assignment risk: time value ${extrinsic:.2f} is below the ${quarterly:.2f} dividend"
                )
    for event in cfg.events:
        when = _to_date(event.get("date"))
        if str(event.get("symbol", "")).upper() == symbol.upper() and when is not None and today <= when <= expiry:
            flags.append(f"event: {event.get('label', 'event')} {when.isoformat()}")
    return flags

"""Whale activity: option contracts with unusually large traded premium or fresh volume.

Schwab chains give aggregate day volume, open interest, last, bid and ask per contract, not individual prints. So a
"whale" here is a contract whose total traded premium (volume x price x 100) is large, and the lean comes from where
the last trade printed against the quote. Treat it as a pointer to where size is trading, not proof of intent.
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from tool.buy_options import INDEX_ETFS, MAG7
from tool.data.base import DataSource
from tool.config import get_config

_W = get_config().whales
WHALE_PREMIUM = _W.whale_premium  # tier "WHALE"
LARGE_PREMIUM = _W.large_premium  # tier "Large"
MIN_VOLUME = _W.min_volume
FRESH_VOL_OI = _W.fresh_vol_oi  # more traded today than the whole open interest: new positions
INDEX_SCALE = _W.index_scale  # SPY/QQQ/IWM trade so much size that the single-stock thresholds would flag everything
EXTRA_NAMES = list(_W.extra_names)
WHALE_UNIVERSE = list(dict.fromkeys(INDEX_ETFS + MAG7 + EXTRA_NAMES))


@dataclass(frozen=True)
class WhaleRow:
    symbol: str
    right: str  # C | P
    expiry: date
    dte: int
    strike: float
    spot: float
    otm_pct: float  # (strike - spot) / spot, signed
    volume: int
    open_interest: int | None
    vol_oi: float | None
    price: float
    premium: float  # volume * price * 100
    side: str  # ask | bid | mid
    lean: str  # bullish | bearish | neutral
    tier: str  # WHALE | Large
    delta: float | None
    iv: float | None
    fresh: bool
    intent: str = "unknown"  # opening | likely opening | closing or rolling | unknown
    burst_volume: int | None = None  # contracts added since the previous scan (None: no comparable scan)
    burst_premium: float | None = None
    is_burst: bool = False  # the premium added in one scan interval is large; NOT a sweep or block (no print data)
    noise: str | None = None  # why the row is hidden by default (deep ITM, dividend capture)


def _num(value: object) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(out) else float(out)


def trade_side(last: float | None, bid: float | None, ask: float | None) -> str:
    """Where the last trade printed: at the ask (buyer paid up), at the bid (seller hit it), or between."""
    if last is None or bid is None or ask is None or ask <= 0:
        return "mid"
    spread = ask - bid
    if last >= ask - 0.2 * spread:
        return "ask"
    if last <= bid + 0.2 * spread:
        return "bid"
    return "mid"


def lean(right: str, side: str) -> str:
    if side == "mid":
        return "neutral"
    bought = side == "ask"
    return "bullish" if (right == "C") == bought else "bearish"


def classify_intent(volume: float, open_interest: float | None) -> str:
    """Opening vs closing from today's volume against the open interest published this morning.

    Volume above open interest means at least (volume - OI) contracts are new. It is a read, not a certainty: the
    next morning's open-interest change is what confirms it."""
    if open_interest is None or open_interest <= 0:
        return "unknown"
    ratio = volume / open_interest
    if ratio >= _W.opening_vol_oi:
        return "opening"
    if ratio >= _W.likely_opening_vol_oi:
        return "likely opening"
    return "closing or rolling"


def noise_reason(
    symbol: str, right: str, strike: float, spot: float, delta: float | None, price: float, today: date,
    fundamentals: dict | None = None,
) -> str | None:
    """Why a flow is probably not a directional view: deep in the money (stock replacement or a hedge), or an ITM call
    with less time value than the dividend just before the ex-date (dividend capture)."""
    if delta is not None and abs(delta) >= _W.deep_itm_delta:
        return f"deep in the money (|delta| {abs(delta):.2f}): stock replacement or a hedge"
    if right == "C" and strike < spot:
        from tool.cc_flags import ex_dividend
        from tool.config import get_config as _cfg

        ex, quarterly = ex_dividend(symbol, fundamentals, _cfg().covered_calls)
        if ex is not None and quarterly > 0 and 0 <= (ex - today).days <= _W.dividend_capture_days:
            if price - (spot - strike) < quarterly:
                return f"dividend capture: ITM call with less time value than the ${quarterly:.2f} dividend (ex {ex})"
    return None


def whales_from_chain(
    symbol: str, chain: pd.DataFrame, spot: float, today: date, previous: dict | None = None,
    fundamentals: dict | None = None, snapshot: list | None = None,
) -> list[WhaleRow]:
    """Whale rows for one chain. `previous` maps (symbol, right, expiry, strike) -> volume from the last comparable scan
    (None = no comparable scan, so no burst); `snapshot`, if given, collects every contract with volume for the next scan."""
    rows: list[WhaleRow] = []
    if chain is None or chain.empty or "volume" not in chain:
        return rows
    scale = INDEX_SCALE if symbol in INDEX_ETFS else 1.0
    for _, c in chain.iterrows():
        volume = _num(c.get("volume"))
        if volume is None or volume < MIN_VOLUME:
            continue
        if snapshot is not None:
            snapshot.append((symbol, str(c["right"]), c["expiry"].isoformat(), float(c["strike"]), int(volume)))
        bid, ask, last, mid = _num(c.get("bid")), _num(c.get("ask")), _num(c.get("last")), _num(c.get("mid"))
        price = last if last and last > 0 else mid
        if price is None or price <= 0:
            continue
        premium = volume * price * 100
        if premium < LARGE_PREMIUM * scale:
            continue
        oi = _num(c.get("open_interest"))
        vol_oi = volume / oi if oi and oi > 0 else None
        side = trade_side(last, bid, ask)
        expiry = c["expiry"]
        strike = float(c["strike"])
        right = str(c["right"])
        delta = _num(c.get("delta"))
        burst_volume = burst_premium = None
        is_burst = False
        if previous is not None:
            burst_volume = max(int(volume) - previous.get((symbol, right, expiry.isoformat(), strike), 0), 0)
            burst_premium = burst_volume * price * 100
            is_burst = burst_premium >= (_W.burst_premium or LARGE_PREMIUM) * scale
        rows.append(
            WhaleRow(
                symbol=symbol, right=right, expiry=expiry, dte=(expiry - today).days, strike=strike,
                spot=spot, otm_pct=(strike - spot) / spot, volume=int(volume),
                open_interest=None if oi is None else int(oi), vol_oi=vol_oi, price=price, premium=premium,
                side=side, lean=lean(right, side),
                tier="WHALE" if premium >= WHALE_PREMIUM * scale else "Large",
                delta=delta, iv=_num(c.get("iv")),
                fresh=vol_oi is not None and vol_oi >= FRESH_VOL_OI,
                intent=classify_intent(volume, oi), burst_volume=burst_volume, burst_premium=burst_premium,
                is_burst=is_burst, noise=noise_reason(symbol, right, strike, spot, delta, price, today, fundamentals),
            )
        )
    return rows


def scan_whales(
    symbols: list[str], source: DataSource, today: date, expiries: int = 2, previous: dict | None = None,
    fundamentals: dict | None = None, snapshot: list | None = None,
) -> tuple[list[WhaleRow], list[str]]:
    """Nearest `expiries` expirations (0DTE first when it exists), both calls and puts. Returns (rows, errors)."""
    rows: list[WhaleRow] = []
    errors: list[str] = []
    for symbol in symbols:
        try:
            spot = source.spot(symbol)
            for right in ("C", "P"):
                target = today
                for _ in range(expiries):
                    chain = source.option_chain(symbol, expiry=target, right=right)
                    if chain.empty:
                        break
                    rows.extend(whales_from_chain(symbol, chain, spot, today, previous, fundamentals, snapshot))
                    target = chain["expiry"].iloc[0] + timedelta(days=1)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    return rows, errors

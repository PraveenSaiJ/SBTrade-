"""Record one at-the-money implied-vol reading per symbol per day, the base for IV rank.

Schwab does not give an IV history, so rank only becomes available after enough days have been recorded
(indicators.MIN_IV_HISTORY). Until then `iv_rank` returns None rather than a made-up number.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from tool.indicators import hv_rank, iv_rank


def atm_iv(chain: pd.DataFrame, spot: float) -> float | None:
    """Implied vol of the contract whose strike is nearest the stock price (None when unquoted)."""
    if chain is None or chain.empty or "iv" not in chain:
        return None
    usable = chain[pd.to_numeric(chain["iv"], errors="coerce") > 0]
    if usable.empty:
        return None
    nearest = usable.iloc[(usable["strike"] - spot).abs().argsort()].iloc[0]
    return float(nearest["iv"])


def record_atm_iv(source, store, symbols: list[str], today: date, expiry: date) -> int:
    """Save today's ATM IV (from the `expiry` call chain) for symbols not yet recorded today. Returns rows saved."""
    todo = [s for s in dict.fromkeys(symbols) if s not in store.iv_symbols_on(today)]
    readings: dict[str, float] = {}
    for symbol in todo:
        try:
            iv = atm_iv(source.option_chain(symbol, expiry=expiry, right="C"), source.spot(symbol))
        except Exception:
            continue
        if iv is not None:
            readings[symbol] = iv
    return store.save_iv(today, readings)


def make_rank_fn(store):
    """rank(symbol, closes) -> (rank, 'IV' | 'HV'): the real IV rank once 60 days of IV are stored, else the HV rank."""
    from tool.config import get_config

    cfg = get_config().puts

    def rank(symbol: str, closes: pd.Series) -> tuple[float | None, str]:
        history = store.iv_history(symbol)
        if history:
            ivr = iv_rank(history[-1], history)
            if ivr is not None:
                return ivr, "IV"
        return hv_rank(closes, cfg.hv_window, cfg.hv_lookback), "HV"

    return rank


def current_iv_rank(store, symbol: str, current_iv: float | None) -> float | None:
    return iv_rank(current_iv, store.iv_history(symbol))

"""Every threshold the tool uses, in one place. Defaults live here; `config/settings.yaml` (optional) overrides them:

    screener:
      min_volume_ratio: 1.5
      trend_filter: sma200      # sma200 | sma50_rising | off
    options:
      naked_delta_low: 0.15
    account:
      account_size: 100000

Modules read their constants from `get_config()` instead of hard-coding numbers. Unknown keys are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path

import yaml

SETTINGS_PATH = Path("config/settings.yaml")
TREND_FILTERS = ("sma200", "sma50_rising", "off")


@dataclass(frozen=True)
class ScreenerConfig:
    min_volume_ratio: float = 1.5  # breakout-day volume / average of the prior 20 sessions
    hold_days: int = 10  # trading days: earnings inside this window exclude the setup
    trend_filter: str = "sma200"
    rsi_max: float = 75.0  # RSI(14) above this is overextended
    overextension_atr: float = 2.0  # (spot - entry) above this many ATR(14) is overextended
    resistance_lookback: int = 60  # bars searched for the highest swing high
    swing_window: int = 2  # a swing high is the highest bar with this many lower bars on each side
    post_earnings_days: int = 10  # trading days after a report that count as "recently reported"
    earnings_flag_days: int = 30  # upcoming earnings inside this many calendar days are weighted as risk
    earnings_window_days: int = 60  # the upcoming report is looked up this far ahead
    stop_atr: float = 1.5  # fallback stop = entry - stop_atr x ATR(10) when no swing low can be found
    stop_buffer_atr: float = 0.5  # structural stop sits this many ATR(10) under the level (entry)...
    stop_max_atr: float = 2.0  # ...but never more than this many ATR(10) under the entry
    swing_low_lookback: int = 20  # bars searched for the latest confirmed swing low below the entry
    swing_low_fallback_bars: int = 10  # no fractal low: use the lowest low of this many bars
    # pullback to the 20/50-day average in an uptrend
    ma_touch_atr: float = 0.5
    ma_touch_bars: int = 3
    ma_min_pullback_pct: float = 0.03  # at least this far under the 20-day high
    # bull flag
    flag_pole_pct: float = 0.15
    flag_pole_bars: int = 15
    flag_min_bars: int = 3
    flag_max_bars: int = 10
    flag_max_range_pct: float = 0.08
    flag_max_retrace: float = 0.5
    downtrend_pct: float = 0.03  # 20-day change below -3% is a downtrend
    weak_breakout_volume: float = 1.0  # a close above the high on less than this volume ratio is a weak breakout
    extended_atr: float = 3.0  # more than this many ATR above the 20-day average is extended
    pullback_min_change: float = 0.02  # a pullback needs the 20-day change above this
    t1_atr: float = 1.5  # T1 = first pivot at least this many ATR above the entry (else entry + t1_fallback_atr)
    t1_fallback_atr: float = 2.0
    t2_atr: float = 0.75  # T2 = first pivot at least this many ATR above T1 (else T1 + t2_fallback_atr)
    t2_fallback_atr: float = 1.5
    # ranking score = w_volume*norm(volume) + w_rr*norm(R:R) + w_trend*trend + w_earnings*(1 - earnings_risk)
    w_volume: float = 0.35
    w_rr: float = 0.25
    w_trend: float = 0.20
    w_earnings: float = 0.20
    volume_norm_cap: float = 3.0  # volume ratio at (or above) which norm(volume) = 1
    rr_norm_cap: float = 3.0  # R:R at (or above) which norm(R:R) = 1


@dataclass(frozen=True)
class OptionsConfig:
    # sold options: delta bands and quote quality
    naked_delta_low: float = 0.15
    naked_delta_high: float = 0.25
    put_delta_low: float = 0.15
    put_delta_high: float = 0.25
    covered_delta_low: float = 0.20
    covered_delta_high: float = 0.30
    max_spread_pct: float = 0.10  # bid-ask spread as a share of the mid
    naked_breakout_near: float = 0.98  # price this close to the breakout level counts as "ready to break out"
    naked_high_risk_strike_above_breakout: float = 1.05
    put_suspect_annualized: float = 3.0  # above 300% annualized the quote is flagged as suspect (note only)
    put_low_return_note: float = 0.15  # below this annualized return a note is added; nothing is filtered
    # bought options
    swing_delta_low: float = 0.40
    swing_delta_high: float = 0.60
    lotto_delta_low: float = 0.05
    lotto_delta_high: float = 0.25
    lotto_min_price_cap: float = 1.50  # dollars per share; pricier stocks use lotto_price_cap_pct of the stock price
    lotto_price_cap_pct: float = 0.004
    odte_delta_low: float = 0.10
    odte_delta_high: float = 0.45
    odte_price_cap_pct: float = 0.008
    # delta buckets on the dashboard
    bucket_high_max: float = 0.20
    bucket_moderate_max: float = 0.35
    # covered-call monitor
    near_strike: float = 0.01  # close/roll when the stock is within this share of the strike
    min_extrinsic: float = 0.50  # early-assignment risk below this much time value
    mag7: tuple = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA")
    index_etfs: tuple = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class CoveredCallConfig:
    tiers: dict = field(default_factory=lambda: {
        "safe": ["NVDA", "AMZN", "MSFT"],
        "proven": ["SNOW", "ZS", "SHOP"],
        "aggressive": ["CRWV", "IREN"],
    })
    # net return per week the income row aims for, as a share of the stock price (edit per tier)
    tier_targets: dict = field(default_factory=lambda: {
        "safe": 0.015, "proven": 0.015, "aggressive": 0.015, "other": 0.015,
    })
    uncapped: tuple = ("CRWV", "SNOW:rh_ira")  # "SYMBOL" or "SYMBOL:account": held without covered calls
    fee_per_contract: float = 0.0  # opening-leg fee per contract; net return = (bid - fee/100) / stock price
    quarters_per_year: int = 4  # Schwab's dividendAmount is annual; a quarterly payer is assumed
    roll_weeks: int = 3  # roll search: this expiry plus this many weeks out
    events: tuple = ()  # manual list of {symbol, date, label}: investor days, product events


@dataclass(frozen=True)
class PutConfig:
    """Cash-secured puts at support (the Support row next to the weekly-S1 guardrail row)."""

    support_delta_low: float = 0.20
    support_delta_high: float = 0.30
    min_dte: int = 7
    max_dte: int = 45
    max_expiries: int = 4  # chain requests per symbol: the first two Fridays, the monthly, the last in the window
    earnings_rank_min: float = 60.0  # earnings expiries show only with the switch on and a rank above this
    hv_window: int = 20  # realized-volatility window for the HV rank stand-in
    hv_lookback: int = 252  # the HV rank compares today's HV with this many trading days
    support_swing_lookback: int = 60
    cash: dict = field(default_factory=dict)  # {account: dollars} available for cash-secured puts; empty = unset


@dataclass(frozen=True)
class CallConfig:
    """Directional and LEAPS calls to buy."""

    directional_delta_low: float = 0.60
    directional_delta_high: float = 0.70
    leaps_delta_low: float = 0.70
    leaps_delta_high: float = 0.80
    directional_min_dte: int = 14
    directional_max_dte: int = 60
    directional_max_expiries: int = 3  # chain requests per symbol: the first Friday, the monthly, the last in the window
    leaps_min_dte: int = 270
    leaps_window_days: int = 200  # how far past the 270-day mark to look for the first listed expiry
    leaps_strike_count: int = 40  # in-the-money 0.70-0.80 delta calls sit far from the money
    iv_crush_rank: float = 50.0  # earnings before expiry and a rank above this = IV crush risk
    leaps_earnings_days: int = 14  # a LEAPS is only flagged for earnings this close (it spans several reports anyway)
    max_directional: int = 40  # candidates per run (best score first) to bound chain requests
    max_leaps: int = 60
    earnings_days: int = 7  # names reporting within this many days get an earnings call idea
    earnings_delta_low: float = 0.45
    earnings_delta_high: float = 0.55
    max_earnings: int = 40
    earnings_max_ratio: float = 1.0  # the tab shows only calls whose break-even / expected move is at or under this


@dataclass(frozen=True)
class LevelConfig:
    """Support and resistance zones (sr_levels.py)."""

    fractal_window: int = 3  # a swing point beats this many bars on each side
    fractal_lookback: int = 120
    merge_atr: float = 0.5  # levels within this many ATR(10) of a zone's mean merge into it
    zone_pad_atr: float = 0.25  # each zone is padded by this many ATR on both sides
    touch_lookback: int = 120  # bars searched for touches of a zone
    profile_days: int = 60  # volume profile window
    profile_bins: int = 40
    hvn_min_ratio: float = 1.3  # a bin is a volume node at this multiple of the average bin volume
    hvn_max_nodes: int = 3
    oi_top_n: int = 3  # open-interest strikes kept per side
    oi_band: float = 0.15  # only strikes within this share of the price
    min_zone_score: float = 0.30  # a zone must score at least this to cap a target or set a put strike
    score_touches: float = 0.5
    score_volume: float = 0.5


@dataclass(frozen=True)
class WhaleConfig:
    whale_premium: float = 1_000_000.0  # tier "WHALE"
    large_premium: float = 250_000.0  # tier "Large"
    min_volume: int = 300
    fresh_vol_oi: float = 1.0  # more traded today than the whole open interest: new positions
    index_scale: float = 10.0  # SPY/QQQ/IWM thresholds are this many times higher
    opening_vol_oi: float = 1.0  # volume above open interest: at least that many new contracts (opening)
    likely_opening_vol_oi: float = 0.5  # volume at least this share of open interest: likely opening
    burst_premium: float | None = None  # premium added since the last scan that makes a BURST (None = large_premium)
    burst_max_gap_minutes: int = 45  # a previous scan older than this is not comparable
    deep_itm_delta: float = 0.90  # |delta| at or above this is stock replacement or a hedge, not a view: noise
    dividend_capture_days: int = 5  # an ITM call with thin time value this close to the ex-date is dividend capture
    flow_ratio: float = 1.5  # net flow is bullish/bearish when one side is this many times the other
    extra_names: tuple = (
        "AMD", "AVGO", "PLTR", "MU", "COIN", "SMCI", "NFLX", "CRWD", "MSTR", "HOOD", "SOFI", "ARM", "TSM", "SNOW",
    )


@dataclass(frozen=True)
class FreshnessConfig:
    market_open: str = "09:30"
    market_close: str = "16:00"
    morning_grace_end: str = "10:30"  # first scheduled runs may still be going until then
    intraday_max_minutes: int = 30
    hourly_max_hours: int = 2


@dataclass(frozen=True)
class LongTermConfig:
    min_quality_passes: int = 4
    extended_over_sma50: float = 0.10
    max_stop_distance: float = 0.12
    hold_months: int = 12
    min_bars: int = 200
    # holdings review (holding_review.py)
    max_weight: float = 0.15  # a position above this share of the stock portfolio gets a concentration warning
    corr_window: int = 120  # trading days of returns used to find overlapping holdings
    corr_threshold: float = 0.80  # pairs at or above this are listed as overlapping
    corr_store_min: float = 0.50  # pairs at or above this are saved (so the threshold can change without a rerun)
    conviction_weights: dict = field(default_factory=lambda: {
        "quality": 0.30, "trend": 0.20, "growth": 0.15, "valuation": 0.15, "fcf": 0.10, "resilience": 0.10,
    })
    growth_cap: float = 0.30  # revenue / EPS growth at or above this scores the full growth mark
    fcf_cap: float = 0.25  # FCF margin at or above this scores the full mark
    pe_cheap: float = 0.80  # forward P/E at or under this share of its 5-year average scores full valuation marks
    pe_rich: float = 1.50  # at or above this share it scores zero
    drawdown_cap: float = 0.50  # a 50% fall from the 52-week high scores zero resilience
    sma200_cap: float = 0.25  # 25% above the 200-day average scores the full mark
    min_data_share: float = 0.50  # a conviction score needs at least this share of the weights to have data


@dataclass(frozen=True)
class PerformanceConfig:
    horizon_days: int = 10  # trading days: the 1-2 week swing hold
    min_sample: int = 20  # decided ideas needed before the statistics are called meaningful


@dataclass(frozen=True)
class AccountConfig:
    account_size: float | None = None  # dollars; position sizing is skipped until this is set
    risk_pct: float = 0.01  # share of the account risked per trade


@dataclass(frozen=True)
class JobsConfig:
    history_days: int = 90  # job-run history older than this is pruned
    notify: bool = True  # desktop notification on job failure or Schwab sign-in needed


@dataclass(frozen=True)
class DataConfig:
    rate_limit_per_min: int = 120  # Schwab market-data limit
    quote_batch: int = 100  # symbols per quotes request
    spot_cache_seconds: int = 15
    cache_dir: str = "data/cache"
    token_days: int = 7  # Schwab refresh-token lifetime
    options_workers: int = 4  # option families built at the same time (the 120/min limiter is shared by all of them)


@dataclass(frozen=True)
class Config:
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    whales: WhaleConfig = field(default_factory=WhaleConfig)
    covered_calls: CoveredCallConfig = field(default_factory=CoveredCallConfig)
    puts: PutConfig = field(default_factory=PutConfig)
    calls: CallConfig = field(default_factory=CallConfig)
    levels: LevelConfig = field(default_factory=LevelConfig)
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)
    longterm: LongTermConfig = field(default_factory=LongTermConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    data: DataConfig = field(default_factory=DataConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)


def _override(section: object, raw: dict | None) -> object:
    """Replace the fields of a section dataclass that appear in `raw`; unknown keys are ignored."""
    changes = {}
    known = {f.name: f for f in fields(section)}
    for key, value in (raw or {}).items():
        if key not in known:
            continue
        if isinstance(getattr(section, key), tuple) and isinstance(value, list):
            value = tuple(value)
        changes[key] = value
    return replace(section, **changes)


def load_all(path: Path = SETTINGS_PATH) -> Config:
    """Defaults, overridden by the matching sections of settings.yaml."""
    cfg = Config()
    if not Path(path).exists():
        return cfg
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    screener_raw = dict(raw.get("screener") or {})
    if screener_raw.get("trend_filter") is False:
        screener_raw["trend_filter"] = "off"  # YAML 1.1 reads an unquoted `off` as the boolean False
    cfg = replace(
        cfg,
        screener=_override(cfg.screener, screener_raw),
        **{name: _override(getattr(cfg, name), raw.get(name)) for name in (
            "options", "whales", "covered_calls", "puts", "calls", "levels", "freshness", "longterm", "performance", "account", "data", "jobs")},
    )
    if cfg.screener.trend_filter not in TREND_FILTERS:
        raise ValueError(f"trend_filter must be one of {TREND_FILTERS}, got {cfg.screener.trend_filter!r}")
    return cfg


def load_config(path: Path = SETTINGS_PATH) -> ScreenerConfig:
    """The screener section only (kept for existing callers)."""
    return load_all(path).screener


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The process-wide configuration, loaded once from config/settings.yaml."""
    return load_all()

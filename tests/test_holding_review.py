from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from tool.config import LongTermConfig, load_all
from tool.data.earnings import FinnhubEvents, parse_metrics
from tool.holding_review import (
    HoldingReview, conviction_score, correlated_pairs, is_fund, keep_suggestion, returns_frame, review_holdings,
)
from tool.holdings import Holdings, Lot
from tool.store import Store

CFG = LongTermConfig()


# ---------------------------------------------------------------- conviction score
def test_full_marks_and_zero_marks():
    top = conviction_score(6, 3, 0.30, 0.30, 0.80, 0.25, 0.0, 0.25, CFG)
    assert top == pytest.approx(100.0)
    bottom = conviction_score(0, 0, -0.10, -0.20, 1.50, 0.0, -0.50, -0.10, CFG)
    assert bottom == pytest.approx(0.0)


def test_a_missing_part_is_dropped_and_the_rest_reweighted():
    # only quality 3/6 (0.5 x 30%) and trend 3/3 (1.0 x 20%) are known: 0.35 of 0.50 available weight = 70
    assert conviction_score(3, 3, None, None, None, None, None, None, CFG) == pytest.approx(70.0)


def test_too_little_data_gives_no_score_instead_of_a_made_up_one():
    # a fund: only trend (20%) and resilience (10%) exist, 30% of the weight, under the 50% needed
    assert conviction_score(None, 3, None, None, None, None, -0.1, 0.1, CFG) is None
    assert conviction_score(None, None, None, None, None, None, None, None, CFG) is None


def test_valuation_part_is_cheap_to_rich_between_the_two_ratios():
    loose = LongTermConfig(min_data_share=0.2)  # only valuation (15%) and FCF (10%) have data here
    cheap = conviction_score(None, None, None, None, 0.80, 0.25, None, None, loose)  # valuation 1.0, fcf 1.0
    mid = conviction_score(None, None, None, None, 1.15, 0.25, None, None, loose)  # valuation 0.5, fcf 1.0
    assert cheap == pytest.approx(100.0) and mid == pytest.approx(100 * (0.15 * 0.5 + 0.10) / 0.25)


def test_weights_are_configurable(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("longterm:\n  max_weight: 0.10\n  corr_threshold: 0.9\n  conviction_weights: {quality: 1.0}\n", encoding="utf-8")
    cfg = load_all(path).longterm
    assert cfg.max_weight == 0.10 and cfg.corr_threshold == 0.9 and cfg.conviction_weights == {"quality": 1.0}
    assert LongTermConfig().max_weight == 0.15 and LongTermConfig().corr_window == 120


# ---------------------------------------------------------------- review per holding
GOOD = {"revChangeTTM": 20.0, "epsChangePercentTTM": 25.0, "operatingMarginTTM": 30.0, "returnOnEquity": 25.0,
        "totalDebtToEquity": 50.0, "peRatio": 25.0, "epsTTM": 5.0, "high52": 120.0}


def flat_bars(price=100.0, n=260):
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6}, index=idx)


def holdings() -> Holdings:
    return Holdings(lots=[
        Lot(symbol="AAA", account="rh_taxable", shares=100, cost=50.0),
        Lot(symbol="AAA", account="rh_ira", shares=100, cost=70.0),  # same stock in a second account
        Lot(symbol="BBB", account="rh_taxable", shares=100, cost=80.0),
        Lot(symbol="FND", account="rh_taxable", shares=100, cost=10.0),
    ])


def test_fund_detection():
    assert is_fund({}) and is_fund(None) and is_fund({"revChangeTTM": 0, "epsTTM": 0, "peRatio": 0, "high52": 65.0})
    assert not is_fund(GOOD)


def test_review_sums_accounts_computes_weights_and_the_price_columns():
    prices = {"AAA": 110.0, "BBB": 110.0, "FND": 50.0}
    fundamentals = {"AAA": GOOD, "BBB": GOOD, "FND": {"high52": 65.0}}
    metrics = {"AAA": {"forward_pe": 20.0, "pe_5y": 25.0, "fcf_margin": 0.30}}
    bars = {s: flat_bars() for s in prices}
    reviews = {r.symbol: r for r in review_holdings(holdings(), prices, fundamentals, metrics, bars, CFG)}
    a = reviews["AAA"]
    assert a.shares == 200 and a.value == 22_000.0 and a.gain_pct == pytest.approx(110 / 60 - 1)  # average cost 60
    assert sum(r.weight for r in reviews.values()) == pytest.approx(1.0)
    assert a.weight == pytest.approx(22_000 / (22_000 + 11_000 + 5_000))
    assert a.concentrated and reviews["BBB"].concentrated  # 58% and 29%: both over the 15% default
    assert not reviews["FND"].concentrated  # 13%
    assert a.drawdown == pytest.approx(110 / 120 - 1) and a.dist_sma200 == pytest.approx(0.10)
    assert a.pe_vs_avg == pytest.approx(0.8) and a.fcf_margin == 0.30 and a.quality == 6 and a.trend == 2
    assert a.rev_growth == pytest.approx(0.20) and a.eps_growth == pytest.approx(0.25) and a.conviction is not None


def test_funds_keep_price_columns_but_no_fundamentals_or_conviction():
    prices = {"AAA": 110.0, "BBB": 110.0, "FND": 50.0}
    reviews = {r.symbol: r for r in review_holdings(
        holdings(), prices, {"AAA": GOOD, "BBB": GOOD, "FND": {"high52": 65.0}}, {}, {s: flat_bars() for s in prices}, CFG)}
    fnd = reviews["FND"]
    assert fnd.fund and fnd.conviction is None and fnd.quality is None and fnd.rev_growth is None
    assert fnd.drawdown == pytest.approx(50 / 65 - 1) and fnd.weight is not None  # price-based columns stay


def test_concentration_flag_follows_the_configured_maximum():
    prices = {"AAA": 110.0, "BBB": 110.0, "FND": 50.0}
    args = (holdings(), prices, {"AAA": GOOD, "BBB": GOOD, "FND": {}}, {}, {s: flat_bars() for s in prices})
    loose = {r.symbol: r.concentrated for r in review_holdings(*args, cfg=LongTermConfig(max_weight=0.60))}
    assert not any(loose.values())
    tight = {r.symbol: r.concentrated for r in review_holdings(*args, cfg=LongTermConfig(max_weight=0.20))}
    assert tight == {"AAA": True, "BBB": True, "FND": False}


def test_finnhub_growth_is_the_fallback_when_schwab_has_none():
    f = {**GOOD, "revChangeTTM": None, "epsChangePercentTTM": None}
    [a] = [r for r in review_holdings(holdings(), {"AAA": 110.0}, {"AAA": f}, {"AAA": {"rev_growth_ttm": 12.0, "eps_growth_ttm": 8.0}},
                                      {"AAA": flat_bars()}, CFG) if r.symbol == "AAA"]
    assert a.rev_growth == pytest.approx(0.12) and a.eps_growth == pytest.approx(0.08)


# ---------------------------------------------------------------- correlation
def bars_from_returns(returns: np.ndarray) -> pd.DataFrame:
    close = 100 * np.cumprod(1 + returns)
    idx = pd.bdate_range(end="2026-10-02", periods=len(close))
    return pd.DataFrame({"close": close}, index=idx)


def test_overlapping_pairs_are_found_above_the_threshold_only():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 0.02, 200)
    bars = {
        "X": bars_from_returns(x), "Y": bars_from_returns(x + rng.normal(0, 0.002, 200)),  # moves with X
        "Z": bars_from_returns(rng.normal(0, 0.02, 200)),  # independent
        "W": bars_from_returns(-x),  # inverse: strongly negative, not an overlap
        "SHORT": bars_from_returns(x[:50]),  # too little history
    }
    frame = returns_frame(list(bars), bars, 120)
    assert set(frame.columns) == {"X", "Y", "Z", "W"}  # SHORT is left out
    pairs = correlated_pairs(frame, 0.8)
    assert [(r.a, r.b) for r in pairs.itertuples()] == [("X", "Y")] and pairs.iloc[0]["corr"] > 0.95
    assert len(correlated_pairs(frame, 0.0)) >= 1 and correlated_pairs(frame[["X"]], 0.8).empty


def test_keep_suggestion_names_the_higher_conviction_holding():
    def rv(sym, conviction):
        return HoldingReview(
            symbol=sym, shares=1, price=1, value=1, weight=0.1, gain_pct=None, rev_growth=None, eps_growth=None,
            forward_pe=None, pe_5y=None, pe_vs_avg=None, fcf_margin=None, drawdown=None, dist_sma200=None, quality=None,
            trend=None, conviction=conviction, fund=False, concentrated=False)

    reviews = {"A": rv("A", 80), "B": rv("B", 50), "C": rv("C", 52), "D": rv("D", None)}
    assert keep_suggestion("A", "B", reviews).startswith("keep A") and "trimming B" in keep_suggestion("B", "A", reviews)
    assert "similar conviction" in keep_suggestion("B", "C", reviews)
    assert "no conviction score" in keep_suggestion("A", "D", reviews)


# ---------------------------------------------------------------- Finnhub metrics and the daily cache
PAYLOAD = {
    "metric": {"forwardPE": 24.19, "revenueGrowthTTMYoy": 17.79, "epsGrowthTTMYoy": 31.56, "revenueGrowth5Y": 14.57, "epsGrowth5Y": 17.38},
    "series": {"annual": {
        "pe": [{"period": "2026", "v": 20.0}, {"period": "2025", "v": 30.0}, {"period": "2024", "v": 40.0},
               {"period": "2023", "v": -5.0}, {"period": "2022", "v": 30.0}, {"period": "2021", "v": 99.0}],
        "fcfMargin": [{"period": "2026", "v": 0.2019}, {"period": "2025", "v": 0.25}],
    }},
}


def test_parse_metrics_takes_the_five_year_pe_from_the_last_five_positive_annual_values():
    m = parse_metrics(PAYLOAD)
    assert m["forward_pe"] == 24.19 and m["fcf_margin"] == 0.2019
    assert m["pe_5y"] == pytest.approx((20 + 30 + 40 + 30) / 4)  # the -5 is dropped, the sixth year is not used
    assert m["rev_growth_ttm"] == 17.79 and m["eps_growth_5y"] == 17.38


def test_parse_metrics_leaves_missing_things_out_for_funds():
    assert parse_metrics({"metric": {"52WeekHigh": 60.0}, "series": {}}) == {}
    assert "pe_5y" not in parse_metrics({"metric": {}, "series": {"annual": {"pe": [{"v": 20.0}, {"v": 30.0}]}}})  # under 3 years


def client(requests: list, fail: set[str] = frozenset()) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        requests.append(symbol)
        if symbol in fail:
            return httpx.Response(500)
        return httpx.Response(200, json=PAYLOAD)

    return httpx.Client(base_url="https://finnhub.io/api/v1", transport=httpx.MockTransport(handler))


def test_metrics_are_cached_for_the_day_and_topped_up(tmp_path):
    asked: list = []
    api = FinnhubEvents(api_key="k", client=client(asked))
    path = tmp_path / "m.json"
    today = date(2026, 10, 23)
    first = api.cached_metrics(["AAA", "BBB"], today, path=path, pause=0)
    assert asked == ["AAA", "BBB"] and first["AAA"]["forward_pe"] == 24.19
    api.cached_metrics(["AAA", "BBB"], today, path=path, pause=0)
    assert asked == ["AAA", "BBB"]  # a second call the same day makes no requests
    api.cached_metrics(["AAA", "CCC"], today, path=path, pause=0)
    assert asked == ["AAA", "BBB", "CCC"]  # only the new symbol is fetched
    api.cached_metrics(["AAA"], date(2026, 10, 24), path=path, pause=0)
    assert asked[-1] == "AAA" and len(asked) == 4  # a new day refetches


def test_a_failing_symbol_is_skipped_and_the_error_never_carries_the_key(tmp_path):
    asked: list = []
    api = FinnhubEvents(api_key="secret-key", client=client(asked, fail={"BAD"}))
    out = api.cached_metrics(["BAD", "AAA"], date(2026, 10, 23), path=tmp_path / "m.json", pause=0)
    assert "BAD" not in out and "AAA" in out
    with pytest.raises(Exception) as err:
        api.company_metrics("BAD")
    assert "secret-key" not in str(err.value) and "token" not in str(err.value)


# ---------------------------------------------------------------- storage
def test_the_saved_review_is_replaced_each_run(tmp_path):
    prices = {"AAA": 110.0, "BBB": 110.0, "FND": 50.0}
    reviews = review_holdings(holdings(), prices, {"AAA": GOOD, "BBB": GOOD, "FND": {}}, {}, {s: flat_bars() for s in prices}, CFG)
    pairs = pd.DataFrame([{"a": "AAA", "b": "BBB", "corr": 0.91}])
    store = Store(tmp_path / "t.sqlite")
    assert store.save_holding_review(reviews, pairs) == 3
    store.save_holding_review(reviews[:2], pd.DataFrame(columns=["a", "b", "corr"]))
    saved = store.latest_holding_review()
    assert list(saved["symbol"]) == [r.symbol for r in reviews[:2]] and store.latest_holding_pairs().empty
    row = saved.iloc[0]
    assert row["weight"] == pytest.approx(reviews[0].weight) and row["concentrated"] == 1 and row["fund"] == 0
    store.close()

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tool.holdings import Holdings, LongOption, Lot, ShortOption, option_account
from tool.portfolio_risk import (
    build_position_rows, collateral_by_account, greeks_by_symbol, held_events, positions_frame, sector_exposure, totals, week_risk,
)

TODAY = date(2026, 10, 23)
FRI = date(2026, 10, 30)
SPY = 500.0


class Source:
    def __init__(self, prices, chains):
        self.prices, self.chains = prices, chains

    def spot(self, symbol):
        if symbol not in self.prices:
            raise ValueError("no quote")
        return self.prices[symbol]

    def option_chain(self, symbol, expiry=None, right="C", **kw):
        return self.chains.get((symbol, right), pd.DataFrame())


def chain(strike, expiry, delta, theta, bid, ask):
    return pd.DataFrame([{"strike": strike, "expiry": expiry, "bid": bid, "ask": ask, "delta": delta, "theta": theta}])


def holdings(**kw):
    base = dict(lots=[Lot(symbol="AAA", account="rh_taxable", shares=300, cost=90.0)], short_options=[], long_options=[])
    base.update(kw)
    return Holdings(**base)


def test_shares_have_delta_one_and_beta_weighting_uses_beta_over_spy():
    rows = build_position_rows(holdings(), Source({"AAA": 100.0}, {}), {"AAA": {"beta": 2.0}}, SPY, TODAY)
    r = rows[0]
    assert (r.kind, r.delta_shares, r.beta, r.beta_assumed) == ("shares", 300, 2.0, False)
    assert r.spy_equiv_shares == pytest.approx(300 * 100 * 2.0 / SPY)  # 120 SPY shares


def test_missing_beta_counts_as_one_and_is_marked():
    r = build_position_rows(holdings(), Source({"AAA": 100.0}, {}), {}, SPY, TODAY)[0]
    assert (r.beta, r.beta_assumed) == (1.0, True)


@pytest.mark.parametrize("side,kind,delta,theta,sign", [
    ("short", "call", 0.30, -0.05, -1),   # short call: negative delta, positive theta
    ("long", "call", 0.30, -0.05, 1),     # long call: positive delta, negative theta
    ("short", "put", -0.25, -0.04, -1),   # short put: positive delta, positive theta
    ("long", "put", -0.25, -0.04, 1),     # long put: negative delta, negative theta
])
def test_option_delta_and_theta_signs_and_quantity(side, kind, delta, theta, sign):
    opt = (ShortOption if side == "short" else LongOption)("AAA", kind, 105.0, FRI, 2)
    h = holdings(**{f"{side}_options": [opt]}, lots=[])
    src = Source({"AAA": 100.0}, {("AAA", "C" if kind == "call" else "P"): chain(105.0, FRI, delta, theta, 1.0, 1.2)})
    r = build_position_rows(h, src, {}, SPY, TODAY)[0]
    assert r.status == "live" and r.kind == f"{side} {kind}"
    assert r.delta_shares == pytest.approx(delta * 100 * 2 * sign)
    assert r.theta_day == pytest.approx(theta * 100 * 2 * sign)
    assert r.option_mid == pytest.approx(1.1) and r.value == pytest.approx(1.1 * 100 * 2 * sign)


def test_expired_and_unquoted_options_are_not_live():
    h = holdings(lots=[], short_options=[ShortOption("AAA", "call", 105.0, date(2026, 10, 16), 1)],
                 long_options=[LongOption("AAA", "call", 110.0, FRI, 1)])
    rows = build_position_rows(h, Source({"AAA": 100.0}, {}), {}, SPY, TODAY)
    assert [r.status for r in rows] == ["no quote", "expired"]  # long first, then short
    assert totals(positions_frame(rows), SPY)["no_quote"] == 1
    assert all(r.delta_shares in (None, 0.0) for r in rows)


def test_options_without_an_account_are_assumed_taxable():
    assert option_account(ShortOption("AAA", "call", 1.0, FRI, 1)) == "rh_taxable"
    assert option_account(ShortOption("AAA", "call", 1.0, FRI, 1, account="rh_ira")) == "rh_ira"


def test_totals_and_greeks_by_symbol_sum_live_rows():
    h = holdings(short_options=[ShortOption("AAA", "call", 105.0, FRI, 1, account="rh_taxable")])
    src = Source({"AAA": 100.0}, {("AAA", "C"): chain(105.0, FRI, 0.30, -0.05, 1.0, 1.2)})
    pos = positions_frame(build_position_rows(h, src, {"AAA": {"beta": 1.0}}, SPY, TODAY))
    t = totals(pos, SPY)
    assert t["delta_dollars"] == pytest.approx((300 - 30) * 100.0)
    assert t["theta_day"] == pytest.approx(5.0)
    assert t["spy_equiv_shares"] == pytest.approx(270 * 100 / SPY) and t["stock_value"] == pytest.approx(30000)
    g = greeks_by_symbol(pos)
    assert list(g["symbol"]) == ["AAA"] and g["delta_shares"].iloc[0] == pytest.approx(270)


def test_sector_and_theme_exposure_sum_to_one():
    pos = pd.DataFrame([
        dict(account="a", symbol="AAA", kind="shares", value=600.0, status="live"),
        dict(account="b", symbol="AAA", kind="shares", value=200.0, status="live"),
        dict(account="a", symbol="BBB", kind="shares", value=200.0, status="live"),
        dict(account="a", symbol="CCC", kind="shares", value=999.0, status="no price"),
    ])
    info = pd.DataFrame([{"symbol": "AAA", "industry": "Tech"}, {"symbol": "BBB", "industry": "Banks"}])
    industry, themes = sector_exposure(pos, info)
    assert dict(zip(industry["group"], industry["weight"])) == pytest.approx({"Tech": 0.8, "Banks": 0.2})
    assert themes["weight"].sum() == pytest.approx(1.0)
    unknown, _ = sector_exposure(pos, pd.DataFrame(columns=["symbol", "industry"]))
    assert list(unknown["group"]) == ["Unknown"]


def week_frame(rows):
    base = dict(account="rh_taxable", qty=1.0, strike=100.0, expiry=FRI.isoformat(), price=95.0, option_mid=1.0, status="live")
    return pd.DataFrame([{**base, **r} for r in rows])


def test_itm_short_call_with_thin_time_value_is_an_assignment_risk():
    pos = week_frame([dict(symbol="AAA", kind="short call", price=104.0, option_mid=4.2)])  # intrinsic 4.0 -> time value 0.20
    out = week_risk(pos, pd.DataFrame(columns=["symbol"]), TODAY)
    assert out.iloc[0]["itm"] == "ITM" and "ASSIGNMENT RISK" in out.iloc[0]["flags"]
    healthy = week_risk(week_frame([dict(symbol="AAA", kind="short call", price=104.0, option_mid=6.0)]),
                        pd.DataFrame(columns=["symbol"]), TODAY)
    assert "ASSIGNMENT RISK" not in healthy.iloc[0]["flags"]


def test_ex_dividend_before_expiry_flags_a_call_whose_time_value_is_below_the_dividend():
    pos = week_frame([dict(symbol="AAA", kind="short call", price=104.0, option_mid=4.6)])  # time value 0.60
    info = pd.DataFrame([{"symbol": "AAA", "ex_div": "2026-10-27", "dividend": 1.0, "next_earnings": "2026-10-28"}])
    flags = week_risk(pos, info, TODAY).iloc[0]["flags"]
    assert "ex-dividend 2026-10-27" in flags and "earnings 2026-10-28 before expiry" in flags


def test_short_call_without_shares_is_naked_and_short_put_shows_assignment_cash():
    pos = pd.concat([
        week_frame([dict(symbol="AAA", kind="short call"), dict(symbol="BBB", kind="short put", qty=2.0, strike=50.0)]),
        week_frame([dict(symbol="AAA", kind="shares", qty=50.0, strike=None, expiry=None)]),
    ], ignore_index=True)
    out = week_risk(pos, pd.DataFrame(columns=["symbol"]), TODAY).set_index("symbol")
    assert "NAKED: 50 shares" in out.loc["AAA", "flags"]
    assert "$10,000 cash needed" in out.loc["BBB", "flags"]


def test_expired_short_is_flagged_not_listed_as_live_risk():
    pos = week_frame([dict(symbol="AAA", kind="short call", expiry="2026-10-16", status="expired")])
    out = week_risk(pos, pd.DataFrame(columns=["symbol"]), TODAY)
    assert "expired 2026-10-16" in out.iloc[0]["flags"] and "itm" not in out.columns


def test_held_events_only_for_held_names_inside_the_window():
    pos = pd.DataFrame([dict(account="a", symbol="AAA", kind="shares", qty=1.0, status="live", expiry=None)])
    info = pd.DataFrame([
        {"symbol": "AAA", "next_earnings": "2026-10-27", "ex_div": "2026-12-01"},
        {"symbol": "ZZZ", "next_earnings": "2026-10-27", "ex_div": None},
    ])
    out = held_events(pos, info, TODAY)
    assert list(zip(out["symbol"], out["event"])) == [("AAA", "earnings")]


def test_collateral_per_account_under_over_and_unset():
    pos = pd.DataFrame([
        dict(account="rh_taxable", kind="short put", strike=50.0, qty=2.0, status="live"),   # 10,000
        dict(account="rh_ira", kind="short put", strike=100.0, qty=1.0, status="live"),      # 10,000
        dict(account="rh_ira", kind="short put", strike=100.0, qty=1.0, status="expired"),   # ignored
        dict(account="ibkr", kind="short put", strike=10.0, qty=1.0, status="live"),         # 1,000, no cash set
    ])
    out = collateral_by_account(pos, {"rh_taxable": 25_000.0, "rh_ira": 5_000.0}).set_index("account")
    assert out.loc["rh_taxable", ["committed", "free", "state"]].tolist() == [10_000.0, 15_000.0, "ok"]
    assert out.loc["rh_ira", "state"] == "over" and out.loc["rh_ira", "committed"] == 10_000.0
    assert out.loc["ibkr", "state"] == "unset" and out.loc["ibkr", "committed"] == 1_000.0

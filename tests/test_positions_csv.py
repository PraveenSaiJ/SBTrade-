from __future__ import annotations

from datetime import date

import pytest

from tool.holdings import Holdings, LongOption, Lot, ShortOption, load_holdings as load_holdings_yaml, save_holdings
from tool.positions_csv import apply_positions, diff_account, parse_option_symbol, parse_positions
from tool.store import Store


@pytest.mark.parametrize("text", [
    "MSFT  261120C00500000",
    "-MSFT261120C500",
    "MSFT 11/20/2026 500.00 C",
    "MSFT 20NOV26 500 C",
])
def test_every_option_symbol_style_parses_to_the_same_contract(text):
    assert parse_option_symbol(text) == ("MSFT", "call", 500.0, date(2026, 11, 20))


def test_puts_and_non_options():
    assert parse_option_symbol("AAPL  261120P00150500") == ("AAPL", "put", 150.5, date(2026, 11, 20))
    assert parse_option_symbol("MSFT") is None


FIDELITY = """Account Number,Symbol,Description,Quantity,Last Price,Cost Basis Total,Average Cost Basis
Z123,MSFT,MICROSOFT CORP,100,517.00,"$40,000.00",$400.00
Z123,-MSFT261120C550,CALL MSFT,-1,3.10,$250.00,
Z123,AMZN261120C00250000,CALL AMZN,2,5.00,$900.00,$4.50
Z123,SPAXX**,HELD IN MONEY MARKET,5000,1,5000,
Z123,FOO,NO BASIS,10,5.00,,
Z123,Pending Activity,,,,,
Z123,Account Total,,,,,
"""


def test_header_aliases_options_short_quantities_and_skipped_rows():
    p = parse_positions(FIDELITY)
    assert [(l.symbol, l.shares, l.cost) for l in p.stocks] == [("MSFT", 100, 400.0)]
    assert [(o.symbol, o.type, o.strike, o.qty, o.net_credit_total) for o in p.shorts] == [("MSFT", "call", 550.0, 1, 250.0)]
    assert [(o.symbol, o.qty, o.avg_price) for o in p.longs] == [("AMZN", 2, 4.5)]
    reasons = " | ".join(why for _, why in p.skipped)
    assert "no cost basis" in reasons and "money-market" in reasons
    assert p.accounts_in_file == ["Z123"]


def test_cost_per_share_comes_from_the_total_when_there_is_no_average():
    text = "Symbol,Quantity,Cost Basis\nNVDA,50,\"$5,000.00\"\n"
    assert parse_positions(text).stocks[0].cost == 100.0


def test_multi_account_file_can_be_filtered_and_a_file_without_header_is_rejected():
    text = "Account,Symbol,Quantity,Average Cost\nA,AAA,10,5\nB,BBB,20,6\n"
    assert [l.symbol for l in parse_positions(text, "B").stocks] == ["BBB"]
    assert parse_positions(text).accounts_in_file == ["A", "B"]
    with pytest.raises(ValueError):
        parse_positions("just,some,words\n1,2,3\n")


def stored() -> Holdings:
    return Holdings(
        lots=[Lot(symbol="MSFT", account="rh_taxable", shares=100, cost=400.0, strategy="core", note="keep"),
              Lot(symbol="OLD", account="rh_taxable", shares=5, cost=10.0),
              Lot(symbol="AAA", account="rh_ira", shares=7, cost=3.0)],
        short_options=[ShortOption("MSFT", "call", 540.0, date(2026, 11, 13), 1, account="rh_taxable"),
                       ShortOption("AAA", "call", 9.0, date(2026, 11, 13), 1, account="rh_ira")],
        long_options=[],
    )


def test_diff_marks_add_update_remove_keep_for_the_chosen_account_only():
    p = parse_positions("Symbol,Quantity,Average Cost\nMSFT,100,400\nNEW,10,20\n")
    result = {c.key: c.result for c in diff_account(stored(), "rh_taxable", p) if c.kind == "stock"}
    assert result == {"MSFT": "keep", "NEW": "add", "OLD": "remove"}  # AAA is in another account, so it is not touched


def test_apply_replaces_one_account_and_keeps_flags_and_other_accounts():
    p = parse_positions("Symbol,Quantity,Average Cost\nMSFT,150,410\nNEW,10,20\n")
    out = apply_positions(stored(), "rh_taxable", p)
    taxable = {l.symbol: l for l in out.lots if l.account == "rh_taxable"}
    assert set(taxable) == {"MSFT", "NEW"} and taxable["MSFT"].shares == 150
    assert taxable["MSFT"].strategy == "core" and taxable["MSFT"].note == "keep"  # flags survive a changed lot
    assert [l.symbol for l in out.lots if l.account == "rh_ira"] == ["AAA"]
    assert [(o.symbol, option_acct) for o, option_acct in ((o, o.account) for o in out.short_options)] == [("AAA", "rh_ira")]


def test_unchanged_stock_keeps_the_stored_lot_exactly():
    s = stored()
    p = parse_positions("Symbol,Quantity,Average Cost\nMSFT,100,400\n")
    out = apply_positions(s, "rh_taxable", p)
    assert [l for l in out.lots if l.symbol == "MSFT"] == [s.lots[0]]


def test_option_account_survives_yaml_and_database_round_trips(tmp_path):
    h = Holdings(lots=[Lot(symbol="AAA", account="rh_ira", shares=1, cost=1.0)],
                 short_options=[ShortOption("AAA", "call", 9.0, date(2026, 11, 13), 1, account="rh_ira"),
                                ShortOption("BBB", "put", 5.0, date(2026, 11, 13), 1)],
                 long_options=[LongOption("CCC", "call", 5.0, date(2026, 11, 13), 1, account="ibkr")])
    path = tmp_path / "h.yaml"
    save_holdings(h, path)
    back = load_holdings_yaml(path)
    assert [o.account for o in back.short_options] == ["rh_ira", None] and back.long_options[0].account == "ibkr"
    store = Store(tmp_path / "t.sqlite")
    store.save_holdings(h)
    again = store.load_holdings()
    assert [o.account for o in again.short_options] == ["rh_ira", None] and again.long_options[0].account == "ibkr"
    store.close()


def test_portfolio_snapshot_round_trip(tmp_path):
    import pandas as pd

    store = Store(tmp_path / "t.sqlite")
    pos = pd.DataFrame([dict(account="a", symbol="AAA", kind="shares", qty=1.0, strike=None, expiry=None, price=2.0, option_mid=None,
                             delta_shares=1.0, theta_day=0.0, beta=1.0, beta_assumed=True, spy_equiv_shares=0.1,
                             delta_dollars=2.0, value=2.0, status="live")])
    info = pd.DataFrame([dict(symbol="AAA", industry="Tech", next_earnings=None, ex_div=None, dividend=None)])
    store.save_portfolio(pos, info, 500.0, as_of="2026-10-23T10:00:00-04:00")
    p, i, as_of, spy = store.latest_portfolio()
    assert as_of.startswith("2026-10-23") and spy == 500.0 and bool(p["beta_assumed"].iloc[0]) is True
    assert i["industry"].iloc[0] == "Tech"
    store.save_portfolio(pos.iloc[0:0], info.iloc[0:0], 501.0)  # a new run replaces the old snapshot
    assert store.latest_portfolio()[0].empty
    store.close()

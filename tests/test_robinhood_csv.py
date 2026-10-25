from __future__ import annotations

import csv
import io
from datetime import date

import pytest

from tool.data.robinhood_csv import (
    apply_stock_recon,
    build_stock_lots,
    open_options,
    options_from_statement,
    parse_activity,
    reconcile_stocks,
)
from tool.holdings import Holdings, Lot

HEADER = ["Activity Date", "Process Date", "Settle Date", "Instrument", "Description", "Trans Code", "Quantity", "Price", "Amount"]


def statement(rows: list[tuple]) -> str:
    """rows: (date, symbol, description, code, qty, price) oldest first; written newest-first like Robinhood."""
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    writer.writerow(HEADER)
    for day, sym, desc, code, qty, price in reversed(rows):
        writer.writerow([day, day, day, sym, desc, code, qty, price, ""])
    return out.getvalue()


def lots_for(rows):
    return build_stock_lots(parse_activity(statement(rows)))


def test_fifo_sell_consumes_oldest_lot_first():
    lots = lots_for([
        ("1/2/2026", "ABC", "ABC corp", "Buy", "100", "$10.00"),
        ("2/2/2026", "ABC", "ABC corp", "Buy", "100", "$20.00"),
        ("3/2/2026", "ABC", "ABC corp", "Sell", "150", "$25.00"),
    ])
    [lot] = lots["ABC"]
    assert lot.shares == pytest.approx(50) and lot.cost == 20.0 and lot.day == date(2026, 2, 2)


def test_forward_split_scales_shares_and_cost():
    lots = lots_for([
        ("1/2/2026", "NOW", "ServiceNow", "Buy", "10", "$100.00"),
        ("2/2/2026", "NOW", "ServiceNow", "SPL", "40", ""),
    ])
    [lot] = lots["NOW"]
    assert lot.shares == pytest.approx(50) and lot.cost == pytest.approx(20.0)


def test_reverse_split_uses_removed_and_added_rows():
    lots = lots_for([
        ("1/2/2026", "XYZ", "Xyz", "Buy", "1000", "$1.00"),
        ("2/2/2026", "XYZ", "Xyz", "SPR", "1000S", ""),
        ("2/2/2026", "XYZ", "Xyz", "SPR", "200", ""),
    ])
    [lot] = lots["XYZ"]
    assert lot.shares == pytest.approx(200) and lot.cost == pytest.approx(5.0)


def test_fully_sold_and_dust_positions_are_dropped():
    lots = lots_for([
        ("1/2/2026", "GONE", "Gone", "Buy", "10", "$5.00"),
        ("2/2/2026", "GONE", "Gone", "Sell", "10", "$6.00"),
        ("1/2/2026", "DUST", "Dust", "Buy", "1.00001", "$5.00"),
        ("2/2/2026", "DUST", "Dust", "Sell", "1", "$6.00"),
    ])
    assert "GONE" not in lots and "DUST" not in lots


TODAY = date(2026, 10, 7)


def options(rows):
    return open_options(parse_activity(statement(rows)), today=TODAY)


def test_open_options_track_opens_closes_and_expirations():
    positions = options([
        ("9/1/2026", "AAA", "AAA 10/16/2026 Call $50.00", "BTO", "3", "$0.50"),
        ("9/2/2026", "AAA", "AAA 10/16/2026 Call $50.00", "STC", "1", "$0.90"),
        ("9/3/2026", "BBB", "BBB 10/9/2026 Call $20.00", "STO", "5", "$1.00"),
        ("9/4/2026", "BBB", "BBB 10/9/2026 Call $20.00", "BTC", "2", "$0.30"),
        ("9/5/2026", "CCC", "CCC 9/25/2026 Call $10.00", "STO", "1", "$0.40"),
        ("9/25/2026", "CCC", "Option Expiration for CCC 9/25/2026 Call $10.00", "OEXP", "1", ""),
        ("9/6/2026", "DDD", "DDD 10/9/2026 Put $30.00", "STO", "2", "$1.00"),
        ("10/1/2026", "DDD", "DDD 10/9/2026 Put $30.00", "OASGN", "2", ""),
    ])
    by_symbol = {p.symbol: p for p in positions}
    assert set(by_symbol) == {"AAA", "BBB"}
    assert by_symbol["AAA"].qty == 2 and by_symbol["AAA"].avg_price == pytest.approx(0.50)
    assert by_symbol["BBB"].qty == -3
    shorts, longs = options_from_statement(positions)
    assert [o.symbol for o in shorts] == ["BBB"] and shorts[0].qty == 3
    assert [o.symbol for o in longs] == ["AAA"]


def test_expired_options_are_not_open():
    assert options([("8/1/2026", "AAA", "AAA 9/18/2026 Call $50.00", "BTO", "1", "$0.50")]) == []


def holdings_with(shares, cost, **flags):
    return Holdings(lots=[Lot(symbol="ABC", account="rh_taxable", shares=shares, cost=cost, **flags)])


def test_transfer_lot_cost_is_solved_to_keep_the_stored_average():
    # stored: 100 shares @ 15 avg (total 1500). Statement: 60 bought @ 10 (600) + 40 transferred in (cost unknown).
    lots = lots_for([
        ("1/2/2026", "ABC", "Abc", "Buy", "60", "$10.00"),
        ("2/2/2026", "ABC", "Abc", "ACATI", "40", ""),
    ])
    [recon] = reconcile_stocks(lots, holdings_with(100, 15.0), "rh_taxable")
    assert recon.status == "match"
    transfer = next(l for l in recon.lots if l.source == "transfer")
    assert transfer.cost == pytest.approx((1500 - 600) / 40)


def test_missing_history_becomes_a_dated_unknown_remainder_lot():
    lots = lots_for([("1/2/2026", "ABC", "Abc", "Buy", "60", "$10.00")])
    [recon] = reconcile_stocks(lots, holdings_with(100, 15.0), "rh_taxable")
    remainder = next(l for l in recon.lots if l.source == "remainder")
    assert remainder.shares == pytest.approx(40) and remainder.day is None
    assert remainder.cost == pytest.approx((1500 - 600) / 40)


def test_symbols_without_statement_activity_are_kept():
    [recon] = reconcile_stocks({}, holdings_with(100, 15.0), "rh_taxable")
    assert recon.lots == [] and "kept as is" in recon.status


def test_apply_replaces_only_that_account_and_keeps_flags():
    holdings = Holdings(
        lots=[
            Lot(symbol="ABC", account="rh_taxable", shares=100, cost=15.0, strategy="wheel", no_covered_calls=True, note="mine"),
            Lot(symbol="ABC", account="rh_ira", shares=50, cost=12.0),
        ]
    )
    lots = lots_for([("1/2/2026", "ABC", "Abc", "Buy", "100", "$15.00")])
    recons = reconcile_stocks(lots, holdings, "rh_taxable")
    updated = apply_stock_recon(holdings, recons, "rh_taxable")
    taxable = updated.lots_for("ABC", "rh_taxable")
    assert len(taxable) == 1 and taxable[0].purchase_date == date(2026, 1, 2)
    assert taxable[0].strategy == "wheel" and taxable[0].no_covered_calls and taxable[0].note == "mine"
    assert updated.lots_for("ABC", "rh_ira") == holdings.lots_for("ABC", "rh_ira")

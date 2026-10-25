from __future__ import annotations

from datetime import date

import pytest

from tool.holdings import Holdings, Lot, ShortOption, add_lot, load_holdings, remove_lots, save_holdings


def _sample() -> Holdings:
    return Holdings(
        lots=[
            Lot(symbol="MSFT", account="rh_taxable", shares=550, cost=408.84, purchase_date=date(2024, 1, 15)),
            Lot(symbol="ALAB", account="rh_taxable", shares=100, cost=242.45, strategy="wheel"),
            Lot(symbol="CRWV", account="rh_ira", shares=201, cost=130.67, no_covered_calls=True, note="uncapped"),
        ],
        short_options=[
            ShortOption(symbol="MSFT", type="call", strike=540.0, expiry=date(2026, 10, 9), qty=5, net_credit_total=10.5)
        ],
        watchlist_scan=["NVDA", "AMD"],
        alab_rules={"atm_call_trigger": 370, "fallback_deadline": "tuesday_close"},
    )


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "holdings.yaml"
    original = _sample()
    save_holdings(original, path)
    assert load_holdings(path) == original


def test_save_keeps_backup_of_previous_file(tmp_path):
    path = tmp_path / "holdings.yaml"
    save_holdings(_sample(), path)
    save_holdings(Holdings(), path)
    assert (tmp_path / "holdings.yaml.bak").exists()
    assert load_holdings(tmp_path / "holdings.yaml.bak") == _sample()


def test_add_lot_uppercases_symbol():
    updated = add_lot(Holdings(), Lot(symbol=" nvda ", account="ibkr", shares=100, cost=30.0))
    assert updated.lots[0].symbol == "NVDA"


@pytest.mark.parametrize(
    "lot",
    [
        Lot(symbol="", account="ibkr", shares=100, cost=30.0),
        Lot(symbol="NVDA", account="robinhood", shares=100, cost=30.0),
        Lot(symbol="NVDA", account="ibkr", shares=0, cost=30.0),
        Lot(symbol="NVDA", account="ibkr", shares=100, cost=0.0),
    ],
)
def test_add_lot_rejects_invalid_input(lot):
    with pytest.raises(ValueError):
        add_lot(Holdings(), lot)


def test_remove_lots_removes_only_matching_symbol_and_account():
    updated = remove_lots(_sample(), "CRWV", "rh_ira")
    assert [(l.symbol, l.account) for l in updated.lots] == [("MSFT", "rh_taxable"), ("ALAB", "rh_taxable")]
    assert updated.short_options == _sample().short_options


def test_database_roundtrip_keeps_purchase_date_and_long_options(tmp_path):
    from tool.holdings import LongOption
    from tool.store import Store

    original = _sample()
    original = Holdings(
        lots=original.lots,
        short_options=original.short_options,
        long_options=[LongOption(symbol="BABA", type="call", strike=230.0, expiry=date(2027, 1, 15), qty=2, avg_price=4.05)],
        watchlist_scan=original.watchlist_scan,
        alab_rules=original.alab_rules,
    )
    store = Store(tmp_path / "t.sqlite")
    store.save_holdings(original)
    assert store.load_holdings() == original
    store.close()


def test_first_use_imports_yaml_into_empty_database(tmp_path, monkeypatch):
    import tool.store as store_mod

    yaml_path = tmp_path / "holdings.yaml"
    save_holdings(_sample(), yaml_path)
    monkeypatch.setattr(store_mod, "DEFAULT_HOLDINGS_PATH", yaml_path)
    store = store_mod.Store(tmp_path / "t.sqlite")
    loaded = store_mod.current_holdings(store)
    assert loaded == _sample()
    assert store.load_holdings() == _sample()  # now persisted in the database
    store.close()

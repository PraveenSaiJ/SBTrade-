from datetime import date

from tool.store import Store
from tool.tracker import assess_position

TODAY = date(2026, 10, 16)


def _pos(**kw):
    base = {"opt_type": "call", "expiry": "2026-11-20", "stop_level": 95.0, "t1": 110.0, "t2": 120.0, "stop_option": 2.0}
    base.update(kw)
    return base


def test_call_signals():
    assert assess_position(_pos(), 100.0, 5.0, TODAY)[0] == "HOLD"
    assert assess_position(_pos(), 94.0, 5.0, TODAY)[0] == "EXIT"
    assert assess_position(_pos(), 100.0, 1.5, TODAY)[0] == "EXIT"
    assert assess_position(_pos(), 111.0, 9.0, TODAY)[0] == "TAKE PROFIT"
    assert "T2" in assess_position(_pos(), 121.0, 12.0, TODAY)[1]
    assert assess_position(_pos(expiry="2026-10-17"), 100.0, 5.0, TODAY)[0] == "EXPIRING"
    assert assess_position(_pos(expiry="2026-10-10"), 100.0, 5.0, TODAY)[0] == "EXPIRED"


def test_put_levels_are_mirrored():
    put = _pos(opt_type="put", stop_level=105.0, t1=90.0, t2=80.0)
    assert assess_position(put, 100.0, 5.0, TODAY)[0] == "HOLD"
    assert assess_position(put, 106.0, 5.0, TODAY)[0] == "EXIT"
    assert assess_position(put, 89.0, 8.0, TODAY)[0] == "TAKE PROFIT"


def test_bought_options_roundtrip(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    oid = store.add_bought("NVDA", "call", 220.0, "2026-11-20", 2, 8.85, "long_call", 200.0, 190.0, 4.0, 215.0, 230.0)
    store.mark_bought(oid, 10.0, 205.0)
    row = store.bought_options("open").iloc[0]
    assert row["last_mid"] == 10.0 and row["qty"] == 2
    store.close_bought(oid, 12.0)
    assert store.bought_options("open").empty and len(store.bought_options("closed")) == 1
    store.close()

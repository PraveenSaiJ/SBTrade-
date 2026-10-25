from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from tool.levels import NY
from tool.screener import classify_setup, evaluate, flag_earnings, load_universe, sell_signals, targets


@pytest.mark.parametrize(
    "spot, change, prior, pivots, expected",
    [
        (96.0, -0.04, 102.0, [100.0], ("downtrend", None)),
        (103.0, 0.03, 102.0, [100.0], ("breakout", 102.0)),
        (116.0, 0.10, 120.0, [100.0, 104.0, 108.0, 112.0], ("extended", 112.0)),
        (105.0, 0.05, 110.0, [101.0, 110.0], ("pullback", 101.0)),
        (100.5, 0.00, 102.0, [100.0], ("range", None)),
    ],
)
def test_classify_setup_rules(spot, change, prior, pivots, expected):
    assert classify_setup(spot, change, sma20=100.0, atr=4.0, prior_high=prior, pivots=pivots) == expected


def test_targets_pick_first_pivots_and_reward_to_risk():
    stop, t1, t2, rr = targets(entry=100.0, atr=4.0, pivots=[95.0, 104.0, 108.0, 113.0, 120.0])
    assert stop == pytest.approx(94.0)  # entry - 1.5 x ATR
    assert t1 == pytest.approx(108.0)  # first pivot >= 100 + 1.5*4 = 106
    assert t2 == pytest.approx(113.0)  # first pivot >= 108 + 0.75*4 = 111
    assert rr == pytest.approx(8.0 / 6.0)  # reward 8 over risk 6


def test_targets_fall_back_to_atr_multiples_without_pivots():
    stop, t1, t2, rr = targets(entry=100.0, atr=4.0, pivots=[])
    assert (stop, t1, t2) == (pytest.approx(94.0), pytest.approx(108.0), pytest.approx(114.0))
    assert rr == pytest.approx(8.0 / 6.0)


def _flat_bars() -> pd.DataFrame:
    """25 business days ending Fri Oct 2 2026: H102/L98/C100 each day (ATR 4.0, SMA 100,
    prior 20D high 102, weekly S1 98)."""
    dates = pd.bdate_range(end="2026-10-02", periods=25)
    return pd.DataFrame(
        {"open": [100.0] * 25, "high": [102.0] * 25, "low": [98.0] * 25, "close": [100.0] * 25, "volume": [1e6] * 25},
        index=dates,
    )


NOW = datetime(2026, 10, 2, 17, 0, tzinfo=NY)


def _breakout_bars(last_volume: float = 2.5e6) -> pd.DataFrame:
    """The flat 25 sessions (H102, V1M) plus a final session that CLOSES at 104 above the 102 high."""
    bars = _flat_bars().copy()
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = [101.0, 105.0, 100.5, 104.0, last_volume]
    return bars


def test_breakout_needs_a_daily_close_above_the_high_not_just_a_last_trade():
    # trading at 103 but the last completed close is 100: not a breakout yet
    setup = evaluate("TEST", _flat_bars(), spot=103.0, now=NOW)
    assert setup.kind != "breakout"
    assert any("needs a daily close above" in n for n in setup.notes)


def test_confirmed_breakout_on_a_close_above_the_high_with_strong_volume():
    from tool.levels import compute_levels

    bars = _breakout_bars()
    setup = evaluate("TEST", bars, spot=104.5, now=NOW)
    atr = compute_levels(bars, now=NOW, spot=104.5).atr10
    assert setup.kind == "breakout" and setup.entry == pytest.approx(102.0)
    assert setup.volume_ratio == pytest.approx(2.5)
    # structural stop: the lower of (entry - 0.5 ATR) and the recent swing low (98), within 2 ATR of the entry
    assert setup.stop == pytest.approx(max(min(102.0 - 0.5 * atr, 98.0), 102.0 - 2 * atr))
    # R:R is measured from the fill, max(spot, entry) = 104.5 here, not from the 102 entry
    assert setup.rr == pytest.approx((setup.t1 - 104.5) / (104.5 - setup.stop))


def test_close_above_the_high_on_light_volume_is_a_weak_breakout_and_not_actionable():
    from tool.screener import ACTIONABLE

    setup = evaluate("TEST", _breakout_bars(last_volume=0.6e6), spot=104.5, now=NOW)
    assert setup.kind == "weak_breakout" and setup.kind not in ACTIONABLE
    assert setup.volume_ratio == pytest.approx(0.6)
    assert any("only 0.6x average volume" in n for n in setup.notes)


def test_earnings_inside_the_window_are_flagged_and_outside_are_not():
    today = date(2026, 10, 13)
    base = evaluate("OSCR", _breakout_bars(), spot=104.5, now=NOW)
    flagged = flag_earnings(base, {"OSCR": date(2026, 11, 5)}, today)  # 23 days out
    assert flagged.earnings == date(2026, 11, 5)
    assert any("EARNINGS 2026-11-05 in 23 days" in n for n in flagged.notes)
    assert flag_earnings(base, {"OSCR": date(2026, 12, 20)}, today).earnings is None  # beyond the 30-day window
    assert flag_earnings(base, {"OTHER": date(2026, 10, 20)}, today).earnings is None
    assert flag_earnings(base, {"OSCR": date(2026, 10, 1)}, today).earnings is None  # already reported


def test_evaluate_without_setup_has_no_targets():
    setup = evaluate("TEST", _flat_bars(), spot=100.5, now=NOW)
    assert setup.kind == "range"
    assert setup.entry is None and setup.stop is None and setup.rr is None


def test_sell_signal_below_weekly_s1():
    signals = sell_signals("TEST", _flat_bars(), spot=97.0, now=NOW)
    assert [s.kind for s in signals] == ["below_weekly_s1"]


def test_no_sell_signal_when_holding_above_s1():
    assert sell_signals("TEST", _flat_bars(), spot=100.0, now=NOW) == []


def test_load_universe_maps_dotted_symbols(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text("Symbol,Security\nMMM,3M\nBRK.B,Berkshire\nBF.B,Brown-Forman\n", encoding="utf-8")
    assert load_universe(path) == ["MMM", "BRK/B", "BF/B"]


def test_load_universe_appends_extra_symbols_without_duplicates(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text("Symbol,Security\nMMM,3M\nAAPL,Apple\n", encoding="utf-8")
    assert load_universe(path, extra=["aapl", "QUBT", "brk.b", "QUBT"]) == ["MMM", "AAPL", "QUBT", "BRK/B"]


def test_latest_setups_keeps_the_newest_row_per_symbol(tmp_path):
    from tool.screener import Setup
    from tool.store import Store

    store = Store(tmp_path / "t.sqlite")
    mk = lambda sym, kind: Setup(symbol=sym, kind=kind, spot=1.0, atr=1.0, sma20=1.0, prior_high=1.0, change_20d=0.0)
    store.save_setups([mk("AAA", "range"), mk("BBB", "pullback")], day=date(2026, 10, 7))
    store.save_setups([mk("AAA", "breakout")], day=date(2026, 10, 8))  # a partial scan on a later day
    latest = store.latest_setups().set_index("symbol")
    assert latest.loc["AAA", "kind"] == "breakout" and latest.loc["AAA", "scan_date"] == "2026-10-08"
    assert latest.loc["BBB", "kind"] == "pullback"  # not hidden by the later partial scan
    store.close()


def test_earnings_calendar_asks_for_one_day_at_a_time_so_finnhub_cannot_truncate_it():
    import httpx

    from tool.data.earnings import FinnhubEvents

    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        requested.append((params["from"], params["to"]))
        day = params["from"]
        return httpx.Response(200, json={"earningsCalendar": [{"symbol": f"S{day[-2:]}", "date": day}]})

    client = httpx.Client(base_url="https://finnhub.io/api/v1", transport=httpx.MockTransport(handler))
    events = FinnhubEvents(api_key="test", client=client).earnings_calendar(date(2026, 10, 13), date(2026, 10, 16))
    assert requested == [("2026-10-13", "2026-10-13"), ("2026-10-14", "2026-10-14"),
                         ("2026-10-15", "2026-10-15"), ("2026-10-16", "2026-10-16")]
    assert [e.symbol for e in events] == ["S13", "S14", "S15", "S16"]

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tool.config import ScreenerConfig
from tool.earnings_move import (
    ExpectedMoveUnavailable, attach_expected_moves, expected_move_for, expiry_target,
)
from tool.screener import Setup
from tool.store import Store

TODAY = date(2026, 10, 22)  # Thursday
CFG = ScreenerConfig()


def chain(expiry: date, quotes: dict[float, tuple[float, float]]) -> pd.DataFrame:
    """quotes: strike -> (bid, ask)."""
    return pd.DataFrame(
        [{"expiry": expiry, "strike": k, "bid": b, "ask": a, "mid": (b + a) / 2} for k, (b, a) in quotes.items()]
    )


class FakeSource:
    """Chains keyed by (right); records the expiry targets it was asked for."""

    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame):
        self.calls, self.puts, self.asked = calls, puts, []

    def option_chain(self, symbol, expiry=None, right="C"):
        self.asked.append((symbol, expiry, right))
        return self.calls if right == "C" else self.puts


# ---- which expiry
def test_a_before_open_report_counts_the_same_day_but_an_after_close_report_needs_the_next_day():
    friday = date(2026, 10, 30)
    assert expiry_target(friday, "bmo") == friday
    assert expiry_target(friday, "dmh") == friday
    assert expiry_target(friday, "amc") == date(2026, 10, 31)
    assert expiry_target(friday, None) == date(2026, 10, 31)  # unknown hour: be safe, take the day after


def test_the_chain_lookup_uses_that_target(tmp_path):
    exp = date(2026, 11, 6)
    src = FakeSource(chain(exp, {100.0: (4.9, 5.1)}), chain(exp, {100.0: (4.4, 4.6)}))
    expected_move_for(src, "AAA", 100.0, date(2026, 11, 5), "amc")
    assert {(e, r) for _, e, r in src.asked} == {(date(2026, 11, 6), "C"), (date(2026, 11, 6), "P")}


# ---- straddle arithmetic
def test_the_straddle_uses_the_strike_nearest_spot_and_divides_by_spot():
    exp = date(2026, 11, 6)
    calls = chain(exp, {95.0: (8.0, 8.4), 100.0: (4.9, 5.1), 105.0: (2.0, 2.2)})
    puts = chain(exp, {95.0: (1.9, 2.1), 100.0: (4.4, 4.6), 105.0: (8.0, 8.4)})
    move = expected_move_for(FakeSource(calls, puts), "AAA", 101.0, date(2026, 11, 5), "bmo")
    assert move.strike == 100.0 and move.expiry == exp
    assert move.call_mid == pytest.approx(5.0) and move.put_mid == pytest.approx(4.5)
    assert move.straddle == pytest.approx(9.5) and move.pct == pytest.approx(9.5 / 101.0)
    assert move.wide_spread is False


def test_wide_spreads_are_flagged_but_still_used():
    exp = date(2026, 11, 6)
    calls = chain(exp, {100.0: (3.0, 5.0)})  # spread 2.0 on a 4.0 mid = 50%
    puts = chain(exp, {100.0: (4.4, 4.6)})
    move = expected_move_for(FakeSource(calls, puts), "AAA", 100.0, date(2026, 11, 5), "bmo")
    assert move.wide_spread is True and move.straddle == pytest.approx(8.5)


def test_a_strike_without_a_two_sided_quote_is_skipped_for_the_next_nearest():
    exp = date(2026, 11, 6)
    calls = chain(exp, {100.0: (0.0, 0.0), 105.0: (2.0, 2.2)})  # nearest strike has no ask
    puts = chain(exp, {100.0: (4.4, 4.6), 105.0: (8.0, 8.4)})
    move = expected_move_for(FakeSource(calls, puts), "AAA", 100.0, date(2026, 11, 5), "bmo")
    assert move.strike == 105.0


# ---- unavailable cases never invent a number
@pytest.mark.parametrize("calls,puts,why", [
    (pd.DataFrame(), pd.DataFrame(), "empty option chain"),
    (chain(date(2026, 11, 6), {100.0: (0.0, 0.0)}), chain(date(2026, 11, 6), {100.0: (0.0, 0.0)}), "two-sided"),
    (chain(date(2026, 11, 6), {100.0: (4.9, 5.1)}), chain(date(2026, 11, 13), {100.0: (4.4, 4.6)}), "different expiries"),
    (chain(date(2026, 11, 6), {100.0: (4.9, 5.1)}), chain(date(2026, 11, 6), {105.0: (4.4, 4.6)}), "two-sided"),
])
def test_unavailable_reasons(calls, puts, why):
    with pytest.raises(ExpectedMoveUnavailable, match=why):
        expected_move_for(FakeSource(calls, puts), "AAA", 100.0, date(2026, 11, 5), "bmo")


def test_a_missing_expiry_is_reported_as_no_chain():
    class Broken:
        def option_chain(self, *a, **k):
            raise ValueError("No expirations on/after 2026-11-06 for AAA from Schwab")

    with pytest.raises(ExpectedMoveUnavailable, match="no option chain"):
        expected_move_for(Broken(), "AAA", 100.0, date(2026, 11, 5), "bmo")


# ---- scan hook
def make(symbol, **kw) -> Setup:
    base = dict(symbol=symbol, kind="pullback", spot=100.0, atr=3.0, sma20=98.0, prior_high=101.0, change_20d=0.05,
                entry=99.0, stop=95.0, t1=106.0, t2=110.0, rr=1.5, volume_ratio=2.0, last_close=100.0, rsi=60.0,
                atr14=3.0, sma50=95.0, sma50_rising=True, sma200=90.0)
    base.update(kw)
    return Setup(**base)


class MoveSource:
    def __init__(self):
        self.asked: list[str] = []

    def option_chain(self, symbol, expiry=None, right="C"):
        self.asked.append(symbol)
        exp = date(2026, 11, 13)
        if symbol == "NOCHAIN":
            raise ValueError("No expirations")
        return chain(exp, {100.0: (4.9, 5.1)}) if right == "C" else chain(exp, {100.0: (4.9, 5.1)})


def test_moves_are_attached_only_to_survivors_with_a_report_inside_30_days():
    setups = [
        make("NEAR", earnings=date(2026, 11, 12), earnings_hour="amc"),  # 21 days: gets a move
        make("FAR", earnings=date(2026, 12, 15), earnings_hour="bmo"),  # 54 days: skipped
        make("NONE"),  # no report: skipped
        make("DROPPED", earnings=date(2026, 11, 12), excluded="low volume"),  # already excluded: skipped
        make("NOCHAIN", earnings=date(2026, 11, 12), earnings_hour="amc"),
    ]
    src = MoveSource()
    out, failures = attach_expected_moves(setups, src, CFG, TODAY)
    by = {s.symbol: s for s in out}
    assert by["NEAR"].expected_move == pytest.approx(10.0) and by["NEAR"].expected_move_pct == pytest.approx(0.10)
    assert by["NEAR"].em_expiry == date(2026, 11, 13)
    assert by["FAR"].expected_move is None and by["NONE"].expected_move is None and by["DROPPED"].expected_move is None
    assert set(src.asked) == {"NEAR", "NOCHAIN"}  # 2 requests per candidate, nobody else
    assert "NOCHAIN" in failures and by["NOCHAIN"].expected_move is None
    assert [s.symbol for s in out] == [s.symbol for s in setups]  # order preserved


def test_the_real_move_is_passed_to_the_filters():
    # earnings allowed, report 6 trading days out (inside the hold), expected move 10% vs a 4% stop: dropped
    inside = make("IN", earnings=date(2026, 10, 30), earnings_hour="amc")
    out, _ = attach_expected_moves([inside], MoveSource(), CFG, TODAY, allow_earnings=True)
    assert out[0].excluded and "expected move" in out[0].excluded
    # a report after the hold window is not penalised even though the move is large
    after = make("AFTER", earnings=date(2026, 11, 12), earnings_hour="amc")
    out, _ = attach_expected_moves([after], MoveSource(), CFG, TODAY, allow_earnings=True)
    assert out[0].excluded is None and out[0].expected_move_pct == pytest.approx(0.10)


# ---- storage
def test_setup_columns_round_trip(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    s = make("AAA", expected_move=9.5, expected_move_pct=0.095, em_expiry=date(2026, 11, 13))
    store.save_setups([s], day=date(2026, 10, 22))
    row = store.latest_setups().iloc[0]
    assert row["expected_move"] == 9.5 and row["expected_move_pct"] == pytest.approx(0.095)
    assert row["em_expiry"] == "2026-11-13"
    store.close()

from datetime import date

import pandas as pd

from tool.performance import evaluate_idea, summarize


def _bars(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"high": [r[1] for r in rows], "low": [r[2] for r in rows], "close": [r[3] for r in rows]}, index=idx)


AFTER = date(2026, 10, 1)


def test_long_limit_hits_t1_then_t2():
    bars = _bars([("2026-10-02", 101, 99.5, 100.5), ("2026-10-05", 106, 100, 105), ("2026-10-06", 112, 104, 111)])
    out = evaluate_idea(bars, AFTER, entry=100.0, stop=95.0, t1=105.0, t2=110.0)
    assert out.result == "T2" and round(out.r, 1) == 2.0


def test_stop_beats_target_on_the_same_bar():
    bars = _bars([("2026-10-02", 110, 94, 100)])
    out = evaluate_idea(bars, AFTER, entry=100.0, stop=95.0, t1=105.0, t2=110.0)
    assert out.result == "STOP" and out.r == -1.0


def test_t1_then_stop_keeps_t1():
    bars = _bars([("2026-10-02", 106, 99, 105), ("2026-10-05", 104, 94, 95)])
    out = evaluate_idea(bars, AFTER, entry=100.0, stop=95.0, t1=105.0, t2=115.0)
    assert out.result == "T1"


def test_never_filled_and_pending():
    bars = _bars([("2026-10-02", 120, 110, 115)])
    assert evaluate_idea(bars, AFTER, 100.0, 95.0, 105.0, 110.0).result == "PENDING"
    long_bars = _bars([(f"2026-10-{d:02d}", 120, 110, 115) for d in range(2, 31) if date(2026, 10, d).weekday() < 5])
    assert evaluate_idea(long_bars, AFTER, 100.0, 95.0, 105.0, 110.0).result == "NO FILL"


def test_put_stop_trigger_below_entry():
    bars = _bars([("2026-10-02", 101, 99, 99.5), ("2026-10-05", 100, 87, 90)])
    out = evaluate_idea(bars, AFTER, entry=100.0, stop=105.0, t1=92.0, t2=88.0, direction=-1, trigger="stop")
    assert out.result == "T2" and out.r > 2


def test_summary_win_rate():
    scored = pd.DataFrame({"group": ["a", "a", "a"], "result": ["T1", "STOP", "OPEN"], "r": [1.0, -1.0, 0.5]})
    row = summarize(scored).iloc[0]
    assert row["Win rate"] == 50.0 and row["Filled"] == 3 and row["Avg R"] == 0.17


# ---------------------------------------------------------------- Phase 11: excursions, stats, curve
import sqlite3

import pytest

from tool.performance import (
    equity_curve, load_signal_ideas, low_sample, max_drawdown, open_mark, profit_factor, score_ideas,
)


def test_winner_reports_excursions_and_days_held():
    # fills day 1 (low 99.5), T1 on day 2, T2 on day 3; best high 112 = +2.4R, worst low 99 = -0.2R
    bars = _bars([("2026-10-02", 101, 99.5, 100.5), ("2026-10-05", 106, 99, 105), ("2026-10-06", 112, 104, 111)])
    out = evaluate_idea(bars, AFTER, 100.0, 95.0, 105.0, 110.0)
    assert out.result == "T2" and out.days_held == 3
    assert out.mfe_r == 2.4 and out.mae_r == -0.2


def test_stop_has_mae_at_least_one_r_and_same_day_days_held_is_one():
    out = evaluate_idea(_bars([("2026-10-02", 110, 94, 100)]), AFTER, 100.0, 95.0, 105.0, 110.0)
    assert out.result == "STOP" and out.days_held == 1 and out.mae_r <= -1.0 and out.mfe_r == 2.0


def test_open_idea_counts_days_to_the_last_bar():
    bars = _bars([("2026-10-02", 101, 99, 100), ("2026-10-05", 103, 100, 102), ("2026-10-06", 104, 101, 103)])
    out = evaluate_idea(bars, AFTER, 100.0, 95.0, 110.0, 115.0)
    assert (out.result, out.days_held, out.exit_date) == ("OPEN", 3, None) and out.r == pytest.approx(0.6)


def test_short_excursions_flip_direction():
    bars = _bars([("2026-10-02", 101, 99, 99.5), ("2026-10-05", 100, 87, 90)])
    out = evaluate_idea(bars, AFTER, 100.0, 105.0, 92.0, 88.0, direction=-1, trigger="stop")
    assert out.mfe_r > 2 and out.mae_r <= 0


def test_profit_factor_and_expectancy_arithmetic():
    assert profit_factor(pd.Series([2.0, -1.0, 1.0, -1.0])) == 1.5  # 3 won / 2 lost
    assert profit_factor(pd.Series([1.0, 2.0])) is None  # no losses
    assert profit_factor(pd.Series([], dtype=float)) is None
    scored = pd.DataFrame({"group": ["a"] * 4, "result": ["T2", "STOP", "T1", "STOP"], "r": [2.0, -1.0, 1.0, -1.0],
                           "mfe_r": [2.4, 0.5, 1.2, 0.1], "mae_r": [-0.2, -1.0, -0.3, -1.1], "days_held": [3, 1, 2, 4]})
    row = summarize(scored).iloc[0]
    assert row["Expectancy"] == 0.25 and row["Profit factor"] == 1.5 and row["Win rate"] == 50.0
    assert (row["Avg MFE"], row["Avg MAE"], row["Avg days"]) == (1.05, -0.65, 2.5)


def test_equity_curve_is_cumulative_in_exit_order_and_drawdown_is_peak_to_trough():
    scored = pd.DataFrame({
        "symbol": list("ABCDE"), "result": ["T2", "STOP", "STOP", "T1", "OPEN"], "r": [2.0, -1.0, -1.0, 1.0, 0.4],
        "exit_date": [date(2026, 10, 6), date(2026, 10, 8), date(2026, 10, 7), date(2026, 10, 9), None],
    })
    curve = equity_curve(scored)
    assert list(curve["symbol" if "symbol" in curve else "r"]) == [2.0, -1.0, -1.0, 1.0]  # sorted by exit date, open excluded
    assert list(curve["equity"]) == [2.0, 1.0, 0.0, 1.0] and max_drawdown(curve) == 2.0
    assert open_mark(scored) == pytest.approx(0.4)
    assert equity_curve(pd.DataFrame()).empty and max_drawdown(equity_curve(pd.DataFrame())) == 0.0


def test_low_sample_threshold_comes_from_config():
    few = pd.DataFrame({"result": ["T1"] * 5})
    assert low_sample(few) and low_sample(few, 5) is False and low_sample(pd.DataFrame())


def test_signal_loader_splits_recommended_from_excluded_and_defaults_verdict():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE signals (signal_date TEXT, tab TEXT, symbol TEXT, setup TEXT, status TEXT, excluded_code TEXT, "
                 "entry REAL, stop REAL, t1 REAL, t2 REAL, verdict TEXT)")
    conn.executemany("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        ("2026-10-22", "swing", "AAA", "breakout", "recommended", None, 100, 95, 106, 110, None),
        ("2026-10-22", "swing", "BBB", "breakout", "excluded", "earnings", 50, 48, 54, 56, None),
        ("2026-10-22", "swing", "CCC", "pullback", "recommended", None, 20, 19, 22, None, "agree"),
        ("2026-10-22", "swing", "DDD", "failed", "excluded", "failed", None, None, None, None, None),
    ])
    rec = load_signal_ideas(conn, "recommended")
    assert list(rec["symbol"]) == ["AAA", "CCC"] and list(rec["verdict"]) == ["unreviewed", "agree"]
    exc = load_signal_ideas(conn, "excluded")
    assert list(exc["symbol"]) == ["BBB"] and exc["excluded_code"].iloc[0] == "earnings"  # DDD has no levels to score


def test_score_ideas_carries_signal_columns_and_excursions_through():
    bars = _bars([("2026-10-02", 101, 99.5, 100.5), ("2026-10-05", 112, 100, 111)])
    ideas = pd.DataFrame([{"day": "2026-10-01", "symbol": "AAA", "group": "breakout", "entry": 100.0, "stop": 95.0,
                           "t1": 105.0, "t2": 110.0, "verdict": "agree"}])
    out = score_ideas(ideas, lambda s: bars, lambda i: 1, lambda i: "limit", date(2026, 10, 9))
    row = out.iloc[0]
    assert row["verdict"] == "agree" and row["result"] == "T2" and row["days_held"] == 2 and row["mfe_r"] == 2.4

import pandas as pd

from tool.why import buy_why, sell_why, swing_why


def test_swing_breakout_comment_mentions_volume_and_retest():
    row = {"kind": "breakout", "prior_high": 100.0, "volume_ratio": 1.8, "change_20d": 0.12, "entry": 100.0, "rr": 1.4}
    text = swing_why(row)
    assert "1.8x" in text and "retest" in text and "+12%" in text


def test_swing_unknown_kind_is_blank():
    assert swing_why({"kind": "not in scan universe"}) == ""


def test_buy_comment_for_call_and_put():
    row = pd.Series({"entry_level": 50.0, "stop_level": 47.0, "t1": 56.0, "multiple": 3.2, "delta": 0.5, "cost": 400.0})
    setup = {"kind": "pullback", "change_20d": 0.08, "entry": 50.0}
    call = buy_why(row, setup, "long_call")
    assert "above $50.00" in call and "3.2x" in call
    put = buy_why(row, {"kind": "downtrend", "change_20d": -0.1}, "long_put")
    assert "below $50.00" in put and "Downtrend" in put


def test_sell_comment_for_each_family():
    row = {"delta": 0.2, "premium": 150.0, "return_pct": 0.02, "annualized": 0.5, "strike": 40.0, "effective_cost": 38.5}
    assert "80%" in sell_why(row, "csp") and "$38.50" in sell_why(row, "csp")
    assert "Unlimited" in sell_why(row, "naked_call")
    assert "high risk" in sell_why(row, "naked_call_breakout")

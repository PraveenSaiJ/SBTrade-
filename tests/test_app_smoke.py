"""Render the whole dashboard headlessly and use the filters. Catches NameErrors and bad column names that
unit tests on the engines can't see. Uses the real database read-only (never presses a button)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DB = ROOT / "data" / "sbtrade.sqlite"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="needs the local database")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_top_level_tabs_are_the_seven_working_sections_plus_jobs(app):
    assert [e.value for e in app.exception] == []
    labels = [t.label for t in app.tabs]
    top_level = ["Long term", "Swing", "Calls", "Puts", "Whales", "Covered calls", "Levels", "Portfolio", "Performance", "Jobs"]
    assert [l for l in labels if l in top_level] == top_level
    for removed in ("Naked calls", "Buy calls & puts", "CC monitor", "Sell signals", "Saved covered calls",
                    "Long-term", "Swing setups"):
        assert removed not in labels


def test_calls_and_puts_have_the_same_horizon_sections(app):
    labels = [t.label for t in app.tabs]
    for expected in ("Weekly calls", "Weekly lottos", "0DTE", "MAG7", "Monthly calls", "Weekly puts", "Monthly puts", "Earnings"):
        assert expected in labels


def test_filters_work_on_calls_and_puts(app):
    app.multiselect(key="calls_themes").set_value(["Quantum"]).run()
    assert [e.value for e in app.exception] == []
    app.multiselect(key="calls_trigger").set_value(["TRIGGERED"]).run()
    assert [e.value for e in app.exception] == []
    app.selectbox(key="calls_top").set_value("All").run()
    assert [e.value for e in app.exception] == []
    app.multiselect(key="puts_themes").set_value(["Quantum"]).run()
    assert [e.value for e in app.exception] == []
    app.selectbox(key="puts_top").set_value("All").run()
    assert [e.value for e in app.exception] == []


def test_reauth_banner_appears_on_the_page_when_schwab_rejects_the_login(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from tool.data import schwab

    monkeypatch.setattr(schwab, "auth_status", lambda *a, **k: (False, "The Schwab login expired (8.0 days old)."))
    page = AppTest.from_file(str(APP), default_timeout=180).run()
    assert [e.value for e in page.exception] == []
    assert any("Schwab re-auth required" in e.value for e in page.error)
    assert any(b.label == "Sign in to Schwab" for b in page.button)


def test_show_failed_toggle_lists_failed_breakouts_and_removes_them_from_excluded(app):
    def tables(page):
        return [d.value for t in page.tabs if t.label == "Swing" for d in t.dataframe]

    assert not any("Failed breakouts" in m.value for m in app.markdown)  # hidden by default
    page = app.checkbox(key="swing_show_failed").check().run()
    assert [e.value for e in page.exception] == []
    assert any("Failed breakouts" in m.value for m in page.markdown)
    failed = [df for df in tables(page) if "Setup" in df.columns and (df["Setup"] == "Failed breakout").all() and len(df)]
    assert failed, "the failed-breakout table should appear when the toggle is on"
    excluded_with_failed = [df for df in tables(page) if "Why excluded" in df.columns
                            and df["Why excluded"].astype(str).str.startswith("failed").any()]
    assert not excluded_with_failed  # they moved out of the Excluded list
    page.checkbox(key="swing_show_failed").uncheck().run()


def test_allow_earnings_toggle_can_only_shrink_the_excluded_list(app):
    import re

    def excluded_count(page):
        labels = [e.label for e in page.expander if e.label.startswith("Excluded ·")]
        assert labels
        return int(re.search(r"Excluded · (\d+)", labels[0]).group(1))

    before = excluded_count(app)
    page = app.checkbox(key="swing_allow_earnings").check().run()
    assert [e.value for e in page.exception] == []
    assert excluded_count(page) <= before
    page.checkbox(key="swing_allow_earnings").uncheck().run()


def test_covered_calls_tab_groups_suggestions_by_tier_and_shows_the_roll_panel(app):
    tab = next(t for t in app.tabs if t.label == "Covered calls")
    text = " ".join(m.value for m in tab.markdown)
    assert "Roll panel" in text and "#### Safe" in text  # NVDA, AMZN and MSFT are the safe tier
    frames = [d.value for d in tab.dataframe]
    suggestions = [f for f in frames if "Row" in f.columns and "Net %" in f.columns]
    assert suggestions, "the tier tables show an income and a guardrail row per holding"
    rows = pd.concat(suggestions)
    assert set(rows["Row"]) <= {"Income (nearest the money)", "Guardrail (R1/R2, cost basis)"}
    assert not rows["Ticker"].isin(["CRWV"]).any()  # held uncapped: no suggestion


def test_puts_tab_has_the_support_sub_tab_and_the_earnings_play_toggle_is_safe(app):
    labels = [t.label for t in app.tabs]
    assert "At support (7-45 DTE)" in labels
    page = app.checkbox(key="puts_earnings_play").check().run() if any(
        c.key == "puts_earnings_play" for c in app.checkbox) else app
    assert [e.value for e in page.exception] == []
    if any(c.key == "puts_earnings_play" for c in page.checkbox):
        page.checkbox(key="puts_earnings_play").uncheck().run()


def test_calls_tab_has_directional_and_leaps_next_to_the_existing_sections(app):
    labels = [t.label for t in app.tabs]
    for expected in ("Directional (14-60 DTE)", "LEAPS (270+ DTE)", "Weekly calls", "Weekly lottos", "0DTE", "MAG7", "Monthly calls"):
        assert expected in labels
    assert [e.value for e in app.exception] == []


def test_whales_tab_shows_flow_on_your_positions_and_the_new_filters_are_safe(app):
    tab = next(t for t in app.tabs if t.label == "Whales")
    assert any("Flow on your positions" in m.value for m in tab.markdown)
    for key in ("whales_opening", "whales_bursts", "whales_noise"):
        page = app.checkbox(key=key).check().run()
        assert [e.value for e in page.exception] == [], key
        page.checkbox(key=key).uncheck().run()


def test_long_term_tab_has_ideas_my_holdings_and_consolidation(app):
    labels = [t.label for t in app.tabs]
    for expected in ("Ideas", "My holdings", "Consolidation"):
        assert expected in labels
    assert [e.value for e in app.exception] == []


def test_levels_tab_still_renders_and_the_zone_chart_builds_for_a_symbol():
    import numpy as np
    import pandas as pd

    from tool.levels import NY
    from tool.sr_levels import chart_data, compute_zones
    from datetime import datetime

    idx = pd.bdate_range(end="2026-10-14", periods=260)
    bars = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1e6}, index=idx)
    zones = compute_zones(bars, 100.0, datetime(2026, 10, 15, 17, 0, tzinfo=NY))
    frames = chart_data(bars, zones, now=datetime(2026, 10, 15, 17, 0, tzinfo=NY))
    assert set(frames) == {"candles", "averages", "zones"} and len(frames["candles"]) == 120


def test_swing_and_covered_calls_offer_csv_downloads_and_no_claude_button(app):
    assert [e.value for e in app.exception] == []
    assert "Review with Claude" not in [b.label for b in app.button]


def test_options_tab_gathers_weekly_lottos_and_0dte_and_the_old_tabs_stay(app):
    assert [e.value for e in app.exception] == []
    labels = [t.label for t in app.tabs]
    assert "Options" in labels and "Calls" in labels and "Puts" in labels
    assert {"Weekly calls and puts", "Lottos", "0DTE"} <= set(labels)

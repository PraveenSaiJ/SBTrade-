from __future__ import annotations

import argparse
from datetime import date, datetime

import pytest

from tool import cli, notify as notify_mod
from tool.freshness import last_session_date, previous_weekday
from tool.jobs import JobManager
from tool.levels import NY
from tool.market_calendar import early_close, holiday_name, holidays, is_trading_day, next_holiday, next_trading_day, previous_trading_day


# ---------------------------------------------------------------- holiday calendar
def test_2026_holidays_match_the_nyse_calendar():
    assert {d.isoformat(): n for d, n in holidays(2026).items()} == {
        "2026-01-01": "New Year's Day", "2026-01-19": "Martin Luther King Jr. Day", "2026-02-16": "Presidents' Day",
        "2026-04-03": "Good Friday", "2026-05-25": "Memorial Day", "2026-06-19": "Juneteenth",
        "2026-07-03": "Independence Day",  # July 4 is a Saturday: observed Friday
        "2026-09-07": "Labor Day", "2026-11-26": "Thanksgiving Day", "2026-12-25": "Christmas Day",
    }


def test_2027_observances_and_new_year_on_saturday():
    h = holidays(2027)
    assert date(2027, 12, 24) in h and h[date(2027, 12, 24)] == "Christmas Day"  # Dec 25 is a Saturday
    assert date(2027, 7, 5) in h  # July 4 is a Sunday
    assert is_trading_day(date(2027, 12, 31))  # New Year's Day 2028 falls on a Saturday and is not observed
    assert holiday_name(date(2027, 3, 26)) == "Good Friday"


def test_trading_day_helpers_skip_weekends_and_holidays():
    assert not is_trading_day(date(2026, 10, 24)) and is_trading_day(date(2026, 10, 23))
    assert previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)  # Thanksgiving in between
    assert next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)
    assert next_holiday(date(2026, 10, 25)) == (date(2026, 11, 26), "Thanksgiving Day")


def test_early_closes():
    assert early_close(date(2026, 11, 27)) == "Day after Thanksgiving"
    assert early_close(date(2026, 12, 24)) == "Christmas Eve"
    assert early_close(date(2026, 10, 23)) is None


def test_freshness_reads_a_holiday_as_no_session():
    # Monday Sept 7 2026 is Labor Day: at 10:00 on Tuesday the last session is Friday Sept 4
    assert previous_weekday(date(2026, 9, 8)) == date(2026, 9, 4)
    assert last_session_date(datetime(2026, 9, 7, 10, 0, tzinfo=NY)) == date(2026, 9, 4)
    assert last_session_date(datetime(2026, 9, 8, 8, 0, tzinfo=NY)) == date(2026, 9, 4)


# ---------------------------------------------------------------- run history
def test_history_row_opens_on_start_and_closes_on_final_status(tmp_path, monkeypatch):
    jobs = JobManager(tmp_path)
    monkeypatch.setattr("tool.jobs.subprocess.Popen", lambda *a, **k: type("P", (), {"pid": 4242})())
    jobs.start("recheck")
    [run] = jobs.history()
    assert (run["job"], run["trigger"], run["status"], run["ended_at"]) == ("recheck", "dashboard", "running", None)
    jobs.write_state("recheck", next_index=12, total=12, errors=1)  # progress does not close the row
    assert jobs.history()[0]["ended_at"] is None
    jobs.write_state("recheck", status="done")
    [run] = jobs.history()
    assert (run["status"], run["rows"], run["errors"]) == ("done", 12, 1) and run["ended_at"]


def test_dead_pid_closes_the_row_as_stopped_and_failures_keep_the_message(tmp_path, monkeypatch):
    jobs = JobManager(tmp_path)
    monkeypatch.setattr("tool.jobs.subprocess.Popen", lambda *a, **k: type("P", (), {"pid": 999999})())
    monkeypatch.setattr("tool.jobs._pid_alive", lambda pid: False)
    jobs.start("ccmonitor", trigger="scheduled")
    assert jobs.status("ccmonitor")["status"] == "stopped"
    assert jobs.history()[0]["status"] == "stopped" and jobs.history()[0]["trigger"] == "scheduled"
    monkeypatch.setattr("tool.jobs._pid_alive", lambda pid: True)
    jobs.start("whales")
    jobs.write_state("whales", status="failed", error="Schwab login expired")
    run = jobs.history()[0]
    assert (run["job"], run["status"], run["error_text"]) == ("whales", "failed", "Schwab login expired")


def test_skipped_runs_are_recorded_and_old_runs_pruned(tmp_path):
    jobs = JobManager(tmp_path)
    jobs.record_skipped("scan", "market closed: Labor Day")
    assert jobs.history()[0]["status"] == "skipped"
    with jobs._history() as conn:
        conn.execute("UPDATE job_runs SET started_at = '2020-01-01T09:00:00-05:00'")
    assert jobs.prune_history(90) == 1 and jobs.history() == []


# ---------------------------------------------------------------- run-job: holidays and notifications
def run_args(name="recheck", force=False):
    return argparse.Namespace(name=name, force=force)


@pytest.fixture
def quiet(tmp_path, monkeypatch):
    """Isolate jobs dir, notifications and clock from the real project."""
    monkeypatch.setattr("tool.cli.JobManager", lambda: JobManager(tmp_path))
    sent = []
    monkeypatch.setattr("tool.notify.notify", lambda title, message, **k: sent.append((title, message)))
    monkeypatch.setattr("tool.notify.notified_today", lambda key, today, **k: False)
    return sent


def test_run_job_skips_a_holiday_and_force_overrides(quiet, tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 7, 9, 0, tzinfo=NY)  # Labor Day

    monkeypatch.setattr("tool.cli.datetime", Clock)
    monkeypatch.setattr("tool.cli.token_age_days", lambda: 1.0)
    assert cli.cmd_run_job(run_args()) == 0
    history = JobManager(tmp_path).history()
    assert history[0]["status"] == "skipped" and "Labor Day" in history[0]["error_text"]

    started = []
    monkeypatch.setattr(JobManager, "start", lambda self, name, extra_args=None, trigger="dashboard": started.append((name, trigger)) or {})
    monkeypatch.setattr(JobManager, "status", lambda self, name: {"status": "done", "errors": 0})
    assert cli.cmd_run_job(run_args(force=True)) == 0 and started == [("recheck", "scheduled")]


def test_run_job_notifies_when_the_schwab_login_is_missing(quiet, tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 23, 9, 0, tzinfo=NY)

    monkeypatch.setattr("tool.cli.datetime", Clock)
    monkeypatch.setattr("tool.cli.token_age_days", lambda: None)
    assert cli.cmd_run_job(run_args()) == 2
    assert quiet and "sign-in needed" in quiet[0][0]
    assert JobManager(tmp_path).history()[0]["status"] == "failed"


def test_run_job_notifies_on_a_failed_job(quiet, tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 23, 9, 0, tzinfo=NY)

    monkeypatch.setattr("tool.cli.datetime", Clock)
    monkeypatch.setattr("tool.cli.token_age_days", lambda: 2.0)
    monkeypatch.setattr(JobManager, "start", lambda self, name, extra_args=None, trigger="dashboard": {})
    monkeypatch.setattr(JobManager, "status", lambda self, name: {"status": "failed", "error": "boom", "errors": 3})
    assert cli.cmd_run_job(run_args("whales")) == 1
    assert quiet == [("SBTrade job failed: whales", "boom")]


# ---------------------------------------------------------------- notifications
def test_notify_logs_and_survives_a_failing_shell(tmp_path, monkeypatch):
    log = tmp_path / "n.log"

    def boom(*a, **k):
        raise OSError("no powershell")

    monkeypatch.setattr(notify_mod.subprocess, "Popen", boom)
    notify_mod.notify("Title", "It's \"quoted\"", log=log, show=True)  # must not raise
    assert notify_mod.recent(5, log)[0][1:] == ("Title", "It's \"quoted\"")


def test_balloon_command_escapes_quotes_and_dedupe_marker_fires_once(tmp_path):
    script = notify_mod.balloon_command("It's", "a'b")[-1]
    assert "'It''s'" in script and "'a''b'" in script
    assert notify_mod.notified_today("k", "2026-10-23", tmp_path) is False
    assert notify_mod.notified_today("k", "2026-10-23", tmp_path) is True
    assert notify_mod.notified_today("k", "2026-10-24", tmp_path) is False


def test_outcomes_job_is_registered_and_dashboard_jobs_tab_spec_exists():
    from tool.jobs import JOB_SPECS

    assert JOB_SPECS["outcomes"]["args"] == ["outcomes"]


def test_options_summary_prints_buy_side_rows_that_have_an_ask_not_a_bid(capsys):
    from types import SimpleNamespace

    sell = SimpleNamespace(status="recommend", symbol="AAA", expiry_type="support", expiry="2026-11-06", strike=10.0, bid=0.5)
    buy = SimpleNamespace(status="recommend", symbol="BBB", expiry_type="monthly", expiry="2026-11-20", strike=20.0, ask=1.25)  # no bid
    cli._summarize("mixed", [sell, buy, SimpleNamespace(status="skip", symbol="C")])
    out = capsys.readouterr().out
    assert "bid 0.50" in out and "ask 1.25" in out and "2 recommended of 3" in out

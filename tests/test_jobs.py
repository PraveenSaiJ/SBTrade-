from __future__ import annotations

import os

import pytest

from tool.jobs import JobManager


def test_never_run_job_reports_never_run(tmp_path):
    assert JobManager(tmp_path).status("scan")["status"] == "never run"


def test_running_job_with_dead_pid_is_reported_stopped(tmp_path):
    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="running", pid=999999, next_index=10, total=503)
    assert jobs.status("scan")["status"] == "stopped"


def test_resume_continues_from_next_index(tmp_path, monkeypatch):
    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="stopped", next_index=120, total=503, args=["--pause", "1.2"])
    started = {}

    def fake_start(name, extra_args=None):
        started["name"], started["extra"] = name, extra_args
        return {}

    monkeypatch.setattr(jobs, "start", fake_start)
    jobs.resume("scan")
    assert started == {"name": "scan", "extra": ["--start-index", "120"]}


def test_start_refuses_when_job_is_running(tmp_path):
    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="running", pid=os.getpid(), next_index=5, total=503)
    with pytest.raises(RuntimeError):
        jobs.start("scan")


def test_token_age_reads_creation_timestamp(tmp_path):
    import json
    import time

    from tool.data.schwab import token_age_days

    path = tmp_path / "token.json"
    path.write_text(json.dumps({"creation_timestamp": time.time() - 3 * 86400, "token": {}}), encoding="utf-8")
    assert token_age_days(path) == pytest.approx(3.0, abs=0.01)
    assert token_age_days(tmp_path / "missing.json") is None


def test_run_job_refuses_when_schwab_login_is_expired(tmp_path, monkeypatch, capsys):
    import argparse

    from tool import cli

    jobs = JobManager(tmp_path)
    monkeypatch.setattr(cli, "JobManager", lambda: jobs)
    monkeypatch.setattr(cli, "token_age_days", lambda: 8.0)
    monkeypatch.setattr("tool.notify.notify", lambda *a, **k: None)  # no balloon, no log entry from a test
    monkeypatch.setattr("tool.notify.notified_today", lambda *a, **k: True)
    assert cli.cmd_run_job(argparse.Namespace(name="scan", force=True)) == 2
    assert jobs.read_state("scan")["status"] == "failed"
    assert "schwab-login" in capsys.readouterr().err


def test_stop_leaves_a_finished_job_unchanged(tmp_path):
    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="done", next_index=10, total=10)
    assert jobs.stop("scan")["status"] == "done"
    assert jobs.read_state("scan")["status"] == "done"


def test_state_writes_leave_no_temp_files_and_reads_survive_a_blank_file(tmp_path):
    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="running", next_index=3)
    assert [p.name for p in tmp_path.iterdir()] == ["scan.json"]  # the temp file was swapped in, not left behind
    (tmp_path / "scan.json").write_text("", encoding="utf-8")  # what a reader sees mid-write with the old code
    assert jobs.read_state("scan") == {}  # tolerated instead of raising JSONDecodeError


def test_concurrent_readers_never_crash_while_a_job_writes(tmp_path):
    import threading

    jobs = JobManager(tmp_path)
    jobs.write_state("scan", status="running", next_index=0)
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                jobs.read_state("scan")
            except Exception as exc:  # pragma: no cover - this is what the test guards against
                errors.append(exc)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(60):
            jobs.write_state("scan", next_index=i)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    assert errors == [] and jobs.read_state("scan")["next_index"] == 59


def test_schwab_login_keeps_truststore_out_of_the_callback_server_process(tmp_path, monkeypatch):
    import os

    from tool.data import schwab

    monkeypatch.setenv("SCHWAB_APP_KEY", "key")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret")
    seen = {}

    def fake_login_flow(*args, **kwargs):
        seen["flag_during_login"] = os.environ.get("SBTRADE_NO_TRUSTSTORE")
        return object()

    monkeypatch.setattr(schwab.auth, "client_from_login_flow", fake_login_flow)
    schwab.SchwabDataSource(token_path=tmp_path / "token.json").login()
    assert seen["flag_during_login"] == "1"  # the child process inherits this and skips truststore
    assert "SBTRADE_NO_TRUSTSTORE" not in os.environ  # cleared afterwards so token refreshes still use it

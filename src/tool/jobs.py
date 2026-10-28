"""Background jobs (e.g. the S&P 500 scan): start, stop, resume, and status.

Each job runs as a detached `python -m tool.cli <command> --job <name>` process.
Its state lives in data/jobs/<name>.json and its output in data/jobs/<name>.log,
so the dashboard can show progress even after a reload.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from tool.config import get_config
from tool.levels import NY

JOBS_DIR = Path("data/jobs")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_STATUSES = ("done", "failed", "stopped")
RUN_ALL_ORDER = ("ccmonitor", "buys", "weekly", "odte", "whales", "portfolio", "scan_light", "longterm", "options_calls", "options_puts", "scan")
JOB_SPECS = {
    "scan": {"label": "S&P 500 swing scan", "args": ["scan", "--pause", "1.2"]},
    "scan_light": {"label": "Quick refresh: holdings + watchlist swing scan", "args": ["scan", "--light", "--pause", "1.2"]},
    "outcomes": {"label": "Outcome update: refresh bars for open and pending ideas (Performance tab)", "args": ["outcomes"]},
    "recheck": {"label": "Re-check swing setups against live prices", "args": ["recheck"]},
    "ccmonitor": {"label": "Covered-call monitor (every 15 minutes)", "args": ["ccmonitor"]},
    "portfolio": {"label": "Portfolio risk: greeks, beta-weighted delta, sectors, this-week risk", "args": ["portfolio"]},
    "whales": {"label": "Whale activity: large option premium and fresh volume (every 15 minutes)", "args": ["whales"]},
    "longterm": {"label": "Long-term stock recommendations", "args": ["longterm"]},
    "calls": {"label": "Covered-call recommendations (portfolio)", "args": ["calls"]},
    "buys": {"label": "Buy ideas: monthly swing calls/puts and MAG7 (every 15 minutes)", "args": ["buys"]},
    "weekly": {"label": "Weekly calls and puts to buy: weekly swing and lottos (every 30 minutes)", "args": ["weekly"]},
    "odte": {"label": "0DTE calls and puts: index ETFs and MAG7 (every 15 minutes)", "args": ["odte"]},
    "options": {"label": "Options: all call and put families", "args": ["options"]},
    "options_calls": {"label": "Options (calls): earnings, directional, LEAPS, naked calls", "args": ["options", "--part", "calls"]},
    "options_puts": {"label": "Options (puts): at support, cash-secured, naked puts, wheel", "args": ["options", "--part", "puts"]},
}


def _now() -> str:
    return datetime.now(NY).isoformat(timespec="seconds")


HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job        TEXT NOT NULL,
    trigger    TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    status     TEXT NOT NULL,
    rows       INTEGER,
    errors     INTEGER,
    error_text TEXT,
    pid        INTEGER
)
"""


class JobManager:
    def __init__(self, jobs_dir: Path = JOBS_DIR) -> None:
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ run history (data/jobs/history.sqlite)
    def _history(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.dir / "history.sqlite", timeout=10)
        conn.execute(HISTORY_SCHEMA)
        return conn

    def _open_run(self, name: str, trigger: str) -> int | None:
        try:
            with self._history() as conn:
                cur = conn.execute("INSERT INTO job_runs (job, trigger, started_at, status) VALUES (?,?,?,'running')",
                                   (name, trigger, _now()))
                return cur.lastrowid
        except sqlite3.Error:
            return None  # history is a convenience; a job must never fail because of it

    def _close_run(self, state: dict) -> None:
        run_id = state.get("run_id")
        if not run_id:
            return
        try:
            with self._history() as conn:
                conn.execute(
                    "UPDATE job_runs SET ended_at = ?, status = ?, rows = ?, errors = ?, error_text = ? "
                    "WHERE id = ? AND ended_at IS NULL",
                    (_now(), state.get("status"), state.get("total") or state.get("next_index"), state.get("errors"),
                     state.get("error"), run_id),
                )
        except sqlite3.Error:
            pass

    def record_skipped(self, name: str, reason: str, trigger: str = "scheduled", status: str = "skipped") -> None:
        try:
            with self._history() as conn:
                now = _now()
                conn.execute("INSERT INTO job_runs (job, trigger, started_at, ended_at, status, error_text) VALUES (?,?,?,?,?,?)",
                             (name, trigger, now, now, status, reason))
        except sqlite3.Error:
            pass

    def history(self, limit: int = 100) -> list[dict]:
        """Latest runs first: id, job, trigger, started_at, ended_at, status, rows, errors, error_text."""
        try:
            with self._history() as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute("SELECT * FROM job_runs ORDER BY id DESC LIMIT ?", (limit,))]
        except sqlite3.Error:
            return []

    def prune_history(self, days: int | None = None) -> int:
        days = get_config().jobs.history_days if days is None else days
        cutoff = (datetime.now(NY) - timedelta(days=days)).isoformat(timespec="seconds")
        try:
            with self._history() as conn:
                return conn.execute("DELETE FROM job_runs WHERE started_at < ?", (cutoff,)).rowcount
        except sqlite3.Error:
            return 0

    def _state_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def log_path(self, name: str) -> Path:
        return self.dir / f"{name}.log"

    def read_state(self, name: str) -> dict:
        path = self._state_path(name)
        for attempt in range(5):
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05 * (attempt + 1))  # a writer may be mid-replace; look again
        return {}

    def write_state(self, name: str, **fields: object) -> dict:
        state = {**self.read_state(name), **fields, "updated_at": _now()}
        path = self._state_path(name)
        temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        for _ in range(10):
            try:
                os.replace(temp, path)  # atomic: readers see the old file or the new one, never a half-written one
                if fields.get("status") in FINAL_STATUSES:
                    self._close_run(state)
                return state
            except PermissionError:  # Windows refuses while another process has the file open; try again shortly
                time.sleep(0.05)
        # Still blocked: write in place instead. A progress update must never crash the job, and
        # read_state tolerates the brief window in which the file is being rewritten.
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.unlink(missing_ok=True)
        if fields.get("status") in FINAL_STATUSES:
            self._close_run(state)
        return state

    def status(self, name: str) -> dict:
        state = self.read_state(name)
        if state.get("status") == "running" and not _pid_alive(state.get("pid")):
            state = self.write_state(name, status="stopped")
        return state or {"status": "never run"}

    def start(self, name: str, extra_args: list[str] | None = None, trigger: str = "dashboard") -> dict:
        if self.status(name).get("status") == "running":
            raise RuntimeError(f"{name} is already running")
        spec = JOB_SPECS[name]
        args = list(spec["args"]) + list(extra_args or []) + ["--job", name]
        run_id = self._open_run(name, trigger)
        self.write_state(name, status="running", run_id=run_id, started_at=_now(), error=None, errors=0,
                         args=spec["args"][1:] + list(extra_args or []))
        log = open(self.log_path(name), "a", encoding="utf-8")
        log.write(f"\n--- start {_now()}: {' '.join(args)}\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "tool.cli", *args],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        log.close()
        return self.write_state(name, pid=proc.pid)  # the child may already have reported progress; keep its fields

    def launch(self, args: list[str], log_name: str) -> int:
        """Start `tool.cli <args>` detached (it keeps running after the dashboard page closes). Returns its PID."""
        log = open(self.dir / f"{log_name}.log", "a", encoding="utf-8")
        log.write(f"\n--- start {_now()}: {' '.join(args)}\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "tool.cli", *args],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        log.close()
        return proc.pid

    def stop(self, name: str) -> dict:
        state = self.read_state(name)
        if state.get("status") != "running":
            return state  # nothing to stop; keep a finished job's "done"/"failed" status
        if _pid_alive(state.get("pid")):
            _kill_tree(state["pid"])
        return self.write_state(name, status="stopped")

    def resume(self, name: str) -> dict:
        state = self.read_state(name)
        if state.get("status") not in ("stopped", "failed"):
            raise RuntimeError(f"{name} is {state.get('status', 'never run')}; nothing to resume")
        next_index = int(state.get("next_index", 0))
        return self.start(name, extra_args=["--start-index", str(next_index)])


def scheduled_tasks() -> list[dict]:
    """The registered SBTrade-* Windows tasks: name, next run, last run, last result. Empty when unavailable (not Windows, no tasks)."""
    if os.name != "nt":
        return []
    script = (
        "Get-ScheduledTask -TaskName 'SBTrade-*' -ErrorAction SilentlyContinue | ForEach-Object { "
        "$i = $_ | Get-ScheduledTaskInfo; [pscustomobject]@{ name = $_.TaskName; state = [string]$_.State; "
        "next_run = if ($i.NextRunTime) { $i.NextRunTime.ToString('s') } else { $null }; "
        "last_run = if ($i.LastRunTime -and $i.LastRunTime.Year -gt 2000) { $i.LastRunTime.ToString('s') } else { $null }; "
        "last_result = $i.LastTaskResult } } | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=30).stdout.strip()
        data = json.loads(out) if out else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return [data] if isinstance(data, dict) else data


def _pid_alive(pid: object) -> bool:
    if not pid:
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(int(pid)) in out


def _kill_tree(pid: object) -> None:
    subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)

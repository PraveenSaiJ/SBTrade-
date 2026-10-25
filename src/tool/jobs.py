"""Background jobs (e.g. the S&P 500 scan): start, stop, resume, and status.

Each job runs as a detached `python -m tool.cli <command> --job <name>` process.
Its state lives in data/jobs/<name>.json and its output in data/jobs/<name>.log,
so the dashboard can show progress even after a reload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from tool.levels import NY

JOBS_DIR = Path("data/jobs")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ALL_ORDER = ("ccmonitor", "buys", "whales", "portfolio", "scan_light", "longterm", "options", "scan")
JOB_SPECS = {
    "scan": {"label": "S&P 500 swing scan", "args": ["scan", "--pause", "1.2"]},
    "scan_light": {"label": "Quick refresh: holdings + watchlist swing scan", "args": ["scan", "--light", "--pause", "1.2"]},
    "recheck": {"label": "Re-check swing setups against live prices", "args": ["recheck"]},
    "ccmonitor": {"label": "Covered-call monitor (every 15 minutes)", "args": ["ccmonitor"]},
    "portfolio": {"label": "Portfolio risk: greeks, beta-weighted delta, sectors, this-week risk", "args": ["portfolio"]},
    "whales": {"label": "Whale activity: large option premium and fresh volume (every 15 minutes)", "args": ["whales"]},
    "longterm": {"label": "Long-term stock recommendations", "args": ["longterm"]},
    "calls": {"label": "Covered-call recommendations (portfolio)", "args": ["calls"]},
    "buys": {"label": "Buy ideas: long calls/puts and weekly lottos", "args": ["buys"]},
    "options": {"label": "Options: naked calls, cash-secured puts, naked puts, wheel", "args": ["options"]},
}


def _now() -> str:
    return datetime.now(NY).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, jobs_dir: Path = JOBS_DIR) -> None:
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

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
                return state
            except PermissionError:  # Windows refuses while another process has the file open; try again shortly
                time.sleep(0.05)
        # Still blocked: write in place instead. A progress update must never crash the job, and
        # read_state tolerates the brief window in which the file is being rewritten.
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.unlink(missing_ok=True)
        return state

    def status(self, name: str) -> dict:
        state = self.read_state(name)
        if state.get("status") == "running" and not _pid_alive(state.get("pid")):
            state = self.write_state(name, status="stopped")
        return state or {"status": "never run"}

    def start(self, name: str, extra_args: list[str] | None = None) -> dict:
        if self.status(name).get("status") == "running":
            raise RuntimeError(f"{name} is already running")
        spec = JOB_SPECS[name]
        args = list(spec["args"]) + list(extra_args or []) + ["--job", name]
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
        return self.write_state(
            name,
            status="running",
            pid=proc.pid,
            args=spec["args"][1:] + list(extra_args or []),
            started_at=_now(),
        )

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


def _pid_alive(pid: object) -> bool:
    if not pid:
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(int(pid)) in out


def _kill_tree(pid: object) -> None:
    subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)

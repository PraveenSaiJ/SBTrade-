"""Desktop notifications for job failures and Schwab sign-in problems.

Shows a Windows balloon tip through PowerShell (no extra dependency) and always appends to data/jobs/notifications.log, which the
Jobs tab displays. A notification can never crash a job: every failure here is swallowed. Messages carry no secrets or account numbers.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from tool.config import get_config
from tool.levels import NY

LOG = Path("data/jobs/notifications.log")


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''").replace("\r", " ").replace("\n", " ") + "'"


def balloon_command(title: str, message: str, seconds: int = 8) -> list[str]:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$n = New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon = [System.Drawing.SystemIcons]::Warning;"
        "$n.Visible = $true;"
        f"$n.ShowBalloonTip({seconds * 1000}, {_ps_quote(title)}, {_ps_quote(message)}, 'Warning');"
        f"Start-Sleep -Seconds {seconds + 1};$n.Dispose()"
    )
    return ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script]


def notify(title: str, message: str, log: Path = LOG, show: bool | None = None) -> None:
    """Log the notification and (on Windows, when enabled) pop a balloon without waiting for it."""
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(NY).isoformat(timespec='seconds')}\t{title}\t{message}\n")
    except OSError:
        pass
    if show is None:
        show = get_config().jobs.notify and os.name == "nt"
    if not show:
        return
    try:
        subprocess.Popen(balloon_command(title, message), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError:
        pass


def recent(limit: int = 20, log: Path = LOG) -> list[tuple[str, str, str]]:
    """(time, title, message) of the latest notifications, newest first."""
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = [tuple((l.split("\t") + ["", ""])[:3]) for l in lines if l.strip()]
    return out[::-1][:limit]


def notified_today(key: str, today: str, marker_dir: Path = LOG.parent) -> bool:
    """True when `key` was already notified on `today`; otherwise records it (so a daily warning fires once)."""
    path = marker_dir / "notified.txt"
    entry = f"{today} {key}"
    try:
        if entry in path.read_text(encoding="utf-8").splitlines():
            return True
    except OSError:
        pass
    try:
        marker_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except OSError:
        pass
    return False

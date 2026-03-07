"""Helpers for launching the live state-monitor TUI from policy runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field


@dataclass
class RunnerTuiSession:
    """Manage one external monitor-TUI process and a shared status payload file."""

    executable_name: str
    enabled: bool = True
    interval_seconds: float = 0.5
    fields_filter: str = "player_health,player_energy,player_credits,collected_progs"
    _status_file_path: Path | None = field(default=None, init=False, repr=False)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if not self.enabled or self._process is not None:
            return

        handle, status_file = tempfile.mkstemp(prefix="868-runner-status-", suffix=".json")
        os.close(handle)
        self._status_file_path = Path(status_file)
        self.update(
            training_line="training=initializing",
            action_line="action=idle reason=initializing",
        )

        command = [
            sys.executable,
            "-m",
            "src.memory.state_monitor_tui",
            "--exe",
            self.executable_name,
            "--interval",
            str(self.interval_seconds),
            "--fields",
            self.fields_filter,
            "--external-status-file",
            str(self._status_file_path),
        ]
        creationflags = int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        self._process = subprocess.Popen(command, creationflags=creationflags)

    def update(self, *, training_line: str, action_line: str) -> None:
        if not self.enabled or self._status_file_path is None:
            return

        payload = {
            "training_line": str(training_line),
            "action_line": str(action_line),
        }
        staging_path = self._status_file_path.with_suffix(".tmp")
        staging_path.write_text(json.dumps(payload), encoding="utf-8")
        staging_path.replace(self._status_file_path)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._process = None

        if self._status_file_path is not None:
            try:
                self._status_file_path.unlink(missing_ok=True)
            finally:
                self._status_file_path = None

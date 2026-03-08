"""Helpers for launching the live state-monitor TUI from policy runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


@dataclass
class RunnerTuiSession:
    """Manage one external monitor-TUI process and a shared status payload file."""

    executable_name: str
    enabled: bool = True
    interval_seconds: float = 0.5
    fields_filter: str = "player_health,player_energy,player_credits,collected_progs"
    step_through: bool = False
    launch_monitor: bool = True
    external_status_file: str | None = None
    external_control_file: str | None = None
    _status_file_path: Path | None = field(default=None, init=False, repr=False)
    _control_file_path: Path | None = field(default=None, init=False, repr=False)
    _owns_status_file: bool = field(default=False, init=False, repr=False)
    _owns_control_file: bool = field(default=False, init=False, repr=False)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if not self.enabled or self._process is not None:
            return

        if self.external_status_file:
            self._status_file_path = Path(self.external_status_file)
            self._status_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._owns_status_file = False
        else:
            handle, status_file = tempfile.mkstemp(prefix="868-runner-status-", suffix=".json")
            os.close(handle)
            self._status_file_path = Path(status_file)
            self._owns_status_file = True
        if self.external_control_file:
            self._control_file_path = Path(self.external_control_file)
            self._control_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._owns_control_file = False
        else:
            handle, control_file = tempfile.mkstemp(prefix="868-runner-control-", suffix=".json")
            os.close(handle)
            self._control_file_path = Path(control_file)
            self._owns_control_file = True
        self._write_json_payload(
            path=self._control_file_path,
            payload={"advance_counter": 0},
        )
        self.update(
            training_line="training=initializing",
            action_line="action=idle reason=initializing",
        )

        if self.launch_monitor:
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
                "--external-control-file",
                str(self._control_file_path),
            ]
            creationflags = int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
            self._process = subprocess.Popen(command, creationflags=creationflags)

    def update(self, *, training_line: str, action_line: str) -> None:
        if not self.enabled or self._status_file_path is None:
            return

        self._write_json_payload(
            path=self._status_file_path,
            payload={
                "training_line": str(training_line),
                "action_line": str(action_line),
            },
        )

    def wait_for_step_advance(
        self,
        *,
        training_line: str,
        action_line: str,
    ) -> None:
        if not self.enabled or not self.step_through:
            return
        if self._control_file_path is None:
            return

        self.update(
            training_line=training_line,
            action_line=f"{action_line} status=waiting_for_enter",
        )
        baseline_counter = self._read_advance_counter()
        while True:
            current_counter = self._read_advance_counter()
            if current_counter > baseline_counter:
                return
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("TUI closed while waiting for Enter to advance.")
            time.sleep(0.05)

    def _read_advance_counter(self) -> int:
        if self._control_file_path is None or not self._control_file_path.exists():
            return 0
        try:
            payload = json.loads(self._control_file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0
        if not isinstance(payload, dict):
            return 0
        try:
            return int(payload.get("advance_counter", 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _write_json_payload(*, path: Path, payload: dict[str, object]) -> None:
        json_text = json.dumps(payload)
        staging_path = path.with_suffix(".tmp")
        retry_delays = (0.005, 0.01, 0.02, 0.04, 0.08)

        # Prefer atomic replace, but Windows can transiently deny replacement while
        # another process is reading the destination file.
        for delay_seconds in retry_delays:
            try:
                staging_path.write_text(json_text, encoding="utf-8")
                staging_path.replace(path)
                return
            except PermissionError:
                time.sleep(delay_seconds)

        # Fallback path: direct write with retries. Readers tolerate invalid JSON
        # by skipping and polling again on the next tick.
        for delay_seconds in retry_delays:
            try:
                path.write_text(json_text, encoding="utf-8")
                return
            except PermissionError:
                time.sleep(delay_seconds)

        # Final attempt, surfacing any remaining error.
        try:
            path.write_text(json_text, encoding="utf-8")
        finally:
            staging_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._process = None

        if self._status_file_path is not None and self._owns_status_file:
            try:
                self._status_file_path.unlink(missing_ok=True)
            finally:
                self._status_file_path = None
        else:
            self._status_file_path = None
        if self._control_file_path is not None and self._owns_control_file:
            try:
                self._control_file_path.unlink(missing_ok=True)
            finally:
                self._control_file_path = None
        else:
            self._control_file_path = None

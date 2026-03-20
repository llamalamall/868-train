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
from datetime import datetime, timezone
from uuid import uuid4


ACTIVE_RUNNER_SESSION_FILE = Path(tempfile.gettempdir()) / "868-active-runner-session.json"


@dataclass(frozen=True)
class RunnerControlState:
    """Shared runner-control state exchanged with the monitor process."""

    mode: str = "auto"
    advance_counter: int = 0


@dataclass
class RunnerTuiSession:
    """Manage one external monitor-TUI process and a shared status payload file."""

    executable_name: str
    runner_module: str | None = None
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
    _last_advance_counter: int = field(default=0, init=False, repr=False)
    _last_step_was_manual: bool = field(default=False, init=False, repr=False)
    _last_reward_line: str = field(default="", init=False, repr=False)
    _last_next_available_actions_line: str = field(default="", init=False, repr=False)
    _runner_pid: int = field(default=0, init=False, repr=False)
    _session_id: str = field(default="", init=False, repr=False)

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
        initial_control_state = RunnerControlState(
            mode="paused" if self.step_through else "auto",
            advance_counter=0,
        )
        self._write_json_payload(
            path=self._control_file_path,
            payload={
                "mode": initial_control_state.mode,
                "advance_counter": initial_control_state.advance_counter,
            },
        )
        self._last_advance_counter = 0
        self._last_step_was_manual = False
        self._last_reward_line = ""
        self._last_next_available_actions_line = ""
        self._runner_pid = os.getpid()
        self._session_id = uuid4().hex
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
                "--runner-pid",
                str(self._runner_pid),
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

    def update(
        self,
        *,
        training_line: str,
        action_line: str,
        reward_line: str | None = None,
        next_available_actions_line: str | None = None,
    ) -> None:
        if not self.enabled or self._status_file_path is None:
            return
        if reward_line is None:
            reward_text = self._last_reward_line
        else:
            reward_text = str(reward_line)
            self._last_reward_line = reward_text
        if next_available_actions_line is None:
            next_actions_text = self._last_next_available_actions_line
        else:
            next_actions_text = str(next_available_actions_line)
            self._last_next_available_actions_line = next_actions_text

        self._write_json_payload(
            path=self._status_file_path,
            payload={
                "training_line": str(training_line),
                "action_line": str(action_line),
                "reward_line": reward_text,
                "next_available_actions_line": next_actions_text,
                "runner_pid": self._runner_pid,
                "runner_module": str(self.runner_module or ""),
                "runner_executable_name": str(self.executable_name),
                "runner_status_file": str(self._status_file_path),
                "runner_control_file": (
                    str(self._control_file_path) if self._control_file_path is not None else ""
                ),
            },
        )
        self._write_active_session_registry()

    def wait_for_step_gate(
        self,
        *,
        training_line: str,
        action_line: str,
    ) -> None:
        if not self.enabled:
            self._last_step_was_manual = False
            return
        if self._control_file_path is None:
            self._last_step_was_manual = False
            return

        control_state = self._read_control_state()
        if control_state.mode == "auto":
            self._last_advance_counter = control_state.advance_counter
            self._last_step_was_manual = False
            return
        if control_state.advance_counter > self._last_advance_counter:
            self._last_advance_counter = control_state.advance_counter
            self._last_step_was_manual = True
            return

        self.update(
            training_line=training_line,
            action_line=action_line,
        )
        while True:
            control_state = self._read_control_state()
            if control_state.mode == "auto":
                self._last_advance_counter = control_state.advance_counter
                self._last_step_was_manual = False
                return
            if control_state.advance_counter > self._last_advance_counter:
                self._last_advance_counter = control_state.advance_counter
                self._last_step_was_manual = True
                return
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("TUI closed while waiting for step advance.")
            time.sleep(0.05)

    def wait_for_step_advance(
        self,
        *,
        training_line: str,
        action_line: str,
    ) -> None:
        """Backward-compatible alias for the older step-through-only API."""
        self.wait_for_step_gate(training_line=training_line, action_line=action_line)

    def consume_manual_step_flag(self) -> bool:
        """Return and clear whether the most recent step gate used a manual step token."""
        was_manual = self._last_step_was_manual
        self._last_step_was_manual = False
        return was_manual

    def _read_control_state(self) -> RunnerControlState:
        if self._control_file_path is None or not self._control_file_path.exists():
            return RunnerControlState()
        try:
            payload = json.loads(self._control_file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return RunnerControlState()
        if not isinstance(payload, dict):
            return RunnerControlState()
        mode = self._normalize_control_mode(payload.get("mode"))
        try:
            advance_counter = int(payload.get("advance_counter", 0))
        except (TypeError, ValueError):
            advance_counter = 0
        return RunnerControlState(mode=mode, advance_counter=max(0, advance_counter))

    @staticmethod
    def _normalize_control_mode(raw_mode: object) -> str:
        mode_text = str(raw_mode).strip().lower()
        if mode_text == "paused":
            return "paused"
        return "auto"

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
        self._clear_active_session_registry()
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

    def _write_active_session_registry(self) -> None:
        if self._status_file_path is None:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        self._write_json_payload(
            path=ACTIVE_RUNNER_SESSION_FILE,
            payload={
                "session_id": self._session_id,
                "updated_at_utc": timestamp,
                "runner_pid": self._runner_pid,
                "runner_module": str(self.runner_module or ""),
                "runner_executable_name": str(self.executable_name),
                "status_file": str(self._status_file_path),
                "control_file": (
                    str(self._control_file_path) if self._control_file_path is not None else ""
                ),
            },
        )

    def _clear_active_session_registry(self) -> None:
        if not ACTIVE_RUNNER_SESSION_FILE.exists():
            return
        try:
            payload = json.loads(ACTIVE_RUNNER_SESSION_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if str(payload.get("session_id", "")).strip() != self._session_id:
            return
        ACTIVE_RUNNER_SESSION_FILE.unlink(missing_ok=True)

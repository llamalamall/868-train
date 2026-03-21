"""Tests for runner/monitor control-file synchronization helpers."""

from __future__ import annotations

import json
import threading
import time

import src.env.runner_tui as runner_tui_module
from src.env.runner_tui import RunnerTuiSession


def test_runner_tui_start_initializes_paused_mode_when_step_through_enabled(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        step_through=True,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        payload = json.loads(control_file.read_text(encoding="utf-8"))
        assert payload["mode"] == "paused"
        assert payload["advance_counter"] == 0
    finally:
        session.close()


def test_runner_tui_update_writes_reward_line(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        session.update(
            training_line="training=ok",
            action_line="action=move_up",
            reward_line="reward total=+0.100",
        )
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        assert payload["training_line"] == "training=ok"
        assert payload["action_line"] == "action=move_up"
        assert payload["reward_line"] == "reward total=+0.100"
        assert payload["next_available_actions_line"] == ""
        assert payload["runner_pid"] > 0
        assert payload["runner_executable_name"] == "868-HACK.exe"
        assert payload["runner_status_file"] == str(status_file)
        assert payload["runner_control_file"] == str(control_file)
        assert "before_action_line" not in payload
        assert "after_action_line" not in payload
    finally:
        session.close()


def test_runner_tui_writes_and_clears_active_session_registry(tmp_path, monkeypatch) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    registry_file = tmp_path / "active-session.json"
    monkeypatch.setattr(runner_tui_module, "ACTIVE_RUNNER_SESSION_FILE", registry_file)
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        runner_module="src.hybrid.runner",
        enabled=True,
        launch_monitor=False,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
        assert payload["runner_pid"] > 0
        assert payload["runner_module"] == "src.hybrid.runner"
        assert payload["status_file"] == str(status_file)
        assert payload["control_file"] == str(control_file)
    finally:
        session.close()

    assert not registry_file.exists()


def test_runner_tui_update_preserves_previous_reward_line_when_omitted(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        session.update(
            training_line="episode=1 step=1",
            action_line="action=move_up",
            reward_line="reward total=+0.100",
        )
        session.update(
            training_line="episode=1 step=2 waiting=step",
            action_line="action=move_right status=waiting_for_step",
        )

        payload = json.loads(status_file.read_text(encoding="utf-8"))
        assert payload["reward_line"] == "reward total=+0.100"
    finally:
        session.close()


def test_runner_tui_update_preserves_previous_next_available_actions_when_omitted(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        session.update(
            training_line="episode=1 step=1",
            action_line="action=move_up",
            next_available_actions_line="next_available_actions=move_left,move_right",
        )
        session.update(
            training_line="episode=1 step=2 waiting=step",
            action_line="action=move_right status=waiting_for_step",
        )

        payload = json.loads(status_file.read_text(encoding="utf-8"))
        assert payload["next_available_actions_line"] == "next_available_actions=move_left,move_right"
    finally:
        session.close()


def test_wait_for_step_gate_consumes_step_token_issued_before_wait(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        control_file.write_text(
            json.dumps({"mode": "paused", "advance_counter": 1}),
            encoding="utf-8",
        )
        session.wait_for_step_gate(training_line="episode=1 step=1", action_line="action=wait")
        assert session.consume_manual_step_flag() is True
        assert session.consume_manual_step_flag() is False
    finally:
        session.close()


def test_wait_for_step_gate_blocks_until_step_token_when_paused(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        step_through=True,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        done = threading.Event()
        errors: list[Exception] = []

        def _wait() -> None:
            try:
                session.wait_for_step_gate(training_line="episode=1 step=1", action_line="action=wait")
            except Exception as error:  # pragma: no cover - defensive thread capture
                errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        time.sleep(0.1)
        assert not done.is_set()

        control_file.write_text(
            json.dumps({"mode": "paused", "advance_counter": 1}),
            encoding="utf-8",
        )

        assert done.wait(1.0)
        thread.join(timeout=1.0)
        assert not errors
        assert session.consume_manual_step_flag() is True
    finally:
        session.close()


def test_wait_for_step_gate_unblocks_when_resumed_to_auto(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        step_through=True,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        done = threading.Event()

        def _wait() -> None:
            session.wait_for_step_gate(training_line="episode=1 step=1", action_line="action=wait")
            done.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        time.sleep(0.1)
        assert not done.is_set()

        control_file.write_text(
            json.dumps({"mode": "auto", "advance_counter": 0}),
            encoding="utf-8",
        )

        assert done.wait(1.0)
        thread.join(timeout=1.0)
        assert session.consume_manual_step_flag() is False
    finally:
        session.close()


def test_wait_for_step_gate_does_not_override_action_line_while_waiting(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    control_file = tmp_path / "control.json"
    session = RunnerTuiSession(
        executable_name="868-HACK.exe",
        enabled=True,
        launch_monitor=False,
        step_through=True,
        external_status_file=str(status_file),
        external_control_file=str(control_file),
    )

    session.start()
    try:
        done = threading.Event()

        def _wait() -> None:
            session.wait_for_step_gate(
                training_line="episode=1 step=1",
                action_line="action=move_up reason=hybrid_select_action",
            )
            done.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        time.sleep(0.1)
        assert not done.is_set()

        payload = json.loads(status_file.read_text(encoding="utf-8"))
        assert payload["action_line"] == "action=move_up reason=hybrid_select_action"
        assert "waiting_for_step" not in payload["action_line"]

        control_file.write_text(
            json.dumps({"mode": "paused", "advance_counter": 1}),
            encoding="utf-8",
        )

        assert done.wait(1.0)
        thread.join(timeout=1.0)
    finally:
        session.close()

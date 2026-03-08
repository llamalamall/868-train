"""Tests for runner/monitor control-file synchronization helpers."""

from __future__ import annotations

import json
import threading
import time

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
    finally:
        session.close()


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

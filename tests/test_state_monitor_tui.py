"""Tests for TUI control key passthrough mapping."""

from __future__ import annotations

from src.memory.state_monitor_tui import (
    FieldSnapshot,
    PollSnapshot,
    is_fail_state_detected,
    map_tui_key_to_passthrough_key,
)


def test_passthrough_mapping_for_supported_keys() -> None:
    assert map_tui_key_to_passthrough_key("up") == "UP"
    assert map_tui_key_to_passthrough_key("right") == "RIGHT"
    assert map_tui_key_to_passthrough_key("escape") == "ESCAPE"
    assert map_tui_key_to_passthrough_key("space") == "SPACE"
    assert map_tui_key_to_passthrough_key("0") == "0"
    assert map_tui_key_to_passthrough_key("9") == "9"


def test_passthrough_mapping_rejects_unknown_keys() -> None:
    assert map_tui_key_to_passthrough_key("z") is None
    assert map_tui_key_to_passthrough_key("enter") is None


def _snapshot_with_fail(value: str, status: str = "ok") -> PollSnapshot:
    return PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="fail_state",
                data_type="bool",
                confidence="high",
                address="0x200000",
                value=value,
                status=status,
                error="",
            ),
        ),
    )


def test_is_fail_state_detected_true_for_boolean_and_nonzero_numeric_values() -> None:
    assert is_fail_state_detected(_snapshot_with_fail("true"))
    assert is_fail_state_detected(_snapshot_with_fail("1"))
    assert is_fail_state_detected(_snapshot_with_fail("2"))
    assert is_fail_state_detected(_snapshot_with_fail("0.5"))


def test_is_fail_state_detected_false_for_false_zero_or_error_status() -> None:
    assert not is_fail_state_detected(_snapshot_with_fail("false"))
    assert not is_fail_state_detected(_snapshot_with_fail("0"))
    assert not is_fail_state_detected(_snapshot_with_fail("1", status="error"))


def test_is_fail_state_detected_false_when_field_missing() -> None:
    snapshot = PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="player_health",
                data_type="int32",
                confidence="high",
                address="0x200004",
                value="10",
                status="ok",
                error="",
            ),
        ),
    )
    assert not is_fail_state_detected(snapshot)

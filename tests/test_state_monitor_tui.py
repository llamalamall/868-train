"""Tests for TUI control key passthrough mapping."""

from __future__ import annotations

from src.memory.state_monitor_tui import map_tui_key_to_passthrough_key


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


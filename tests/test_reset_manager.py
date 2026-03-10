"""Tests for reset sequence manager behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.env.reset_manager import ResetManagerError, SequenceResetManager


@dataclass
class FakeActionAPI:
    actions: list[str] = field(default_factory=list)

    def perform_action(self, action_name: str) -> None:
        self.actions.append(action_name)


def test_sequence_reset_manager_runs_pre_hook_before_actions() -> None:
    api = FakeActionAPI()
    call_order: list[str] = []

    manager = SequenceResetManager(
        action_api=api,
        sequence=("confirm", "space"),
        before_sequence_hook=lambda: call_order.append("hook"),
        inter_action_delay_seconds=0.0,
    )

    manager.reset()

    assert call_order == ["hook"]
    assert api.actions == ["confirm", "space"]


def test_sequence_reset_manager_wraps_pre_hook_failures() -> None:
    api = FakeActionAPI()

    def _failing_hook() -> None:
        raise RuntimeError("copy failed")

    manager = SequenceResetManager(
        action_api=api,
        sequence=("confirm",),
        before_sequence_hook=_failing_hook,
        inter_action_delay_seconds=0.0,
    )

    with pytest.raises(ResetManagerError, match="pre-sequence hook failed"):
        manager.reset()

    assert api.actions == []

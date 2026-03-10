"""Reset strategy helpers for environment episodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.controller.action_api import ActionExecutionError


class ResetManagerError(RuntimeError):
    """Raised when reset sequence cannot be completed."""


class ResetStrategy(Protocol):
    """Contract for environment reset implementation."""

    def reset(self) -> None:
        """Execute one reset sequence."""


@dataclass
class NoopResetManager:
    """Reset strategy that intentionally performs no input."""

    def reset(self) -> None:
        return None


@dataclass
class SequenceResetManager:
    """Reset strategy that dispatches a fixed action sequence."""

    action_api: AnyActionAPI
    sequence: tuple[str, ...] = ("cancel", "cancel", "confirm")
    before_sequence_hook: Callable[[], None] | None = None
    inter_action_delay_seconds: float = 0.10
    sleep_fn: Callable[[float], None] = time.sleep

    def reset(self) -> None:
        if self.before_sequence_hook is not None:
            try:
                self.before_sequence_hook()
            except Exception as error:
                raise ResetManagerError(
                    f"Reset pre-sequence hook failed: error={error}"
                ) from error
        for action_name in self.sequence:
            try:
                self.action_api.perform_action(action_name)
            except ActionExecutionError as error:
                raise ResetManagerError(
                    f"Reset action failed: action={action_name} error={error}"
                ) from error
            if self.inter_action_delay_seconds > 0:
                self.sleep_fn(self.inter_action_delay_seconds)


class AnyActionAPI(Protocol):
    """Subset of action API required by reset sequence manager."""

    def perform_action(self, action_name: str) -> None:
        """Execute one named action."""

"""High-level deterministic action API for game control."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from src.controller.input_driver import InputDriver

LOGGER = logging.getLogger(__name__)


ActionVerificationHook = Callable[[str, int, int], bool]


@dataclass(frozen=True)
class ActionTimings:
    """Timing configuration for key actions and waits."""

    press_duration_seconds: float = 0.05
    inter_action_delay_seconds: float = 0.05
    retry_delay_seconds: float = 0.05
    wait_action_seconds: float = 0.20
    max_retries: int = 3


@dataclass(frozen=True)
class ActionConfig:
    """Configurable action -> key mapping and key -> virtual-code map."""

    action_key_bindings: dict[str, str] = field(
        default_factory=lambda: {
            "move_up": "UP",
            "move_down": "DOWN",
            "move_left": "LEFT",
            "move_right": "RIGHT",
            "confirm": "ENTER",
            "cancel": "ESCAPE",
        }
    )
    key_codes: dict[str, int] = field(
        default_factory=lambda: {
            "UP": 0x26,
            "DOWN": 0x28,
            "LEFT": 0x25,
            "RIGHT": 0x27,
            "ENTER": 0x0D,
            "ESCAPE": 0x1B,
            "SPACE": 0x20,
            "0": 0x30,
            "1": 0x31,
            "2": 0x32,
            "3": 0x33,
            "4": 0x34,
            "5": 0x35,
            "6": 0x36,
            "7": 0x37,
            "8": 0x38,
            "9": 0x39,
        }
    )
    timings: ActionTimings = field(default_factory=ActionTimings)


class ActionExecutionError(RuntimeError):
    """Raised when action execution configuration is invalid."""


@dataclass
class ActionAPI:
    """Deterministic action interface for game controls."""

    input_driver: InputDriver
    config: ActionConfig = field(default_factory=ActionConfig)
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = LOGGER

    def perform_action(
        self,
        action_name: str,
        *,
        verification_hook: ActionVerificationHook | None = None,
    ) -> None:
        """Execute a configured action with retries and timing controls."""
        assert self.logger is not None
        timestamp = datetime.now(UTC).isoformat()
        self.logger.info("Action start name=%s timestamp=%s", action_name, timestamp)

        if action_name == "wait":
            self.input_driver.sleep_wait(self.config.timings.wait_action_seconds)
            self.input_driver.sleep_wait(self.config.timings.inter_action_delay_seconds)
            return

        key_name = self.config.action_key_bindings.get(action_name)
        if key_name is None:
            raise ActionExecutionError(f"Unknown action '{action_name}'.")

        key_code = self._resolve_key_code(key_name=key_name, source=action_name)
        self._tap_key(action_label=action_name, key_code=key_code, verification_hook=verification_hook)

    def send_key_name(
        self,
        key_name: str,
        *,
        verification_hook: ActionVerificationHook | None = None,
    ) -> None:
        """Tap a configured key name directly (without requiring an action binding)."""
        normalized_name = key_name.strip().upper()
        if not normalized_name:
            raise ActionExecutionError("Key name cannot be empty.")

        key_code = self._resolve_key_code(key_name=normalized_name, source=f"key:{normalized_name}")
        self._tap_key(
            action_label=f"key:{normalized_name}",
            key_code=key_code,
            verification_hook=verification_hook,
        )

    def send_key_name_to_window(
        self,
        key_name: str,
        *,
        hwnd: int,
        verification_hook: ActionVerificationHook | None = None,
    ) -> None:
        """Tap a configured key name directly into a specific window."""
        normalized_name = key_name.strip().upper()
        if not normalized_name:
            raise ActionExecutionError("Key name cannot be empty.")
        if hwnd <= 0:
            raise ActionExecutionError("Window handle must be > 0.")

        key_code = self._resolve_key_code(key_name=normalized_name, source=f"window_key:{normalized_name}")
        self._tap_key_to_window(
            action_label=f"window_key:{normalized_name}",
            key_code=key_code,
            hwnd=hwnd,
            verification_hook=verification_hook,
        )

    def _resolve_key_code(self, *, key_name: str, source: str) -> int:
        key_code = self.config.key_codes.get(key_name)
        if key_code is None:
            raise ActionExecutionError(
                f"No virtual-key code configured for key '{key_name}' used by '{source}'."
            )
        return key_code

    def _tap_key(
        self,
        *,
        action_label: str,
        key_code: int,
        verification_hook: ActionVerificationHook | None,
    ) -> None:
        assert self.logger is not None
        timestamp = datetime.now(UTC).isoformat()
        self.logger.info("Action key tap start action=%s key_code=%s timestamp=%s", action_label, key_code, timestamp)

        def wrapped_verification(tapped_key_code: int, attempt: int) -> bool:
            if verification_hook is None:
                return True
            return verification_hook(action_label, tapped_key_code, attempt)

        self.input_driver.tap_key(
            key_code=key_code,
            retries=self.config.timings.max_retries,
            retry_delay_seconds=self.config.timings.retry_delay_seconds,
            press_duration_seconds=self.config.timings.press_duration_seconds,
            verification_hook=wrapped_verification if verification_hook else None,
        )
        self.input_driver.sleep_wait(self.config.timings.inter_action_delay_seconds)

    def _tap_key_to_window(
        self,
        *,
        action_label: str,
        key_code: int,
        hwnd: int,
        verification_hook: ActionVerificationHook | None,
    ) -> None:
        assert self.logger is not None
        timestamp = datetime.now(UTC).isoformat()
        self.logger.info(
            "Action window key tap start action=%s hwnd=%s key_code=%s timestamp=%s",
            action_label,
            hwnd,
            key_code,
            timestamp,
        )

        def wrapped_verification(tapped_key_code: int, attempt: int) -> bool:
            if verification_hook is None:
                return True
            return verification_hook(action_label, tapped_key_code, attempt)

        self.input_driver.tap_key_to_window(
            hwnd=hwnd,
            key_code=key_code,
            retries=self.config.timings.max_retries,
            retry_delay_seconds=self.config.timings.retry_delay_seconds,
            press_duration_seconds=self.config.timings.press_duration_seconds,
            verification_hook=wrapped_verification if verification_hook else None,
        )
        self.input_driver.sleep_wait(self.config.timings.inter_action_delay_seconds)

    def move_up(self) -> None:
        self.perform_action("move_up")

    def move_down(self) -> None:
        self.perform_action("move_down")

    def move_left(self) -> None:
        self.perform_action("move_left")

    def move_right(self) -> None:
        self.perform_action("move_right")

    def confirm(self) -> None:
        self.perform_action("confirm")

    def cancel(self) -> None:
        self.perform_action("cancel")

    def wait(self) -> None:
        self.perform_action("wait")


def run_smoke_test_sequence(action_api: ActionAPI) -> list[str]:
    """Run a fixed scripted sequence to validate deterministic controls."""
    sequence = [
        "move_up",
        "move_right",
        "confirm",
        "wait",
        "cancel",
        "move_left",
        "move_down",
    ]
    for action in sequence:
        action_api.perform_action(action)
    return sequence

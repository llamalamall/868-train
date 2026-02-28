"""Tests for low-level input driver retry behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.controller.input_driver import InputDriver, InputDriverError


@dataclass
class FakeKeyboardBackend:
    """Fake backend that returns planned dispatch outcomes."""

    outcomes: list[bool]
    calls: list[tuple[int, float]] = field(default_factory=list)

    def tap_virtual_key(self, key_code: int, press_duration_seconds: float) -> bool:
        self.calls.append((key_code, press_duration_seconds))
        if self.outcomes:
            return self.outcomes.pop(0)
        return False


def test_tap_key_retries_after_dispatch_failure() -> None:
    backend = FakeKeyboardBackend(outcomes=[False, True])
    driver = InputDriver(backend=backend)
    driver.tap_key(0x26, retries=2, retry_delay_seconds=0.0, press_duration_seconds=0.01)
    assert backend.calls == [(0x26, 0.01), (0x26, 0.01)]


def test_tap_key_retries_after_verification_failure() -> None:
    backend = FakeKeyboardBackend(outcomes=[True, True])
    driver = InputDriver(backend=backend)
    attempts: list[int] = []

    def verify(_key_code: int, attempt: int) -> bool:
        attempts.append(attempt)
        return attempt >= 2

    driver.tap_key(
        0x0D,
        retries=3,
        retry_delay_seconds=0.0,
        press_duration_seconds=0.01,
        verification_hook=verify,
    )
    assert attempts == [1, 2]
    assert len(backend.calls) == 2


def test_tap_key_raises_after_exhausting_retries() -> None:
    backend = FakeKeyboardBackend(outcomes=[False, False, False])
    driver = InputDriver(backend=backend)
    with pytest.raises(InputDriverError, match="dropped"):
        driver.tap_key(0x25, retries=3, retry_delay_seconds=0.0, press_duration_seconds=0.01)


def test_tap_key_rejects_invalid_retry_value() -> None:
    backend = FakeKeyboardBackend(outcomes=[True])
    driver = InputDriver(backend=backend)
    with pytest.raises(ValueError, match="retries must be"):
        driver.tap_key(0x1B, retries=0)

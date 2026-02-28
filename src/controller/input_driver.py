"""Low-level keyboard input driver with retry-aware key tapping."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

LOGGER = logging.getLogger(__name__)
KEYEVENTF_KEYUP = 0x0002


class InputDriverError(RuntimeError):
    """Raised when keyboard input could not be delivered reliably."""


class KeyboardBackend(Protocol):
    """Backend contract for sending a virtual key tap."""

    def tap_virtual_key(self, key_code: int, press_duration_seconds: float) -> bool:
        """Send key down/up and return whether both events were accepted."""


if hasattr(ctypes, "windll"):
    ULONG_PTR = ctypes.wintypes.WPARAM
else:
    ULONG_PTR = ctypes.c_ulong


class KEYBDINPUT(ctypes.Structure):
    """ctypes mapping for KEYBDINPUT."""

    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    """ctypes mapping for MOUSEINPUT."""

    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    """ctypes mapping for HARDWAREINPUT."""

    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_UNION(ctypes.Union):
    """ctypes mapping for INPUT union."""

    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    """ctypes mapping for INPUT."""

    _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]


class WindowsKeyboardBackend:
    """Windows implementation that dispatches key events via SendInput."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise InputDriverError("WindowsKeyboardBackend is only supported on Windows.")
        self._user32 = ctypes.windll.user32
        self._user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
        self._user32.SendInput.restype = ctypes.c_uint

    def _send_input(self, key_code: int, key_up: bool) -> bool:
        event_flags = KEYEVENTF_KEYUP if key_up else 0
        key_input = KEYBDINPUT(
            wVk=key_code,
            wScan=0,
            dwFlags=event_flags,
            time=0,
            dwExtraInfo=0,
        )
        input_event = INPUT(type=1, union=INPUT_UNION(ki=key_input))
        ctypes.set_last_error(0)
        sent = self._user32.SendInput(1, ctypes.byref(input_event), ctypes.sizeof(INPUT))
        if sent != 1:
            assert LOGGER is not None
            LOGGER.warning(
                "SendInput failed key_code=%s key_up=%s error=%s",
                key_code,
                key_up,
                ctypes.GetLastError(),
            )
        return bool(sent == 1)

    def tap_virtual_key(self, key_code: int, press_duration_seconds: float) -> bool:
        if not self._send_input(key_code=key_code, key_up=False):
            return False
        time.sleep(press_duration_seconds)
        if not self._send_input(key_code=key_code, key_up=True):
            return False
        return True


def _default_backend() -> KeyboardBackend:
    if os.name != "nt":
        raise InputDriverError("Input driver is currently implemented for Windows only.")
    return WindowsKeyboardBackend()


VerificationHook = Callable[[int, int], bool]


@dataclass
class InputDriver:
    """Retry-capable key tap driver."""

    backend: KeyboardBackend | None = None
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = _default_backend()
        if self.logger is None:
            self.logger = LOGGER

    def tap_key(
        self,
        key_code: int,
        *,
        retries: int = 3,
        retry_delay_seconds: float = 0.05,
        press_duration_seconds: float = 0.05,
        verification_hook: VerificationHook | None = None,
    ) -> None:
        """Tap a key with retries when dispatch/verification fails."""
        if retries < 1:
            raise ValueError("retries must be >= 1")

        last_error: InputDriverError | None = None
        for attempt in range(1, retries + 1):
            timestamp = datetime.now(UTC).isoformat()
            assert self.logger is not None  # for type checkers
            assert self.backend is not None
            self.logger.info(
                "Input tap attempt=%s/%s key_code=%s timestamp=%s",
                attempt,
                retries,
                key_code,
                timestamp,
            )
            dispatched = self.backend.tap_virtual_key(
                key_code=key_code, press_duration_seconds=press_duration_seconds
            )
            verified = True if verification_hook is None else verification_hook(key_code, attempt)
            if dispatched and verified:
                return

            reason = "dispatch failed" if not dispatched else "verification failed"
            last_error = InputDriverError(
                f"Key tap considered dropped ({reason}) key_code={key_code} attempt={attempt}/{retries}."
            )
            self.logger.warning("%s", last_error)
            if attempt < retries:
                time.sleep(retry_delay_seconds)

        if last_error is None:
            raise InputDriverError("Unknown key tap failure.")
        raise last_error

    def sleep_wait(self, duration_seconds: float) -> None:
        """Pause deterministically as part of action execution."""
        timestamp = datetime.now(UTC).isoformat()
        assert self.logger is not None
        self.logger.info("Input wait duration=%s timestamp=%s", duration_seconds, timestamp)
        time.sleep(duration_seconds)

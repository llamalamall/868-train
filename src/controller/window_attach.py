"""Window discovery, attach, and focus utilities."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

LOGGER = logging.getLogger(__name__)
SW_RESTORE = 9


class WindowAttachError(RuntimeError):
    """Raised when window discovery/attach/focus operations fail."""


@dataclass(frozen=True)
class AttachedWindow:
    """Attached window details."""

    hwnd: int
    pid: int
    title: str


class WindowBackend(Protocol):
    """Backend contract for window operations."""

    def find_main_window_for_pid(self, pid: int) -> int | None:
        """Return hwnd for a process main window."""

    def get_window_title(self, hwnd: int) -> str:
        """Return window title for hwnd."""

    def show_window(self, hwnd: int) -> None:
        """Restore/show the window."""

    def set_foreground_window(self, hwnd: int) -> bool:
        """Bring a window to foreground."""

    def force_focus_window(self, hwnd: int) -> bool:
        """Try a more aggressive foreground handoff for hwnd."""

    def get_foreground_window(self) -> int | None:
        """Get active foreground window."""

    def is_window(self, hwnd: int) -> bool:
        """True if hwnd is still valid."""


class WindowsWindowBackend:
    """Windows API backend for window discovery and foreground focus."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise WindowAttachError("WindowsWindowBackend is only supported on Windows.")

        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._user32.BringWindowToTop.argtypes = (ctypes.wintypes.HWND,)
        self._user32.BringWindowToTop.restype = ctypes.wintypes.BOOL
        self._user32.SetFocus.argtypes = (ctypes.wintypes.HWND,)
        self._user32.SetFocus.restype = ctypes.wintypes.HWND
        self._user32.SetActiveWindow.argtypes = (ctypes.wintypes.HWND,)
        self._user32.SetActiveWindow.restype = ctypes.wintypes.HWND
        self._user32.GetWindowThreadProcessId.argtypes = (
            ctypes.wintypes.HWND,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        )
        self._user32.GetWindowThreadProcessId.restype = ctypes.wintypes.DWORD
        self._user32.AttachThreadInput.argtypes = (
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
        )
        self._user32.AttachThreadInput.restype = ctypes.wintypes.BOOL
        self._kernel32.GetCurrentThreadId.argtypes = ()
        self._kernel32.GetCurrentThreadId.restype = ctypes.wintypes.DWORD

    def find_main_window_for_pid(self, pid: int) -> int | None:
        found_hwnd: int | None = None

        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_windows_proc(hwnd: int, _lparam: int) -> bool:
            nonlocal found_hwnd
            if not self._user32.IsWindowVisible(hwnd):
                return True
            if self._user32.GetWindow(hwnd, 4):  # GW_OWNER = 4
                return True

            process_id = ctypes.wintypes.DWORD()
            self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if int(process_id.value) == pid:
                found_hwnd = int(hwnd)
                return False
            return True

        self._user32.EnumWindows(enum_windows_proc, 0)
        return found_hwnd

    def get_window_title(self, hwnd: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buffer, length + 1)
        return str(buffer.value)

    def show_window(self, hwnd: int) -> None:
        self._user32.ShowWindow(hwnd, SW_RESTORE)

    def set_foreground_window(self, hwnd: int) -> bool:
        return bool(self._user32.SetForegroundWindow(hwnd))

    def force_focus_window(self, hwnd: int) -> bool:
        foreground = self.get_foreground_window()
        target_thread = int(self._user32.GetWindowThreadProcessId(hwnd, None))
        foreground_thread = (
            int(self._user32.GetWindowThreadProcessId(foreground, None))
            if foreground is not None
            else 0
        )
        current_thread = int(self._kernel32.GetCurrentThreadId())
        attached_pairs: list[tuple[int, int]] = []

        def _attach(a: int, b: int) -> None:
            if a <= 0 or b <= 0 or a == b:
                return
            if bool(self._user32.AttachThreadInput(a, b, True)):
                attached_pairs.append((a, b))

        try:
            _attach(current_thread, target_thread)
            _attach(current_thread, foreground_thread)
            _attach(target_thread, foreground_thread)
            self.show_window(hwnd)
            self._user32.BringWindowToTop(hwnd)
            self._user32.SetActiveWindow(hwnd)
            self._user32.SetFocus(hwnd)
            self._user32.SetForegroundWindow(hwnd)
            return self.get_foreground_window() == hwnd
        finally:
            for a, b in reversed(attached_pairs):
                self._user32.AttachThreadInput(a, b, False)

    def get_foreground_window(self) -> int | None:
        hwnd = int(self._user32.GetForegroundWindow())
        if hwnd == 0:
            return None
        return hwnd

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))


def _default_backend() -> WindowBackend:
    if os.name != "nt":
        raise WindowAttachError("Window attach is currently implemented for Windows only.")
    return WindowsWindowBackend()


def attach_window(
    *,
    pid: int,
    retries: int = 3,
    retry_delay_seconds: float = 0.5,
    backend: WindowBackend | None = None,
    logger: logging.Logger | None = None,
) -> AttachedWindow:
    """Attach to a process window by PID with retry/backoff."""
    if retries < 1:
        raise ValueError("retries must be >= 1")

    active_backend = backend or _default_backend()
    active_logger = logger or LOGGER
    last_error: WindowAttachError | None = None

    for attempt in range(1, retries + 1):
        hwnd = active_backend.find_main_window_for_pid(pid)
        if hwnd is not None:
            title = active_backend.get_window_title(hwnd)
            attached = AttachedWindow(hwnd=hwnd, pid=pid, title=title)
            active_logger.info("Attached to window hwnd=%s pid=%s title=%r", hwnd, pid, title)
            return attached

        last_error = WindowAttachError(f"No window found for pid={pid}.")
        active_logger.warning("Window attach attempt %s/%s failed: %s", attempt, retries, last_error)
        if attempt < retries:
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise WindowAttachError("Unknown window attach failure.")
    raise last_error


def focus_window(
    window: AttachedWindow,
    *,
    retries: int = 3,
    retry_delay_seconds: float = 0.2,
    backend: WindowBackend | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Bring attached window to foreground with retry/backoff."""
    if retries < 1:
        raise ValueError("retries must be >= 1")

    active_backend = backend or _default_backend()
    active_logger = logger or LOGGER
    last_error: WindowAttachError | None = None

    for attempt in range(1, retries + 1):
        if not active_backend.is_window(window.hwnd):
            raise WindowAttachError(f"Window handle is no longer valid: hwnd={window.hwnd}")

        active_backend.show_window(window.hwnd)
        focused = active_backend.set_foreground_window(window.hwnd)
        foreground = active_backend.get_foreground_window()
        if foreground != window.hwnd:
            focused = bool(active_backend.force_focus_window(window.hwnd)) or focused
            foreground = active_backend.get_foreground_window()
        if focused and foreground == window.hwnd:
            active_logger.info("Focused window hwnd=%s pid=%s", window.hwnd, window.pid)
            return

        last_error = WindowAttachError(
            f"Unable to focus window hwnd={window.hwnd}. "
            f"SetForegroundWindow success={focused}, foreground={foreground}."
        )
        active_logger.warning("Window focus attempt %s/%s failed: %s", attempt, retries, last_error)
        if attempt < retries:
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise WindowAttachError("Unknown window focus failure.")
    raise last_error


def wait_for_window_foreground(
    window: AttachedWindow,
    *,
    poll_interval_seconds: float = 0.10,
    backend: WindowBackend | None = None,
    logger: logging.Logger | None = None,
    sleep_fn: callable = time.sleep,
) -> None:
    """Block until the target window regains foreground focus."""
    active_backend = backend or _default_backend()
    active_logger = logger or LOGGER
    warned = False
    poll_interval = max(float(poll_interval_seconds), 0.0)

    while True:
        if not active_backend.is_window(window.hwnd):
            raise WindowAttachError(f"Window handle is no longer valid: hwnd={window.hwnd}")
        foreground = active_backend.get_foreground_window()
        if foreground == window.hwnd:
            if warned:
                active_logger.info("Window foreground restored hwnd=%s pid=%s", window.hwnd, window.pid)
            return
        if not warned:
            active_logger.warning(
                "Window foreground lost; pausing input until game regains focus "
                "hwnd=%s pid=%s foreground=%s",
                window.hwnd,
                window.pid,
                foreground,
            )
            warned = True
        if poll_interval > 0:
            sleep_fn(poll_interval)

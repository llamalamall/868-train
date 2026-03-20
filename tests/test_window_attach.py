"""Tests for window attach/focus behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.controller.window_attach import (
    AttachedWindow,
    WindowAttachError,
    attach_window,
    focus_window,
    wait_for_window_foreground,
)


@dataclass
class FakeWindowBackend:
    """Simple fake backend for window attach/focus tests."""

    find_results: list[int | None] = field(default_factory=list)
    title_by_hwnd: dict[int, str] = field(default_factory=dict)
    valid_windows: set[int] = field(default_factory=set)
    foreground: int | None = None
    set_foreground_success: bool = True
    force_focus_success: bool = False
    foreground_sequence: list[int | None] = field(default_factory=list)
    show_calls: list[int] = field(default_factory=list)
    set_foreground_calls: list[int] = field(default_factory=list)
    force_focus_calls: list[int] = field(default_factory=list)

    def find_main_window_for_pid(self, pid: int) -> int | None:
        if self.find_results:
            return self.find_results.pop(0)
        return None

    def get_window_title(self, hwnd: int) -> str:
        return self.title_by_hwnd.get(hwnd, "")

    def show_window(self, hwnd: int) -> None:
        self.show_calls.append(hwnd)

    def set_foreground_window(self, hwnd: int) -> bool:
        self.set_foreground_calls.append(hwnd)
        if self.set_foreground_success:
            self.foreground = hwnd
            return True
        return False

    def force_focus_window(self, hwnd: int) -> bool:
        self.force_focus_calls.append(hwnd)
        if self.force_focus_success:
            self.foreground = hwnd
            return True
        return False

    def get_foreground_window(self) -> int | None:
        if self.foreground_sequence:
            self.foreground = self.foreground_sequence.pop(0)
        return self.foreground

    def is_window(self, hwnd: int) -> bool:
        return hwnd in self.valid_windows


def test_attach_window_success() -> None:
    backend = FakeWindowBackend(
        find_results=[1234],
        title_by_hwnd={1234: "868-HACK"},
    )
    attached = attach_window(pid=99, backend=backend, retries=1)
    assert attached == AttachedWindow(hwnd=1234, pid=99, title="868-HACK")


def test_attach_window_retries_then_fails() -> None:
    backend = FakeWindowBackend(find_results=[None, None])
    with pytest.raises(WindowAttachError, match="No window found"):
        attach_window(pid=10, backend=backend, retries=2, retry_delay_seconds=0.0)


def test_focus_window_success() -> None:
    backend = FakeWindowBackend(valid_windows={1234}, foreground=None, set_foreground_success=True)
    window = AttachedWindow(hwnd=1234, pid=99, title="868-HACK")
    focus_window(window, backend=backend, retries=1)
    assert backend.show_calls == [1234]
    assert backend.set_foreground_calls == [1234]
    assert backend.force_focus_calls == []
    assert backend.foreground == 1234


def test_focus_window_fails_when_invalid() -> None:
    backend = FakeWindowBackend(valid_windows=set())
    window = AttachedWindow(hwnd=555, pid=99, title="dead")
    with pytest.raises(WindowAttachError, match="no longer valid"):
        focus_window(window, backend=backend, retries=1)


def test_focus_window_retries_then_fails() -> None:
    backend = FakeWindowBackend(valid_windows={777}, foreground=888, set_foreground_success=False)
    window = AttachedWindow(hwnd=777, pid=99, title="868-HACK")
    with pytest.raises(WindowAttachError, match="Unable to focus window"):
        focus_window(window, backend=backend, retries=2, retry_delay_seconds=0.0)


def test_focus_window_uses_force_focus_fallback() -> None:
    backend = FakeWindowBackend(
        valid_windows={444},
        foreground=111,
        set_foreground_success=False,
        force_focus_success=True,
    )
    window = AttachedWindow(hwnd=444, pid=99, title="868-HACK")

    focus_window(window, backend=backend, retries=1)

    assert backend.show_calls == [444]
    assert backend.set_foreground_calls == [444]
    assert backend.force_focus_calls == [444]
    assert backend.foreground == 444


def test_wait_for_window_foreground_waits_until_target_is_active() -> None:
    backend = FakeWindowBackend(
        valid_windows={333},
        foreground_sequence=[111, 222, 333],
    )
    window = AttachedWindow(hwnd=333, pid=99, title="868-HACK")
    sleeps: list[float] = []

    wait_for_window_foreground(
        window,
        backend=backend,
        poll_interval_seconds=0.05,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [0.05, 0.05]


def test_wait_for_window_foreground_fails_when_window_becomes_invalid() -> None:
    backend = FakeWindowBackend(
        valid_windows=set(),
        foreground=111,
    )
    window = AttachedWindow(hwnd=333, pid=99, title="868-HACK")

    with pytest.raises(WindowAttachError, match="no longer valid"):
        wait_for_window_foreground(window, backend=backend, poll_interval_seconds=0.0)

"""Tests for window attach/focus behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.controller.window_attach import (
    AttachedWindow,
    WindowAttachError,
    attach_window,
    focus_window,
)


@dataclass
class FakeWindowBackend:
    """Simple fake backend for window attach/focus tests."""

    find_results: list[int | None] = field(default_factory=list)
    title_by_hwnd: dict[int, str] = field(default_factory=dict)
    valid_windows: set[int] = field(default_factory=set)
    foreground: int | None = None
    set_foreground_success: bool = True
    show_calls: list[int] = field(default_factory=list)
    set_foreground_calls: list[int] = field(default_factory=list)

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

    def get_foreground_window(self) -> int | None:
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

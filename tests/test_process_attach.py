"""Tests for process attach behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.memory.process_attach import ProcessAttachError, attach_process, close_attached_process


@dataclass
class FakeProcessBackend:
    """Simple fake backend for process attach tests."""

    by_name: dict[str, int] = field(default_factory=dict)
    by_pid: dict[int, str] = field(default_factory=dict)
    openable_pids: set[int] = field(default_factory=set)
    close_calls: list[int] = field(default_factory=list)

    def find_pid_by_executable(self, executable_name: str) -> int | None:
        return self.by_name.get(executable_name.lower())

    def get_executable_name(self, pid: int) -> str | None:
        return self.by_pid.get(pid)

    def open_process(self, pid: int) -> int | None:
        if pid in self.openable_pids:
            return pid + 1000
        return None

    def close_handle(self, handle: int) -> None:
        self.close_calls.append(handle)


def test_attach_process_by_pid() -> None:
    backend = FakeProcessBackend(
        by_pid={77: "868-HACK.exe"},
        openable_pids={77},
    )
    attached = attach_process(pid=77, backend=backend, retries=1)
    assert attached.pid == 77
    assert attached.executable_name == "868-HACK.exe"
    assert attached.handle == 1077


def test_attach_process_by_executable_name_is_case_insensitive() -> None:
    backend = FakeProcessBackend(
        by_name={"868-hack.exe": 42},
        by_pid={42: "868-HACK.exe"},
        openable_pids={42},
    )
    attached = attach_process(executable_name="868-hack.exe", backend=backend, retries=1)
    assert attached.pid == 42
    assert attached.executable_name == "868-hack.exe"


def test_attach_process_retries_then_fails() -> None:
    backend = FakeProcessBackend(by_name={}, by_pid={})
    with pytest.raises(ProcessAttachError, match="Process not found"):
        attach_process(executable_name="missing.exe", backend=backend, retries=2, retry_delay_seconds=0.0)


def test_close_attached_process_calls_backend() -> None:
    backend = FakeProcessBackend()
    attached = attach_process(
        pid=7,
        backend=FakeProcessBackend(by_pid={7: "868-HACK.exe"}, openable_pids={7}),
        retries=1,
    )
    close_attached_process(attached, backend=backend)
    assert backend.close_calls == [attached.handle]


def test_attach_process_requires_pid_or_name() -> None:
    backend = FakeProcessBackend()
    with pytest.raises(ProcessAttachError, match="either pid or executable_name"):
        attach_process(backend=backend, retries=1)

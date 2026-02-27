"""Process discovery and attach utilities."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

LOGGER = logging.getLogger(__name__)

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_ATTACH_ACCESS = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ

TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class ProcessAttachError(RuntimeError):
    """Raised when process discovery or process attach fails."""


@dataclass(frozen=True)
class AttachedProcess:
    """Attached process details."""

    pid: int
    executable_name: str
    handle: int


class ProcessBackend(Protocol):
    """Backend contract for process lookup/open operations."""

    def find_pid_by_executable(self, executable_name: str) -> int | None:
        """Resolve PID by executable name."""

    def get_executable_name(self, pid: int) -> str | None:
        """Get executable name for a PID."""

    def open_process(self, pid: int) -> int | None:
        """Open process and return native handle."""

    def close_handle(self, handle: int) -> None:
        """Close a previously-opened native handle."""


class PROCESSENTRY32W(ctypes.Structure):
    """ctypes mapping for PROCESSENTRY32W."""

    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("cntUsage", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("cntThreads", ctypes.wintypes.DWORD),
        ("th32ParentProcessID", ctypes.wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class WindowsProcessBackend:
    """Windows API backend for process discovery and process handle attachment."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise ProcessAttachError("WindowsProcessBackend is only supported on Windows.")

        self._kernel32 = ctypes.windll.kernel32

    def _iter_processes(self) -> list[tuple[int, str]]:
        snapshot = self._kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            error_code = ctypes.GetLastError()
            raise ProcessAttachError(
                f"Failed to list running processes (CreateToolhelp32Snapshot error={error_code})."
            )

        process_entry = PROCESSENTRY32W()
        process_entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        results: list[tuple[int, str]] = []

        try:
            has_entry = bool(self._kernel32.Process32FirstW(snapshot, ctypes.byref(process_entry)))
            while has_entry:
                results.append((int(process_entry.th32ProcessID), str(process_entry.szExeFile)))
                has_entry = bool(self._kernel32.Process32NextW(snapshot, ctypes.byref(process_entry)))
        finally:
            self._kernel32.CloseHandle(snapshot)

        return results

    def find_pid_by_executable(self, executable_name: str) -> int | None:
        target = executable_name.lower()
        for pid, exe_name in self._iter_processes():
            if exe_name.lower() == target:
                return pid
        return None

    def get_executable_name(self, pid: int) -> str | None:
        for current_pid, exe_name in self._iter_processes():
            if current_pid == pid:
                return exe_name
        return None

    def open_process(self, pid: int) -> int | None:
        handle = self._kernel32.OpenProcess(PROCESS_ATTACH_ACCESS, False, pid)
        if not handle:
            return None
        return int(handle)

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


def _default_backend() -> ProcessBackend:
    if os.name != "nt":
        raise ProcessAttachError("Process attach is currently implemented for Windows only.")
    return WindowsProcessBackend()


def _attach_once(
    *,
    backend: ProcessBackend,
    pid: int | None,
    executable_name: str | None,
) -> AttachedProcess:
    if pid is None and executable_name is None:
        raise ProcessAttachError("attach_process requires either pid or executable_name.")

    resolved_pid = pid
    resolved_name = executable_name

    if resolved_pid is None and resolved_name is not None:
        resolved_pid = backend.find_pid_by_executable(resolved_name)
        if resolved_pid is None:
            raise ProcessAttachError(f"Process not found for executable '{resolved_name}'.")

    if resolved_pid is None:
        raise ProcessAttachError("Unable to resolve target process PID.")

    if resolved_name is None:
        resolved_name = backend.get_executable_name(resolved_pid)
        if resolved_name is None:
            raise ProcessAttachError(f"PID {resolved_pid} is not running.")

    handle = backend.open_process(resolved_pid)
    if handle is None:
        raise ProcessAttachError(
            f"Failed to attach to PID {resolved_pid} ({resolved_name}). "
            "The process may have exited or permissions are insufficient."
        )

    return AttachedProcess(pid=resolved_pid, executable_name=resolved_name, handle=handle)


def attach_process(
    *,
    pid: int | None = None,
    executable_name: str | None = None,
    retries: int = 3,
    retry_delay_seconds: float = 0.5,
    backend: ProcessBackend | None = None,
    logger: logging.Logger | None = None,
) -> AttachedProcess:
    """Attach to a running process by PID or executable name with retry/backoff."""
    if retries < 1:
        raise ValueError("retries must be >= 1")

    active_logger = logger or LOGGER
    active_backend = backend or _default_backend()
    last_error: ProcessAttachError | None = None

    for attempt in range(1, retries + 1):
        try:
            attached = _attach_once(
                backend=active_backend,
                pid=pid,
                executable_name=executable_name,
            )
            active_logger.info(
                "Attached to process pid=%s executable=%s",
                attached.pid,
                attached.executable_name,
            )
            return attached
        except ProcessAttachError as error:
            last_error = error
            active_logger.warning("Process attach attempt %s/%s failed: %s", attempt, retries, error)
            if attempt < retries:
                time.sleep(retry_delay_seconds)

    if last_error is None:
        raise ProcessAttachError("Unknown process attach failure.")
    raise last_error


def close_attached_process(process: AttachedProcess, backend: ProcessBackend | None = None) -> None:
    """Close native process handle for an attached process."""
    active_backend = backend or _default_backend()
    active_backend.close_handle(process.handle)

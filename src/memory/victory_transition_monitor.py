"""Debugger-style monitor for victory-screen transition breakpoints.

This monitor attaches to the live game process with the Windows debug API and
uses hardware breakpoints for the UI-level victory latches:
- the normal-victory branch that sets `DAT_14006b8d9 = 1`
- the alternate `868 points` branch that sets `DAT_14006b8de = 1`

Each hit snapshots key game-state fields before the normal transition tears the
runtime object graph down.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.offsets import OffsetEntry, load_offset_registry
from src.memory.process_attach import (
    AttachedProcess,
    ProcessAttachError,
    attach_process,
    close_attached_process,
)
from src.memory.reader import ReadFailure, ReadResult, ProcessMemoryReader

LOGGER = logging.getLogger(__name__)

PREFERRED_IMAGE_BASE = 0x140000000
ROOT_POINTER_MODULE_OFFSET = 0x115808
FIELD_OFFSET_VICTORY_ACTIVE = 0x1B96
FIELD_OFFSET_VICTORY_PENDING = 0x1B98
NORMAL_VICTORY_FLAG_SET_PREFERRED_VA = 0x14000AADE
POINTS_VICTORY_FLAG_SET_PREFERRED_VA = 0x140040734
DEFAULT_OUTPUT_PATH = Path("artifacts") / "debug" / "victory-transition-events.jsonl"
DEFAULT_WAIT_TIMEOUT_MS = 100

EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
UNLOAD_DLL_DEBUG_EVENT = 7

EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004

DBG_CONTINUE = 0x00010002
TRAP_FLAG = 0x100

CONTEXT_AMD64 = 0x00100000
CONTEXT_DEBUG_REGISTERS = CONTEXT_AMD64 | 0x00000010


@dataclass(frozen=True)
class DebugBreakpointTarget:
    """Breakpoint target definition."""

    name: str
    slot: int
    kind: str
    access: str
    field_offset: int | None = None
    preferred_va: int | None = None


@dataclass(frozen=True)
class SnapshotFieldValue:
    """Serialized snapshot field value."""

    status: str
    value: Any | None = None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class WatchAddressSet:
    """Resolved runtime addresses for active watchpoints."""

    normal_victory_flag_set: int | None
    points_victory_flag_set: int | None


@dataclass(frozen=True)
class DebugRegisterWatch:
    """Hardware breakpoint register assignment."""

    slot: int
    address: int
    access: str
    size: int = 1


class VictoryTransitionMonitorError(RuntimeError):
    """Raised when debugger-style transition monitoring cannot proceed."""


class EXCEPTION_RECORD(ctypes.Structure):
    _fields_ = [
        ("ExceptionCode", ctypes.wintypes.DWORD),
        ("ExceptionFlags", ctypes.wintypes.DWORD),
        ("ExceptionRecord", ctypes.c_void_p),
        ("ExceptionAddress", ctypes.c_void_p),
        ("NumberParameters", ctypes.wintypes.DWORD),
        ("ExceptionInformation", ctypes.c_ulonglong * 15),
    ]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("ExceptionRecord", EXCEPTION_RECORD),
        ("dwFirstChance", ctypes.wintypes.DWORD),
    ]


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hThread", ctypes.wintypes.HANDLE),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
    ]


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", ctypes.wintypes.HANDLE),
        ("hProcess", ctypes.wintypes.HANDLE),
        ("hThread", ctypes.wintypes.HANDLE),
        ("lpBaseOfImage", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", ctypes.wintypes.DWORD),
        ("nDebugInfoSize", ctypes.wintypes.DWORD),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", ctypes.wintypes.WORD),
    ]


class EXIT_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", ctypes.wintypes.DWORD)]


class EXIT_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", ctypes.wintypes.DWORD)]


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", ctypes.wintypes.HANDLE),
        ("lpBaseOfDll", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", ctypes.wintypes.DWORD),
        ("nDebugInfoSize", ctypes.wintypes.DWORD),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", ctypes.wintypes.WORD),
    ]


class UNLOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("lpBaseOfDll", ctypes.c_void_p)]


class DEBUG_EVENT_UNION(ctypes.Union):
    _fields_ = [
        ("Exception", EXCEPTION_DEBUG_INFO),
        ("CreateThread", CREATE_THREAD_DEBUG_INFO),
        ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO),
        ("ExitThread", EXIT_THREAD_DEBUG_INFO),
        ("ExitProcess", EXIT_PROCESS_DEBUG_INFO),
        ("LoadDll", LOAD_DLL_DEBUG_INFO),
        ("UnloadDll", UNLOAD_DLL_DEBUG_INFO),
    ]


class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [
        ("dwDebugEventCode", ctypes.wintypes.DWORD),
        ("dwProcessId", ctypes.wintypes.DWORD),
        ("dwThreadId", ctypes.wintypes.DWORD),
        ("u", DEBUG_EVENT_UNION),
    ]


class DebugContext(ctypes.Structure):
    """Partial x64 CONTEXT sufficient for debug registers."""

    _fields_ = [
        ("P1Home", ctypes.c_ulonglong),
        ("P2Home", ctypes.c_ulonglong),
        ("P3Home", ctypes.c_ulonglong),
        ("P4Home", ctypes.c_ulonglong),
        ("P5Home", ctypes.c_ulonglong),
        ("P6Home", ctypes.c_ulonglong),
        ("ContextFlags", ctypes.wintypes.DWORD),
        ("MxCsr", ctypes.wintypes.DWORD),
        ("SegCs", ctypes.c_ushort),
        ("SegDs", ctypes.c_ushort),
        ("SegEs", ctypes.c_ushort),
        ("SegFs", ctypes.c_ushort),
        ("SegGs", ctypes.c_ushort),
        ("SegSs", ctypes.c_ushort),
        ("EFlags", ctypes.wintypes.DWORD),
        ("Dr0", ctypes.c_ulonglong),
        ("Dr1", ctypes.c_ulonglong),
        ("Dr2", ctypes.c_ulonglong),
        ("Dr3", ctypes.c_ulonglong),
        ("Dr6", ctypes.c_ulonglong),
        ("Dr7", ctypes.c_ulonglong),
    ]


def _default_targets() -> tuple[DebugBreakpointTarget, ...]:
    return (
        DebugBreakpointTarget(
            name="normal_victory_flag_set",
            slot=0,
            kind="code_execute",
            access="execute",
            preferred_va=NORMAL_VICTORY_FLAG_SET_PREFERRED_VA,
        ),
        DebugBreakpointTarget(
            name="points_victory_flag_set",
            slot=1,
            kind="code_execute",
            access="execute",
            preferred_va=POINTS_VICTORY_FLAG_SET_PREFERRED_VA,
        ),
    )


def _runtime_address_from_preferred(*, module_base: int, preferred_va: int) -> int:
    return int(module_base) + (int(preferred_va) - PREFERRED_IMAGE_BASE)


def _build_watch_addresses(*, module_base: int | None, root_address: int | None) -> WatchAddressSet:
    del root_address
    normal_victory_flag_set = (
        _runtime_address_from_preferred(
            module_base=int(module_base),
            preferred_va=NORMAL_VICTORY_FLAG_SET_PREFERRED_VA,
        )
        if module_base is not None
        else None
    )
    points_victory_flag_set = (
        _runtime_address_from_preferred(
            module_base=int(module_base),
            preferred_va=POINTS_VICTORY_FLAG_SET_PREFERRED_VA,
        )
        if module_base is not None
        else None
    )
    return WatchAddressSet(
        normal_victory_flag_set=normal_victory_flag_set,
        points_victory_flag_set=points_victory_flag_set,
    )


def _build_debug_register_watches(addresses: WatchAddressSet) -> tuple[DebugRegisterWatch, ...]:
    watches: list[DebugRegisterWatch] = []
    if addresses.normal_victory_flag_set is not None:
        watches.append(DebugRegisterWatch(slot=0, address=addresses.normal_victory_flag_set, access="execute"))
    if addresses.points_victory_flag_set is not None:
        watches.append(DebugRegisterWatch(slot=1, address=addresses.points_victory_flag_set, access="execute"))
    return tuple(watches)


def _encode_breakpoint_length(size: int) -> int:
    if size == 1:
        return 0b00
    if size == 2:
        return 0b01
    if size == 4:
        return 0b11
    if size == 8:
        return 0b10
    raise ValueError(f"Unsupported debug-register size: {size}")


def _encode_breakpoint_access(access: str) -> int:
    normalized = str(access).strip().lower()
    if normalized == "execute":
        return 0b00
    if normalized == "write":
        return 0b01
    if normalized == "readwrite":
        return 0b11
    raise ValueError(f"Unsupported debug-register access type: {access}")


def _build_dr7(watches: tuple[DebugRegisterWatch, ...]) -> int:
    value = 0
    for watch in watches:
        if watch.slot < 0 or watch.slot > 3:
            raise ValueError(f"Hardware breakpoint slot must be 0..3, got {watch.slot}.")
        value |= 1 << (watch.slot * 2)
        control_shift = 16 + (watch.slot * 4)
        value |= _encode_breakpoint_access(watch.access) << control_shift
        value |= _encode_breakpoint_length(watch.size) << (control_shift + 2)
    return value


def _triggered_slots(dr6: int) -> tuple[int, ...]:
    return tuple(slot for slot in range(4) if dr6 & (1 << slot))


def _clear_watch_enable_bits(dr7: int, slot: int) -> int:
    shift = slot * 2
    return int(dr7) & ~(0b11 << shift)


def _module_base_resolver(module_name: str, module_base: int | None) -> ReadResult[int]:
    if module_base is None:
        return ReadResult.fail(
            ReadFailure(
                code="module_base_unavailable",
                message=f"Module base for '{module_name}' is not available yet.",
            )
        )
    return ReadResult.ok(int(module_base))


def _read_entry_value(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry,
    address: int,
) -> SnapshotFieldValue:
    normalized_type = entry.data_type.strip().lower()
    if normalized_type == "bool":
        result = reader.read_bool(address)
    elif normalized_type == "int32":
        result = reader.read_int32(address)
    else:
        return SnapshotFieldValue(
            status="unsupported",
            error_code="unsupported_data_type",
            message=f"Unsupported snapshot data_type '{entry.data_type}'.",
        )
    if not result.is_ok:
        failure = result.error or ReadFailure(code="read_failed", message="Unknown read failure.")
        return SnapshotFieldValue(
            status="error",
            error_code=failure.code,
            message=failure.message,
        )
    return SnapshotFieldValue(status="ok", value=result.value)


def _load_snapshot_entries() -> dict[str, OffsetEntry]:
    wanted = {
        "current_sector",
        "sector_progression_index",
        "player_x",
        "player_y",
        "run_active",
        "victory_active",
        "victory_pending",
    }
    registry = load_offset_registry()
    return {entry.name: entry for entry in registry.entries if entry.name in wanted}


def _read_snapshot_fields(
    *,
    reader: ProcessMemoryReader,
    entries: dict[str, OffsetEntry],
    module_base: int | None,
) -> dict[str, dict[str, Any]]:
    from src.memory.pointer_chain import resolve_offset_entry_address

    payload: dict[str, dict[str, Any]] = {}
    for field_name in (
        "current_sector",
        "sector_progression_index",
        "player_x",
        "player_y",
        "run_active",
        "victory_pending",
        "victory_active",
    ):
        entry = entries.get(field_name)
        if entry is None:
            payload[field_name] = {
                "status": "missing",
                "error_code": "entry_missing",
                "message": f"Offset registry entry '{field_name}' is not defined.",
            }
            continue
        resolved = resolve_offset_entry_address(
            reader=reader,
            entry=entry,
            module_base_resolver=lambda module_name: _module_base_resolver(module_name, module_base),
        )
        if not resolved.is_ok or resolved.value is None:
            error = resolved.error
            payload[field_name] = {
                "status": "error",
                "error_code": error.code if error is not None else "resolve_failed",
                "message": error.message if error is not None else "Failed resolving entry address.",
            }
            continue
        field_value = _read_entry_value(reader=reader, entry=entry, address=resolved.value)
        payload[field_name] = {
            "status": field_value.status,
            "value": field_value.value,
            "error_code": field_value.error_code,
            "message": field_value.message,
            "address": f"0x{resolved.value:X}",
        }
    return payload


def _snapshot_field_bool(snapshot: dict[str, dict[str, Any]], field_name: str) -> bool | None:
    field = snapshot.get(field_name)
    if not isinstance(field, dict) or field.get("status") != "ok":
        return None
    value = field.get("value")
    if isinstance(value, bool):
        return value
    try:
        if value is None:
            return None
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attach as a debugger and capture state snapshots when the game executes "
            "the normal-victory or special-points victory-flag set instructions."
        )
    )
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument("--pid", type=int, default=None, help="Attach to a specific PID instead of resolving by name.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL file for captured breakpoint events.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum number of breakpoint hits to record (0 = unlimited).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Optional overall timeout in seconds (0 = unlimited).",
    )
    parser.add_argument(
        "--launch-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch the executable if it is not already running.",
    )
    parser.add_argument(
        "--wait-timeout-ms",
        type=int,
        default=DEFAULT_WAIT_TIMEOUT_MS,
        help="WaitForDebugEvent timeout in milliseconds before polling for root changes again.",
    )
    return parser


class VictoryTransitionMonitor:
    """Debugger-backed transition monitor."""

    def __init__(
        self,
        *,
        attached: AttachedProcess,
        exe_name: str,
        output_path: Path,
        max_events: int,
        timeout_seconds: float,
        wait_timeout_ms: int,
        logger: logging.Logger | None = None,
    ) -> None:
        if wait_timeout_ms < 1:
            raise ValueError("wait_timeout_ms must be >= 1.")
        self._attached = attached
        self._exe_name = exe_name
        self._output_path = output_path
        self._max_events = int(max_events)
        self._timeout_seconds = float(timeout_seconds)
        self._wait_timeout_ms = int(wait_timeout_ms)
        self._logger = logger or LOGGER
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._reader = ProcessMemoryReader(process_handle=attached.handle)
        self._snapshot_entries = _load_snapshot_entries()
        self._targets = _default_targets()
        self._thread_handles: dict[int, int] = {}
        self._module_base: int | None = None
        self._current_root: int | None = None
        self._last_applied_module_base: int | None = None
        self._step_over_threads: set[int] = set()
        self._event_count = 0
        self._debug_attached = False
        self._detached = False
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        started = dt.datetime.now(dt.timezone.utc)
        self._attach_debugger()
        try:
            with self._output_path.open("a", encoding="ascii") as handle:
                while True:
                    if self._max_events > 0 and self._event_count >= self._max_events:
                        break
                    if self._timeout_seconds > 0:
                        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
                        if elapsed >= self._timeout_seconds:
                            break
                    event = DEBUG_EVENT()
                    has_event = bool(
                        self._kernel32.WaitForDebugEvent(ctypes.byref(event), self._wait_timeout_ms)
                    )
                    if not has_event:
                        error_code = ctypes.get_last_error()
                        if error_code == 121:
                            self._refresh_root_and_breakpoints()
                            continue
                        raise VictoryTransitionMonitorError(
                            f"WaitForDebugEvent failed (error={error_code})."
                        )
                    continue_status = self._handle_debug_event(event=event, output_handle=handle)
                    self._kernel32.ContinueDebugEvent(
                        int(event.dwProcessId),
                        int(event.dwThreadId),
                        continue_status,
                    )
                    if int(event.dwDebugEventCode) == EXIT_PROCESS_DEBUG_EVENT:
                        break
        finally:
            self.close()
        return self._event_count

    def close(self) -> None:
        if self._detached:
            return
        self._detached = True
        try:
            self._apply_breakpoints_to_all_threads(root_address=None)
        except VictoryTransitionMonitorError as error:  # pragma: no cover - defensive cleanup
            if "error=6" not in str(error):
                self._logger.exception("Failed clearing thread hardware breakpoints during debugger detach.")
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.exception("Failed clearing thread hardware breakpoints during debugger detach.")
        for thread_handle in tuple(self._thread_handles.values()):
            if thread_handle:
                self._kernel32.CloseHandle(ctypes.c_void_p(thread_handle))
        self._thread_handles.clear()
        if self._debug_attached:
            stop_ok = self._kernel32.DebugActiveProcessStop(int(self._attached.pid))
            if not stop_ok:
                error_code = ctypes.get_last_error()
                if error_code not in (0, 5, 87, 1168):
                    self._logger.warning(
                        "DebugActiveProcessStop failed for pid=%s (error=%s).",
                        self._attached.pid,
                        error_code,
                    )
        close_attached_process(self._attached)

    def _attach_debugger(self) -> None:
        if not self._kernel32.DebugActiveProcess(int(self._attached.pid)):
            raise VictoryTransitionMonitorError(
                f"DebugActiveProcess failed for pid={self._attached.pid} (error={ctypes.get_last_error()})."
            )
        self._debug_attached = True
        if not self._kernel32.DebugSetProcessKillOnExit(False):
            raise VictoryTransitionMonitorError(
                f"DebugSetProcessKillOnExit failed (error={ctypes.get_last_error()})."
            )

    def _handle_debug_event(self, *, event: DEBUG_EVENT, output_handle: Any) -> int:
        code = int(event.dwDebugEventCode)
        if code == CREATE_PROCESS_DEBUG_EVENT:
            self._handle_create_process(event)
            self._close_event_file_handle(int(event.u.CreateProcessInfo.hFile or 0))
            return DBG_CONTINUE
        if code == CREATE_THREAD_DEBUG_EVENT:
            self._remember_thread_handle(
                thread_id=int(event.dwThreadId),
                thread_handle=int(event.u.CreateThread.hThread or 0),
            )
            self._refresh_root_and_breakpoints(threads_already_suspended=True)
            return DBG_CONTINUE
        if code == LOAD_DLL_DEBUG_EVENT:
            self._close_event_file_handle(int(event.u.LoadDll.hFile or 0))
            return DBG_CONTINUE
        if code == EXIT_THREAD_DEBUG_EVENT:
            self._forget_thread_handle(int(event.dwThreadId))
            return DBG_CONTINUE
        if code == EXIT_PROCESS_DEBUG_EVENT:
            return DBG_CONTINUE
        if code != EXCEPTION_DEBUG_EVENT:
            return DBG_CONTINUE

        exception = event.u.Exception.ExceptionRecord
        exception_code = int(exception.ExceptionCode)
        if exception_code == EXCEPTION_BREAKPOINT:
            return DBG_CONTINUE
        if exception_code != EXCEPTION_SINGLE_STEP:
            return DBG_CONTINUE

        thread_handle = self._thread_handles.get(int(event.dwThreadId))
        if thread_handle is None:
            return DBG_CONTINUE
        context = self._get_thread_debug_context(thread_handle)
        slot_hits = _triggered_slots(int(context.Dr6))
        if not slot_hits:
            if int(event.dwThreadId) in self._step_over_threads:
                self._restore_breakpoints_after_step_over(
                    thread_id=int(event.dwThreadId),
                    thread_handle=thread_handle,
                    context=context,
                )
                return DBG_CONTINUE
            self._clear_thread_dr6(thread_handle, context)
            return DBG_CONTINUE

        context.Dr6 = 0
        execute_slot: int | None = None
        for slot in slot_hits:
            target = next((item for item in self._targets if item.slot == slot), None)
            if target is None:
                continue
            self._write_event_record(
                output_handle=output_handle,
                target=target,
                thread_id=int(event.dwThreadId),
                exception_address=int(exception.ExceptionAddress or 0),
            )
            if target.kind == "code_execute" and execute_slot is None:
                execute_slot = slot
        if execute_slot is not None:
            self._begin_step_over(
                thread_id=int(event.dwThreadId),
                thread_handle=thread_handle,
                context=context,
                slot=execute_slot,
            )
            return DBG_CONTINUE
        self._set_thread_debug_context(thread_handle, context)
        return DBG_CONTINUE

    def _handle_create_process(self, event: DEBUG_EVENT) -> None:
        process_info = event.u.CreateProcessInfo
        self._module_base = int(process_info.lpBaseOfImage or 0) or None
        self._remember_thread_handle(
            thread_id=int(event.dwThreadId),
            thread_handle=int(process_info.hThread or 0),
        )
        self._refresh_root_and_breakpoints(threads_already_suspended=True)

    def _remember_thread_handle(self, *, thread_id: int, thread_handle: int) -> None:
        if thread_handle <= 0:
            return
        existing = self._thread_handles.get(thread_id)
        if existing is not None and existing != thread_handle:
            self._kernel32.CloseHandle(ctypes.c_void_p(existing))
        self._thread_handles[thread_id] = thread_handle

    def _forget_thread_handle(self, thread_id: int) -> None:
        handle = self._thread_handles.pop(thread_id, None)
        self._step_over_threads.discard(thread_id)
        if handle:
            self._kernel32.CloseHandle(ctypes.c_void_p(handle))

    def _close_event_file_handle(self, handle: int) -> None:
        if handle > 0:
            self._kernel32.CloseHandle(ctypes.c_void_p(handle))

    def _resolve_root_address(self) -> int | None:
        if self._module_base is None:
            return None
        pointer_result = self._reader.read_pointer(int(self._module_base) + ROOT_POINTER_MODULE_OFFSET)
        if not pointer_result.is_ok or pointer_result.value in (None, 0):
            return None
        return int(pointer_result.value)

    def _refresh_root_and_breakpoints(
        self,
        *,
        force: bool = False,
        threads_already_suspended: bool = False,
    ) -> None:
        new_root = self._resolve_root_address()
        if (
            not force
            and new_root == self._current_root
            and self._module_base == self._last_applied_module_base
        ):
            return
        self._current_root = new_root
        self._last_applied_module_base = self._module_base
        self._apply_breakpoints_to_all_threads(
            root_address=self._current_root,
            threads_already_suspended=threads_already_suspended,
        )

    def _apply_breakpoints_to_all_threads(
        self,
        *,
        root_address: int | None,
        threads_already_suspended: bool = False,
    ) -> None:
        for thread_handle in tuple(self._thread_handles.values()):
            self._apply_thread_breakpoints(
                thread_handle=thread_handle,
                root_address=root_address,
                suspend_thread=not threads_already_suspended,
            )

    def _apply_thread_breakpoints(
        self,
        *,
        thread_handle: int,
        root_address: int | None,
        suspend_thread: bool,
    ) -> None:
        if thread_handle <= 0:
            return
        if suspend_thread:
            suspend_result = int(self._kernel32.SuspendThread(ctypes.c_void_p(thread_handle)))
            if suspend_result == 0xFFFFFFFF:
                raise VictoryTransitionMonitorError(
                    f"SuspendThread failed for thread handle {thread_handle} (error={ctypes.get_last_error()})."
                )
        try:
            context = self._get_thread_debug_context(thread_handle)
            addresses = _build_watch_addresses(module_base=self._module_base, root_address=root_address)
            watches = _build_debug_register_watches(addresses)
            context.Dr0 = addresses.normal_victory_flag_set or 0
            context.Dr1 = addresses.points_victory_flag_set or 0
            context.Dr2 = 0
            context.Dr3 = 0
            context.Dr6 = 0
            context.Dr7 = _build_dr7(watches)
            self._set_thread_debug_context(thread_handle, context)
        finally:
            if suspend_thread:
                resume_result = int(self._kernel32.ResumeThread(ctypes.c_void_p(thread_handle)))
                if resume_result == 0xFFFFFFFF:
                    raise VictoryTransitionMonitorError(
                        f"ResumeThread failed for thread handle {thread_handle} (error={ctypes.get_last_error()})."
                    )

    def _get_thread_debug_context(self, thread_handle: int) -> DebugContext:
        context = DebugContext()
        context.ContextFlags = CONTEXT_DEBUG_REGISTERS
        if not self._kernel32.GetThreadContext(ctypes.c_void_p(thread_handle), ctypes.byref(context)):
            raise VictoryTransitionMonitorError(
                f"GetThreadContext failed for thread handle {thread_handle} (error={ctypes.get_last_error()})."
            )
        return context

    def _set_thread_debug_context(self, thread_handle: int, context: DebugContext) -> None:
        if not self._kernel32.SetThreadContext(ctypes.c_void_p(thread_handle), ctypes.byref(context)):
            raise VictoryTransitionMonitorError(
                f"SetThreadContext failed for thread handle {thread_handle} (error={ctypes.get_last_error()})."
            )

    def _clear_thread_dr6(self, thread_handle: int, context: DebugContext) -> None:
        context.Dr6 = 0
        self._set_thread_debug_context(thread_handle, context)

    def _begin_step_over(
        self,
        *,
        thread_id: int,
        thread_handle: int,
        context: DebugContext,
        slot: int,
    ) -> None:
        context.Dr7 = _clear_watch_enable_bits(int(context.Dr7), slot)
        context.EFlags = int(context.EFlags) | TRAP_FLAG
        context.Dr6 = 0
        self._step_over_threads.add(thread_id)
        self._set_thread_debug_context(thread_handle, context)

    def _restore_breakpoints_after_step_over(
        self,
        *,
        thread_id: int,
        thread_handle: int,
        context: DebugContext,
    ) -> None:
        self._step_over_threads.discard(thread_id)
        addresses = _build_watch_addresses(module_base=self._module_base, root_address=self._current_root)
        watches = _build_debug_register_watches(addresses)
        context.Dr0 = addresses.normal_victory_flag_set or 0
        context.Dr1 = addresses.points_victory_flag_set or 0
        context.Dr2 = 0
        context.Dr3 = 0
        context.Dr6 = 0
        context.Dr7 = _build_dr7(watches)
        context.EFlags = int(context.EFlags) & ~TRAP_FLAG
        self._set_thread_debug_context(thread_handle, context)

    def _write_event_record(
        self,
        *,
        output_handle: Any,
        target: DebugBreakpointTarget,
        thread_id: int,
        exception_address: int,
    ) -> dict[str, Any]:
        self._event_count += 1
        addresses = _build_watch_addresses(module_base=self._module_base, root_address=self._current_root)
        watched_address: int | None
        if target.name == "normal_victory_flag_set":
            watched_address = addresses.normal_victory_flag_set
        else:
            watched_address = addresses.points_victory_flag_set

        record = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "event_index": self._event_count,
            "pid": int(self._attached.pid),
            "thread_id": int(thread_id),
            "target_name": target.name,
            "target_kind": target.kind,
            "watched_address": (f"0x{watched_address:X}" if watched_address is not None else None),
            "exception_address": (f"0x{exception_address:X}" if exception_address > 0 else None),
            "module_base": (f"0x{self._module_base:X}" if self._module_base is not None else None),
            "root_address": (f"0x{self._current_root:X}" if self._current_root is not None else None),
            "snapshot": _read_snapshot_fields(
                reader=self._reader,
                entries=self._snapshot_entries,
                module_base=self._module_base,
            ),
        }
        output_handle.write(json.dumps(record, sort_keys=True) + "\n")
        output_handle.flush()
        self._logger.info(
            "victory_transition_breakpoint_hit\ttarget=%s\tthread_id=%s\troot=%s",
            target.name,
            thread_id,
            record["root_address"] or "-",
        )
        return record


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_events < 0:
        parser.error("--max-events must be >= 0.")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be >= 0.")
    if args.pid is None and not str(args.exe).strip():
        parser.error("--exe is required when --pid is not provided.")

    attached: AttachedProcess | None = None
    monitor: VictoryTransitionMonitor | None = None
    try:
        attached = attach_process(
            pid=int(args.pid) if args.pid is not None else None,
            executable_name=None if args.pid is not None else str(args.exe),
            retries=3,
            retry_delay_seconds=0.5,
            launch_if_missing=bool(args.launch_if_missing),
        )
        monitor = VictoryTransitionMonitor(
            attached=attached,
            exe_name=str(args.exe),
            output_path=Path(args.output),
            max_events=int(args.max_events),
            timeout_seconds=float(args.timeout_seconds),
            wait_timeout_ms=int(args.wait_timeout_ms),
        )
        hits = monitor.run()
        print(
            "victory_transition_monitor_complete\toutput={output}\thits={hits}".format(
                output=Path(args.output),
                hits=hits,
            )
        )
    except (ProcessAttachError, VictoryTransitionMonitorError) as error:
        raise SystemExit(str(error)) from error
    finally:
        if monitor is not None:
            monitor.close()
        elif attached is not None:
            close_attached_process(attached)


__all__ = [
    "DebugBreakpointTarget",
    "DebugRegisterWatch",
    "SnapshotFieldValue",
    "VictoryTransitionMonitor",
    "VictoryTransitionMonitorError",
    "_build_debug_register_watches",
    "_build_dr7",
    "_build_parser",
    "_build_watch_addresses",
    "_default_targets",
    "_read_snapshot_fields",
    "_runtime_address_from_preferred",
    "_triggered_slots",
    "main",
]


if __name__ == "__main__":
    main()

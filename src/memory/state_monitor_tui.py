"""Interactive TUI monitor for configured memory offsets.

Usage:
    python -m src.memory.state_monitor_tui
    python -m src.memory.state_monitor_tui --fields player_energy,player_credits --interval 0.25

Controls:
    q: quit
    z: pause/resume polling
    r: refresh now
    p: toggle resolve-each-poll
    arrows / 0-9 / escape / space: pass key input through to the game window
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.controller.action_api import ActionAPI, ActionExecutionError
from src.controller.input_driver import InputDriver, InputDriverError
from src.controller.window_attach import AttachedWindow, WindowAttachError, attach_window
from src.config.offsets import OffsetEntry, load_offset_registry
from src.memory.pointer_chain import resolve_offset_entry_address
from src.memory.process_attach import AttachedProcess, attach_process, close_attached_process
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult

TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
BYTES_TYPE_REGEX = re.compile(r"^bytes\[(\d+)\]$", re.IGNORECASE)
PASSTHROUGH_KEY_MAP = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "0": "0",
    "escape": "ESCAPE",
    "space": "SPACE",
}


class StateMonitorError(RuntimeError):
    """Raised when the state monitor cannot proceed."""


class MODULEENTRY32W(ctypes.Structure):
    """ctypes mapping for MODULEENTRY32W."""

    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("GlblcntUsage", ctypes.wintypes.DWORD),
        ("ProccntUsage", ctypes.wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("modBaseSize", ctypes.wintypes.DWORD),
        ("hModule", ctypes.wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256),
        ("szExePath", ctypes.c_wchar * 260),
    ]


@dataclass(frozen=True)
class FieldSnapshot:
    """Single field snapshot shown in the TUI."""

    name: str
    data_type: str
    confidence: str
    address: str
    value: str
    status: str
    error: str


@dataclass(frozen=True)
class PollSnapshot:
    """All field snapshots for one polling tick."""

    timestamp: str
    fields: tuple[FieldSnapshot, ...]


def map_tui_key_to_passthrough_key(key: str) -> str | None:
    """Map a Textual key identifier to a configured control key name."""
    return PASSTHROUGH_KEY_MAP.get(key.strip().lower())


def _get_kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise StateMonitorError("State monitor TUI is only supported on Windows.")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _find_module_base(pid: int, module_name: str, kernel32: ctypes.WinDLL) -> int:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snapshot == INVALID_HANDLE_VALUE:
        error_code = ctypes.get_last_error()
        raise StateMonitorError(
            f"CreateToolhelp32Snapshot(module) failed for pid={pid} (error={error_code})."
        )

    module_entry = MODULEENTRY32W()
    module_entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
    wanted = module_name.lower()

    try:
        has_module = bool(kernel32.Module32FirstW(snapshot, ctypes.byref(module_entry)))
        while has_module:
            current_name = str(module_entry.szModule).lower()
            if current_name == wanted:
                base = ctypes.cast(module_entry.modBaseAddr, ctypes.c_void_p).value
                if base is None:
                    raise StateMonitorError(
                        f"Module '{module_name}' found, but base address is null."
                    )
                return int(base)
            has_module = bool(kernel32.Module32NextW(snapshot, ctypes.byref(module_entry)))
    finally:
        kernel32.CloseHandle(snapshot)

    raise StateMonitorError(f"Module '{module_name}' not found in pid={pid}.")


class MemoryStateMonitor:
    """Runtime engine that polls configured fields from target process memory."""

    def __init__(
        self,
        *,
        executable_name: str,
        config_path: Path | None,
        fields_filter: str,
        resolve_each_poll: bool,
    ) -> None:
        registry = load_offset_registry(config_path=config_path)
        self._entries = self._select_entries(registry.entries, fields_filter)
        if not self._entries:
            raise StateMonitorError("No entries selected for monitoring.")

        self._executable_name = executable_name
        self.resolve_each_poll = resolve_each_poll

        self._kernel32 = _get_kernel32()
        self._attached: AttachedProcess | None = None
        self._reader: ProcessMemoryReader | None = None
        self._module_cache: dict[str, int] = {}
        self._resolved_cache: dict[str, int] = {}

    @property
    def attached(self) -> AttachedProcess:
        """Return current attached process metadata."""
        if self._attached is None:
            raise StateMonitorError("Monitor is not attached.")
        return self._attached

    def start(self) -> None:
        """Attach to target process and initialize reader."""
        self._attached = attach_process(
            executable_name=self._executable_name,
            retries=5,
            retry_delay_seconds=0.5,
        )
        self._reader = ProcessMemoryReader(process_handle=self._attached.handle)
        self._module_cache.clear()
        self._resolved_cache.clear()

    def stop(self) -> None:
        """Detach from process handle if attached."""
        if self._attached is not None:
            close_attached_process(self._attached)
            self._attached = None
            self._reader = None
            self._module_cache.clear()
            self._resolved_cache.clear()

    def _select_entries(
        self,
        entries: tuple[OffsetEntry, ...],
        fields_filter: str,
    ) -> tuple[OffsetEntry, ...]:
        if not fields_filter.strip():
            return entries
        requested = {name.strip() for name in fields_filter.split(",") if name.strip()}
        selected = tuple(entry for entry in entries if entry.name in requested)
        missing = sorted(requested - {entry.name for entry in selected})
        if missing:
            raise StateMonitorError(f"Requested fields not found in offsets config: {', '.join(missing)}")
        return selected

    def _module_base_resolver(self, module_name: str) -> ReadResult[int]:
        if self._attached is None:
            return ReadResult.fail(
                ReadFailure(code="not_attached", message="Monitor is not attached to a process.")
            )
        if module_name in self._module_cache:
            return ReadResult.ok(self._module_cache[module_name])

        try:
            base = _find_module_base(self._attached.pid, module_name, self._kernel32)
        except Exception as error:
            return ReadResult.fail(
                ReadFailure(
                    code="module_base_resolve_failed",
                    message=f"Unable to resolve module base for '{module_name}'.",
                    detail=str(error),
                )
            )
        self._module_cache[module_name] = base
        return ReadResult.ok(base)

    def _decode_value(self, entry: OffsetEntry, address: int) -> tuple[str, str, str]:
        if self._reader is None:
            return ("", "error", "reader_not_initialized")

        normalized_type = entry.data_type.strip().lower()
        if normalized_type == "int32":
            result = self._reader.read_int32(address)
            return self._from_result(result)
        if normalized_type == "int64":
            result = self._reader.read_int64(address)
            return self._from_result(result)
        if normalized_type == "uint32":
            raw = self._reader.read_bytes(address, 4)
            if not raw.is_ok:
                return self._from_result(raw)
            return (str(int.from_bytes(raw.value or b"\x00" * 4, "little", signed=False)), "ok", "")
        if normalized_type == "uint64":
            result = self._reader.read_uint64(address)
            return self._from_result(result)
        if normalized_type == "float":
            result = self._reader.read_float32(address)
            if not result.is_ok:
                return self._from_result(result)
            return (f"{result.value:.6g}" if result.value is not None else "", "ok", "")
        if normalized_type == "bool":
            result = self._reader.read_bool(address)
            if not result.is_ok:
                return self._from_result(result)
            return ("true" if result.value else "false", "ok", "")
        if normalized_type == "array<int32>":
            return self._decode_array_int32(address)

        bytes_match = BYTES_TYPE_REGEX.fullmatch(entry.data_type.strip())
        if bytes_match:
            size = int(bytes_match.group(1))
            raw = self._reader.read_bytes(address, size)
            if not raw.is_ok:
                return self._from_result(raw)
            value = raw.value.hex(" ") if raw.value is not None else ""
            return (value, "ok", "")

        return ("", "error", f"unsupported_data_type:{entry.data_type}")

    def _decode_array_int32(self, address: int) -> tuple[str, str, str]:
        if self._reader is None:
            return ("", "error", "reader_not_initialized")

        begin_result = self._reader.read_pointer(address)
        if not begin_result.is_ok:
            return self._from_result(begin_result)
        end_result = self._reader.read_pointer(address + self._reader.pointer_size)
        if not end_result.is_ok:
            return self._from_result(end_result)

        begin = begin_result.value or 0
        end = end_result.value or 0
        if begin == 0 and end == 0:
            return ("count=0", "ok", "")
        if begin == 0 or end == 0:
            return (f"begin=0x{begin:X} end=0x{end:X}", "error", "vector_null_boundary")
        if end < begin:
            return (f"begin=0x{begin:X} end=0x{end:X}", "error", "vector_end_before_begin")
        if (end - begin) % 4 != 0:
            return (f"begin=0x{begin:X} end=0x{end:X}", "error", "vector_unaligned_span")

        count = (end - begin) // 4
        preview_count = min(count, 6)
        preview: list[str] = []
        for index in range(preview_count):
            value_result = self._reader.read_int32(begin + index * 4)
            if not value_result.is_ok:
                return (
                    f"count={count}",
                    "error",
                    f"vector_preview_read_failed:{value_result.error.code if value_result.error else 'unknown'}",
                )
            preview.append(str(value_result.value))

        suffix = "" if count <= preview_count else ", ..."
        preview_text = ", ".join(preview)
        return (f"count={count} [{preview_text}{suffix}]", "ok", "")

    def _from_result(self, result: ReadResult[Any]) -> tuple[str, str, str]:
        if result.is_ok:
            return (str(result.value), "ok", "")
        if result.error is None:
            return ("", "error", "unknown_read_failure")
        detail = f" ({result.error.detail})" if result.error.detail else ""
        return ("", "error", f"{result.error.code}{detail}")

    def poll(self) -> PollSnapshot:
        """Read all selected fields once."""
        if self._reader is None:
            raise StateMonitorError("Monitor is not attached.")

        if self.resolve_each_poll:
            self._module_cache.clear()
            self._resolved_cache.clear()

        rows: list[FieldSnapshot] = []
        for entry in self._entries:
            address_hex = "N/A"
            value = ""
            status = "error"
            error = ""

            try:
                if entry.name in self._resolved_cache:
                    resolved_address = self._resolved_cache[entry.name]
                else:
                    resolve_result = resolve_offset_entry_address(
                        reader=self._reader,
                        entry=entry,
                        module_base_resolver=self._module_base_resolver,
                    )
                    if not resolve_result.is_ok or resolve_result.value is None:
                        resolve_error = (
                            resolve_result.error.code if resolve_result.error is not None else "resolve_failed"
                        )
                        rows.append(
                            FieldSnapshot(
                                name=entry.name,
                                data_type=entry.data_type,
                                confidence=entry.confidence,
                                address=address_hex,
                                value=value,
                                status="resolve_error",
                                error=resolve_error,
                            )
                        )
                        continue
                    resolved_address = resolve_result.value
                    self._resolved_cache[entry.name] = resolved_address

                address_hex = f"0x{resolved_address:X}"
                value, status, error = self._decode_value(entry, resolved_address)
            except Exception as poll_error:  # pragma: no cover - runtime diagnostics path
                status = "error"
                error = str(poll_error)

            rows.append(
                FieldSnapshot(
                    name=entry.name,
                    data_type=entry.data_type,
                    confidence=entry.confidence,
                    address=address_hex,
                    value=value,
                    status=status,
                    error=error,
                )
            )

        return PollSnapshot(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            fields=tuple(rows),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive TUI memory state monitor")
    parser.add_argument(
        "--exe",
        default="868-HACK.exe",
        help="Target executable name for process attach (default: 868-HACK.exe).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to offsets config (default: src/config/offsets.json).",
    )
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated field names to monitor. Defaults to all entries.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--resolve-each-poll",
        action="store_true",
        help="Re-resolve pointer chains every poll.",
    )
    return parser


def _run_tui(engine: MemoryStateMonitor, interval: float) -> None:
    try:
        from textual import events
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.widgets import DataTable, Footer, Header, Static
    except ModuleNotFoundError as import_error:  # pragma: no cover - dependency guard path
        raise StateMonitorError(
            "Missing dependency 'textual'. Install dependencies with `pip install -e .[dev]`."
        ) from import_error

    class MonitorApp(App[None]):
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("z", "toggle_pause", "Pause/Resume"),
            Binding("r", "refresh_now", "Refresh"),
            Binding("p", "toggle_resolve_mode", "Resolve/Cache"),
            Binding("up", "passthrough_up", show=False, priority=True),
            Binding("down", "passthrough_down", show=False, priority=True),
            Binding("left", "passthrough_left", show=False, priority=True),
            Binding("right", "passthrough_right", show=False, priority=True),
        ]

        paused = False

        def __init__(self, monitor_engine: MemoryStateMonitor, poll_interval: float) -> None:
            super().__init__()
            self._engine = monitor_engine
            self._poll_interval = poll_interval
            self._status_widget: Static | None = None
            self._table: DataTable | None = None
            self._last_snapshot: PollSnapshot | None = None
            self._controls = ActionAPI(input_driver=InputDriver())
            self._window: AttachedWindow | None = None
            self._control_state = "controls=init"
            self._last_command = ""

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield DataTable(id="state_table")
            yield Static("", id="status_line")
            yield Footer()

        def on_mount(self) -> None:
            self._table = self.query_one("#state_table", DataTable)
            self._status_widget = self.query_one("#status_line", Static)

            self._table.cursor_type = "none"
            self._table.zebra_stripes = True
            self._table.add_columns(
                "Field",
                "Type",
                "Confidence",
                "Address",
                "Value",
                "Status",
                "Error",
            )
            self._initialize_controls()
            self._refresh_once()
            self.set_interval(self._poll_interval, self._on_tick)

        def _initialize_controls(self) -> None:
            try:
                self._window = attach_window(
                    pid=self._engine.attached.pid,
                    retries=3,
                    retry_delay_seconds=0.3,
                )
                self._control_state = "controls=enabled(window)"
            except WindowAttachError as error:
                self._window = None
                self._control_state = f"controls=disabled ({error})"

        def _on_tick(self) -> None:
            if not self.paused:
                self._refresh_once()

        def _refresh_once(self) -> None:
            snapshot = self._engine.poll()
            if self._table is None or self._status_widget is None:
                return

            self._table.clear(columns=False)
            for row in snapshot.fields:
                self._table.add_row(
                    row.name,
                    row.data_type,
                    row.confidence,
                    row.address,
                    row.value,
                    row.status,
                    row.error,
                )

            self._last_snapshot = snapshot
            self._update_status_line()

        def _update_status_line(self) -> None:
            if self._status_widget is None or self._last_snapshot is None:
                return

            snapshot = self._last_snapshot
            mode = "resolve-each-poll" if self._engine.resolve_each_poll else "cached-addresses"
            paused_flag = "paused" if self.paused else "running"
            command = f" | command={self._last_command}" if self._last_command else ""
            self._status_widget.update(
                f"{snapshot.timestamp} | pid={self._engine.attached.pid} | {paused_flag} | "
                f"mode={mode} | interval={self._poll_interval:.2f}s | {self._control_state}{command}"
            )

        def action_toggle_pause(self) -> None:
            self.paused = not self.paused
            self._update_status_line()

        def action_refresh_now(self) -> None:
            self._refresh_once()

        def action_toggle_resolve_mode(self) -> None:
            self._engine.resolve_each_poll = not self._engine.resolve_each_poll
            self._refresh_once()

        def action_passthrough_up(self) -> None:
            self._send_passthrough_key("UP")

        def action_passthrough_down(self) -> None:
            self._send_passthrough_key("DOWN")

        def action_passthrough_left(self) -> None:
            self._send_passthrough_key("LEFT")

        def action_passthrough_right(self) -> None:
            self._send_passthrough_key("RIGHT")

        def on_key(self, event: events.Key) -> None:
            if event.key in {"up", "down", "left", "right"}:
                return
            passthrough_key = map_tui_key_to_passthrough_key(event.key)
            if passthrough_key is None:
                return
            event.stop()
            self._send_passthrough_key(passthrough_key)

        def _send_passthrough_key(self, key_name: str) -> None:
            try:
                if self._window is None:
                    self._window = attach_window(
                        pid=self._engine.attached.pid,
                        retries=3,
                        retry_delay_seconds=0.3,
                    )
                try:
                    self._controls.send_key_name_to_window(key_name, hwnd=self._window.hwnd)
                except InputDriverError:
                    self._window = attach_window(
                        pid=self._engine.attached.pid,
                        retries=3,
                        retry_delay_seconds=0.3,
                    )
                    self._controls.send_key_name_to_window(key_name, hwnd=self._window.hwnd)
                self._control_state = "controls=enabled(window)"
                self._last_command = f"sent:{key_name}"
            except (WindowAttachError, InputDriverError, ActionExecutionError) as error:
                self._last_command = f"failed:{key_name}"
                self._control_state = f"controls=error ({error})"
            self._update_status_line()

    MonitorApp(engine, interval).run()


def main() -> None:
    """CLI entrypoint for interactive memory-state TUI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be > 0.")

    engine = MemoryStateMonitor(
        executable_name=args.exe,
        config_path=args.config,
        fields_filter=args.fields,
        resolve_each_poll=args.resolve_each_poll,
    )

    try:
        engine.start()
        _run_tui(engine, args.interval)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

"""Live offset smoke test for validating configured pointer chains.

Usage:
    python -m src.memory.offset_smoke_test
    python -m src.memory.offset_smoke_test --fields player_energy,player_credits --interval 0.25
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import datetime as dt
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from src.config.offsets import OffsetEntry, load_offset_registry
from src.memory.process_attach import attach_process, close_attached_process

TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class OffsetSmokeTestError(RuntimeError):
    """Raised when live smoke testing cannot proceed."""


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
class ResolvedField:
    """Resolved memory target for an offset entry."""

    entry: OffsetEntry
    address: int


def _get_kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise OffsetSmokeTestError("Offset smoke test is only supported on Windows.")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _read_exact(handle: int, address: int, size: int, kernel32: ctypes.WinDLL) -> bytes:
    buffer = (ctypes.c_ubyte * size)()
    bytes_read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        ctypes.c_void_p(handle),
        ctypes.c_void_p(address),
        ctypes.byref(buffer),
        size,
        ctypes.byref(bytes_read),
    )
    if not ok:
        error_code = ctypes.get_last_error()
        raise OffsetSmokeTestError(
            f"ReadProcessMemory failed at 0x{address:X} (size={size}, error={error_code})."
        )
    if bytes_read.value != size:
        raise OffsetSmokeTestError(
            f"Short read at 0x{address:X}: requested {size}, got {bytes_read.value}."
        )
    return bytes(buffer)


def _read_u64(handle: int, address: int, kernel32: ctypes.WinDLL) -> int:
    return struct.unpack("<Q", _read_exact(handle, address, 8, kernel32))[0]


def _read_i32(handle: int, address: int, kernel32: ctypes.WinDLL) -> int:
    return struct.unpack("<i", _read_exact(handle, address, 4, kernel32))[0]


def _read_u8(handle: int, address: int, kernel32: ctypes.WinDLL) -> int:
    return _read_exact(handle, address, 1, kernel32)[0]


def _find_module_base(pid: int, module_name: str, kernel32: ctypes.WinDLL) -> int:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snapshot == INVALID_HANDLE_VALUE:
        error_code = ctypes.get_last_error()
        raise OffsetSmokeTestError(
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
                    raise OffsetSmokeTestError(
                        f"Module '{module_name}' found, but base address is null."
                    )
                return int(base)
            has_module = bool(kernel32.Module32NextW(snapshot, ctypes.byref(module_entry)))
    finally:
        kernel32.CloseHandle(snapshot)

    raise OffsetSmokeTestError(f"Module '{module_name}' not found in pid={pid}.")


def _resolve_entry_address(
    *,
    pid: int,
    process_handle: int,
    entry: OffsetEntry,
    kernel32: ctypes.WinDLL,
) -> int:
    if entry.base.kind == "module":
        base_address = _find_module_base(pid, entry.base.value, kernel32)
    else:
        base_address = int(entry.base.value, 16)

    cursor = base_address
    for chain_step in entry.pointer_chain:
        pointer_address = cursor + chain_step
        cursor = _read_u64(process_handle, pointer_address, kernel32)
        if cursor == 0:
            raise OffsetSmokeTestError(
                f"Null pointer while resolving '{entry.name}' at 0x{pointer_address:X}."
            )

    return cursor + entry.read_offset


def _decode_value(
    *,
    process_handle: int,
    entry: OffsetEntry,
    resolved_address: int,
    kernel32: ctypes.WinDLL,
) -> str:
    normalized_type = entry.data_type.strip().lower()
    if normalized_type == "int32":
        return str(_read_i32(process_handle, resolved_address, kernel32))
    if normalized_type == "bool":
        return "true" if _read_u8(process_handle, resolved_address, kernel32) != 0 else "false"
    if normalized_type == "int64":
        return str(struct.unpack("<q", _read_exact(process_handle, resolved_address, 8, kernel32))[0])
    if normalized_type == "uint32":
        return str(struct.unpack("<I", _read_exact(process_handle, resolved_address, 4, kernel32))[0])
    if normalized_type == "float":
        return f"{struct.unpack('<f', _read_exact(process_handle, resolved_address, 4, kernel32))[0]:.6g}"
    if normalized_type == "array<int32>":
        begin = _read_u64(process_handle, resolved_address, kernel32)
        end = _read_u64(process_handle, resolved_address + 8, kernel32)
        if begin == 0 or end == 0 or end < begin:
            return f"array<int32>(invalid begin=0x{begin:X} end=0x{end:X})"
        count = (end - begin) // 4
        return f"array<int32>(count={count})"
    raise OffsetSmokeTestError(f"Unsupported data_type '{entry.data_type}' for field '{entry.name}'.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live offset smoke test")
    parser.add_argument(
        "--exe",
        default="868-HACK.exe",
        help="Target executable name for process attach (default: 868-HACK.exe).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to offsets.json (default: src/config/offsets.json).",
    )
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated field names to read. Defaults to all offset entries.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of polling iterations (0 = run until Ctrl+C).",
    )
    parser.add_argument(
        "--resolve-each-loop",
        action="store_true",
        help="Re-resolve pointer chains each poll instead of caching target addresses.",
    )
    return parser


def _select_entries(entries: tuple[OffsetEntry, ...], fields_arg: str) -> tuple[OffsetEntry, ...]:
    if not fields_arg.strip():
        return entries

    requested = {field.strip() for field in fields_arg.split(",") if field.strip()}
    selected = tuple(entry for entry in entries if entry.name in requested)
    missing = sorted(requested - {entry.name for entry in selected})
    if missing:
        raise OffsetSmokeTestError(f"Requested fields not found in offsets config: {', '.join(missing)}")
    return selected


def _poll(
    *,
    pid: int,
    process_handle: int,
    fields: tuple[OffsetEntry, ...],
    interval: float,
    iterations: int,
    resolve_each_loop: bool,
    kernel32: ctypes.WinDLL,
) -> None:
    cached: dict[str, ResolvedField] = {}
    if not resolve_each_loop:
        for entry in fields:
            cached[entry.name] = ResolvedField(
                entry=entry,
                address=_resolve_entry_address(
                    pid=pid,
                    process_handle=process_handle,
                    entry=entry,
                    kernel32=kernel32,
                ),
            )

    print(
        "timestamp"
        + "".join(f"\t{entry.name}" for entry in fields)
        + "".join(f"\t{entry.name}_addr" for entry in fields)
    )

    remaining = iterations
    while True:
        if remaining == 0 and iterations > 0:
            break

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        row_values: list[str] = [timestamp]

        for entry in fields:
            try:
                if resolve_each_loop:
                    address = _resolve_entry_address(
                        pid=pid,
                        process_handle=process_handle,
                        entry=entry,
                        kernel32=kernel32,
                    )
                else:
                    address = cached[entry.name].address
                value_text = _decode_value(
                    process_handle=process_handle,
                    entry=entry,
                    resolved_address=address,
                    kernel32=kernel32,
                )
                row_values.append(value_text)
                row_values.append(f"0x{address:X}")
            except Exception as error:  # pragma: no cover - runtime diagnostics path
                row_values.append(f"ERR:{error}")
                row_values.append("N/A")

        print("\t".join(row_values))

        if iterations > 0:
            remaining -= 1
        time.sleep(interval)


def main() -> None:
    """CLI entrypoint for live offset smoke testing."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be > 0.")
    if args.iterations < 0:
        raise SystemExit("--iterations must be >= 0.")

    registry = load_offset_registry(config_path=args.config)
    fields = _select_entries(registry.entries, args.fields)
    if not fields:
        raise SystemExit("No fields selected for smoke test.")

    kernel32 = _get_kernel32()
    attached = attach_process(executable_name=args.exe, retries=5, retry_delay_seconds=0.5)
    try:
        print(
            f"Attached pid={attached.pid} exe={attached.executable_name} "
            f"fields={','.join(field.name for field in fields)}"
        )
        _poll(
            pid=attached.pid,
            process_handle=attached.handle,
            fields=fields,
            interval=args.interval,
            iterations=args.iterations,
            resolve_each_loop=args.resolve_each_loop,
            kernel32=kernel32,
        )
    finally:
        close_attached_process(attached)


if __name__ == "__main__":
    main()


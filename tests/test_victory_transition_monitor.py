"""Tests for victory transition debugger monitor helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from src.config.offsets import OffsetBase, OffsetEntry
from src.memory.reader import BackendReadResponse, ProcessMemoryReader
from src.memory.victory_transition_monitor import (
    DEFAULT_OUTPUT_PATH,
    FIELD_OFFSET_VICTORY_ACTIVE,
    FIELD_OFFSET_VICTORY_PENDING,
    NORMAL_VICTORY_FLAG_SET_PREFERRED_VA,
    POINTS_VICTORY_FLAG_SET_PREFERRED_VA,
    ROOT_POINTER_MODULE_OFFSET,
    DebugRegisterWatch,
    _clear_watch_enable_bits,
    _build_debug_register_watches,
    _build_dr7,
    _build_parser,
    _build_watch_addresses,
    _default_targets,
    _read_snapshot_fields,
    _runtime_address_from_preferred,
    _triggered_slots,
)


@dataclass
class FakeMemoryBackend:
    """Simple fake memory backend for monitor tests."""

    memory_by_address: dict[int, bytes] = field(default_factory=dict)
    error_by_address: dict[int, int] = field(default_factory=dict)

    def read_memory(self, process_handle: int, address: int, size: int) -> BackendReadResponse:
        if address in self.error_by_address:
            return BackendReadResponse(data=b"", bytes_read=0, error_code=self.error_by_address[address])
        data = self.memory_by_address.get(address)
        if data is None:
            return BackendReadResponse(data=b"", bytes_read=0, error_code=487)
        return BackendReadResponse(data=data[:size], bytes_read=min(size, len(data)), error_code=None)


def _entry(name: str, data_type: str, read_offset: int) -> OffsetEntry:
    return OffsetEntry(
        name=name,
        data_type=data_type,
        base=OffsetBase(kind="module", value="868-HACK.exe"),
        pointer_chain=(ROOT_POINTER_MODULE_OFFSET,),
        read_offset=read_offset,
        confidence="medium",
        notes="test entry",
    )


def test_default_targets_cover_normal_and_points_victory_flags() -> None:
    targets = _default_targets()

    assert [target.name for target in targets] == [
        "normal_victory_flag_set",
        "points_victory_flag_set",
    ]
    assert [target.slot for target in targets] == [0, 1]


def test_runtime_address_from_preferred_uses_module_base_delta() -> None:
    assert _runtime_address_from_preferred(
        module_base=0x0000019900000000,
        preferred_va=NORMAL_VICTORY_FLAG_SET_PREFERRED_VA,
    ) == 0x000001990000AADE


def test_build_watch_addresses_uses_module_victory_flag_addresses() -> None:
    addresses = _build_watch_addresses(module_base=0x200000000, root_address=0x300000000)

    assert addresses.normal_victory_flag_set == 0x200000000 + (
        NORMAL_VICTORY_FLAG_SET_PREFERRED_VA - 0x140000000
    )
    assert addresses.points_victory_flag_set == 0x200000000 + (
        POINTS_VICTORY_FLAG_SET_PREFERRED_VA - 0x140000000
    )


def test_build_dr7_enables_execute_breakpoints() -> None:
    dr7 = _build_dr7(
        (
            DebugRegisterWatch(slot=0, address=0x1111, access="execute"),
            DebugRegisterWatch(slot=1, address=0x2222, access="execute"),
        )
    )

    assert dr7 & (1 << 0)
    assert dr7 & (1 << 2)
    assert ((dr7 >> 16) & 0b11) == 0b00
    assert ((dr7 >> 20) & 0b11) == 0b00


def test_triggered_slots_decodes_dr6_low_bits() -> None:
    assert _triggered_slots(0b0101) == (0, 2)


def test_clear_watch_enable_bits_disables_requested_slot() -> None:
    original = _build_dr7(
        (
            DebugRegisterWatch(slot=0, address=0x1111, access="execute"),
            DebugRegisterWatch(slot=1, address=0x2222, access="execute"),
        )
    )

    cleared = _clear_watch_enable_bits(original, 0)

    assert cleared & (1 << 0) == 0
    assert cleared & (1 << 2)


def test_build_debug_register_watches_uses_module_base_only_execute_targets() -> None:
    watches = _build_debug_register_watches(
        _build_watch_addresses(module_base=0x180000000, root_address=None)
    )

    assert [watch.slot for watch in watches] == [0, 1]
    assert [watch.access for watch in watches] == ["execute", "execute"]


def test_read_snapshot_fields_resolves_and_reads_registry_entries() -> None:
    module_base = 0x10000000
    root_address = 0x20000000
    backend = FakeMemoryBackend(
        memory_by_address={
            module_base + ROOT_POINTER_MODULE_OFFSET: struct.pack("<Q", root_address),
            root_address + 0x19B8: struct.pack("<i", 7),
            root_address + 0x1BD4: struct.pack("<i", 8),
            root_address + 0x19C8: struct.pack("<i", 5),
            root_address + 0x19CC: struct.pack("<i", 4),
            root_address + 0x1B94: b"\x01",
            root_address + FIELD_OFFSET_VICTORY_PENDING: b"\x01",
            root_address + FIELD_OFFSET_VICTORY_ACTIVE: b"\x00",
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    entries = {
        "current_sector": _entry("current_sector", "int32", 0x19B8),
        "sector_progression_index": _entry("sector_progression_index", "int32", 0x1BD4),
        "player_x": _entry("player_x", "int32", 0x19C8),
        "player_y": _entry("player_y", "int32", 0x19CC),
        "run_active": _entry("run_active", "bool", 0x1B94),
        "victory_pending": _entry("victory_pending", "bool", FIELD_OFFSET_VICTORY_PENDING),
        "victory_active": _entry("victory_active", "bool", FIELD_OFFSET_VICTORY_ACTIVE),
    }

    snapshot = _read_snapshot_fields(reader=reader, entries=entries, module_base=module_base)

    assert snapshot["current_sector"]["status"] == "ok"
    assert snapshot["current_sector"]["value"] == 7
    assert snapshot["sector_progression_index"]["value"] == 8
    assert snapshot["player_x"]["value"] == 5
    assert snapshot["player_y"]["value"] == 4
    assert snapshot["run_active"]["value"] is True
    assert snapshot["victory_pending"]["value"] is True
    assert snapshot["victory_active"]["value"] is False


def test_victory_monitor_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.exe == "868-HACK.exe"
    assert args.pid is None
    assert Path(args.output) == DEFAULT_OUTPUT_PATH
    assert args.max_events == 0
    assert args.timeout_seconds == 0.0
    assert args.launch_if_missing is False

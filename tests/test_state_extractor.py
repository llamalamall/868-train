"""Tests for normalized state extraction."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import pytest

from src.config.offsets import OffsetBase, OffsetEntry, OffsetRegistry
from src.memory.reader import BackendReadResponse, ProcessMemoryReader
from src.state.extractor import extract_state


@dataclass
class FakeMemoryBackend:
    """Simple fake memory backend for state-extractor tests."""

    memory_by_address: dict[int, bytes] = field(default_factory=dict)
    error_by_address: dict[int, int] = field(default_factory=dict)

    def read_memory(self, process_handle: int, address: int, size: int) -> BackendReadResponse:
        if address in self.error_by_address:
            return BackendReadResponse(data=b"", bytes_read=0, error_code=self.error_by_address[address])

        data = self.memory_by_address.get(address)
        if data is None:
            return BackendReadResponse(data=b"", bytes_read=0, error_code=487)

        bytes_read = min(size, len(data))
        return BackendReadResponse(data=data[:bytes_read], bytes_read=bytes_read, error_code=None)


def _entry(name: str, data_type: str, address: int) -> OffsetEntry:
    return OffsetEntry(
        name=name,
        data_type=data_type,
        base=OffsetBase(kind="absolute", value=f"0x{address:X}"),
        pointer_chain=(),
        confidence="high",
        notes="test",
        read_offset=0,
    )


def test_extract_state_returns_normalized_snapshot_for_core_fields() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 12),
            0x200004: struct.pack("<i", 5),
            0x200008: struct.pack("<i", 41),
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(
        reader=reader,
        registry=registry,
        timestamp_fn=lambda: "2026-03-03T00:00:00+00:00",
    )

    assert snapshot.timestamp_utc == "2026-03-03T00:00:00+00:00"
    assert snapshot.health.status == "ok"
    assert snapshot.health.value == 12
    assert snapshot.energy.status == "ok"
    assert snapshot.energy.value == 5
    assert snapshot.currency.status == "ok"
    assert snapshot.currency.value == 41
    assert snapshot.fail_state.status == "ok"
    assert snapshot.fail_state.value is False


def test_extract_state_marks_missing_core_field_with_explicit_metadata() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_credits", "int32", 0x200008),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 7),
            0x200008: struct.pack("<i", 9),
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.energy.status == "missing"
    assert snapshot.energy.value is None
    assert snapshot.energy.error_code == "field_not_configured"


def test_extract_state_derives_fail_state_when_health_is_negative_one() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", -1),
            0x200004: struct.pack("<i", 3),
            0x200008: struct.pack("<i", 15),
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.fail_state.status == "ok"
    assert snapshot.fail_state.value is True
    assert snapshot.fail_state.source_field == "derived:player_health==-1"


def test_extract_state_uses_explicit_fail_state_when_configured() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("fail_state", "bool", 0x20000C),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 20),
            0x200004: struct.pack("<i", 8),
            0x200008: struct.pack("<i", 30),
            0x20000C: b"\x01",
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.fail_state.status == "ok"
    assert snapshot.fail_state.value is True
    assert snapshot.fail_state.source_field == "fail_state"


def test_extract_state_derives_fail_state_from_run_active_when_available() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("run_active", "bool", 0x20000C),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 20),
            0x200004: struct.pack("<i", 8),
            0x200008: struct.pack("<i", 30),
            0x20000C: b"\x00",
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.fail_state.status == "ok"
    assert snapshot.fail_state.value is True
    assert snapshot.fail_state.source_field == "derived:not_run_active"


def test_extract_state_reports_invalid_field_read() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 10),
            0x200004: struct.pack("<i", 4),
        },
        error_by_address={0x200008: 299},
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.currency.status == "invalid"
    assert snapshot.currency.value is None
    assert snapshot.currency.error_code == "read_failed"


def test_extract_state_inventory_empty_when_collected_progs_vector_is_empty() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("collected_progs", "array<int32>", 0x200100),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 10),
            0x200004: struct.pack("<i", 3),
            0x200008: struct.pack("<i", 21),
            0x200100: struct.pack("<Q", 0),
            0x200108: struct.pack("<Q", 0),
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.inventory.status == "ok"
    assert snapshot.inventory.raw_prog_ids == ()
    assert snapshot.inventory.collected_progs == ()
    assert snapshot.inventory.unknown_prog_ids == ()


def test_extract_state_inventory_partial_vector_reports_invalid() -> None:
    begin = 0x300000
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("collected_progs", "array<int32>", 0x200100),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 10),
            0x200004: struct.pack("<i", 3),
            0x200008: struct.pack("<i", 21),
            0x200100: struct.pack("<Q", begin),
            0x200108: struct.pack("<Q", begin + 12),  # count=3
            begin: struct.pack("<i", 2),
            begin + 4: struct.pack("<i", 4),
            # Missing third value at begin+8 simulates partial/truncated vector content.
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.inventory.status == "invalid"
    assert snapshot.inventory.error_code == "read_failed"


def test_extract_state_inventory_populated_preserves_unknown_ids_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    begin = 0x300100
    prog_ids = (2, 4, 2, 99)
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("collected_progs", "array<int32>", 0x200100),
        ),
    )
    memory = {
        0x200000: struct.pack("<i", 10),
        0x200004: struct.pack("<i", 3),
        0x200008: struct.pack("<i", 21),
        0x200100: struct.pack("<Q", begin),
        0x200108: struct.pack("<Q", begin + len(prog_ids) * 4),
    }
    for index, prog_id in enumerate(prog_ids):
        memory[begin + index * 4] = struct.pack("<i", prog_id)

    backend = FakeMemoryBackend(memory_by_address=memory)
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    caplog.set_level("WARNING")
    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.inventory.status == "ok"
    assert snapshot.inventory.raw_prog_ids == prog_ids
    assert snapshot.inventory.unknown_prog_ids == (99,)
    assert tuple(item.prog_id for item in snapshot.inventory.collected_progs) == (2, 4, 99)
    assert tuple(item.count for item in snapshot.inventory.collected_progs) == (2, 1, 1)
    assert snapshot.inventory.collected_progs[0].name == ".show"
    assert snapshot.inventory.collected_progs[1].name == ".pull"
    assert snapshot.inventory.collected_progs[2].name is None
    assert "Unknown collected prog id detected: 99" in caplog.text

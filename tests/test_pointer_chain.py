"""Tests for pointer-chain resolution behavior."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from src.config.offsets import OffsetBase, OffsetEntry
from src.memory.pointer_chain import resolve_offset_entry_address, resolve_pointer_chain
from src.memory.reader import BackendReadResponse, ProcessMemoryReader, ReadResult


@dataclass
class FakeMemoryBackend:
    """Simple fake memory backend for pointer-chain tests."""

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


def test_resolve_pointer_chain_success() -> None:
    base = 0x140000000
    root = 0x50000000
    backend = FakeMemoryBackend(memory_by_address={base + 0x115808: struct.pack("<Q", root)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = resolve_pointer_chain(
        reader=reader,
        base_address=base,
        pointer_chain=(0x115808,),
        final_offset=0x1B60,
    )

    assert result.is_ok
    assert result.value == root + 0x1B60
    assert result.traversed_pointers == (root,)


def test_resolve_pointer_chain_rejects_null_pointer() -> None:
    base = 0x140000000
    backend = FakeMemoryBackend(memory_by_address={base + 0x10: struct.pack("<Q", 0)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = resolve_pointer_chain(
        reader=reader,
        base_address=base,
        pointer_chain=(0x10,),
    )

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "null_pointer"


def test_resolve_pointer_chain_wraps_reader_failure() -> None:
    base = 0x140000000
    backend = FakeMemoryBackend(error_by_address={base + 0x20: 299})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = resolve_pointer_chain(
        reader=reader,
        base_address=base,
        pointer_chain=(0x20,),
    )

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "pointer_read_failed"
    assert result.error.read_failure is not None
    assert result.error.read_failure.code == "read_failed"


def test_resolve_offset_entry_address_uses_module_base_resolver() -> None:
    module_base = 0x140000000
    root = 0x50000000
    entry = OffsetEntry(
        name="player_energy",
        data_type="int32",
        base=OffsetBase(kind="module", value="868-HACK.exe"),
        pointer_chain=(0x115808,),
        confidence="high",
        notes="test",
        read_offset=0x1B60,
    )
    backend = FakeMemoryBackend(memory_by_address={module_base + 0x115808: struct.pack("<Q", root)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    def resolve_module_base(module_name: str) -> ReadResult[int]:
        assert module_name == "868-HACK.exe"
        return ReadResult.ok(module_base)

    result = resolve_offset_entry_address(
        reader=reader,
        entry=entry,
        module_base_resolver=resolve_module_base,
    )

    assert result.is_ok
    assert result.value == root + 0x1B60


def test_resolve_offset_entry_address_requires_module_resolver() -> None:
    entry = OffsetEntry(
        name="player_credits",
        data_type="int32",
        base=OffsetBase(kind="module", value="868-HACK.exe"),
        pointer_chain=(0x115808,),
        confidence="high",
        notes="test",
        read_offset=0x1B5C,
    )
    reader = ProcessMemoryReader(process_handle=1, backend=FakeMemoryBackend())

    result = resolve_offset_entry_address(reader=reader, entry=entry, module_base_resolver=None)

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "module_base_resolver_missing"

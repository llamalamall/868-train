"""Tests for typed process memory reading primitives."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from src.memory.reader import BackendReadResponse, ProcessMemoryReader


@dataclass
class FakeMemoryBackend:
    """Simple fake memory backend for reader tests."""

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


def test_read_int32_success() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: struct.pack("<i", 1234)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_int32(0x200000)

    assert result.is_ok
    assert result.value == 1234


def test_read_bool_success() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200100: b"\x01"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_bool(0x200100)

    assert result.is_ok
    assert result.value is True


def test_read_rejects_null_address() -> None:
    backend = FakeMemoryBackend()
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_int32(0)

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "null_address"


def test_read_rejects_invalid_size() -> None:
    backend = FakeMemoryBackend()
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_bytes(0x200000, 0)

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "invalid_size_value"


def test_read_returns_short_read_error() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200200: b"\xAA\xBB"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_int32(0x200200)

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "short_read"


def test_read_propagates_backend_failure() -> None:
    backend = FakeMemoryBackend(error_by_address={0x200300: 299})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    result = reader.read_int32(0x200300)

    assert not result.is_ok
    assert result.error is not None
    assert result.error.code == "read_failed"
    assert result.error.detail == "error_code=299"


def test_read_pointer_uses_pointer_size() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x300000: struct.pack("<I", 0xDEADBEEF)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend, pointer_size=4)

    result = reader.read_pointer(0x300000)

    assert result.is_ok
    assert result.value == 0xDEADBEEF

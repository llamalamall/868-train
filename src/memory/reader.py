"""Typed process memory read primitives."""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from src.memory.validators import AddressRange, ValidationIssue, validate_address, validate_read_size

T = TypeVar("T")


@dataclass(frozen=True)
class ReadFailure:
    """Structured read failure for diagnostics and control flow."""

    code: str
    message: str
    address: int | None = None
    size: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ReadResult(Generic[T]):
    """Typed memory read result with explicit success/failure state."""

    value: T | None
    error: ReadFailure | None = None

    @property
    def is_ok(self) -> bool:
        """Return true when a read succeeded."""
        return self.error is None

    @classmethod
    def ok(cls, value: T) -> ReadResult[T]:
        """Create a successful result."""
        return cls(value=value, error=None)

    @classmethod
    def fail(cls, failure: ReadFailure) -> ReadResult[T]:
        """Create a failed result."""
        return cls(value=None, error=failure)


@dataclass(frozen=True)
class BackendReadResponse:
    """Low-level backend read response."""

    data: bytes
    bytes_read: int
    error_code: int | None = None


class MemoryReadBackend(Protocol):
    """Backend contract for process memory reads."""

    def read_memory(self, process_handle: int, address: int, size: int) -> BackendReadResponse:
        """Read raw bytes from the target process."""


class WindowsMemoryReadBackend:
    """Windows backend using ReadProcessMemory."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("WindowsMemoryReadBackend is only supported on Windows.")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def read_memory(self, process_handle: int, address: int, size: int) -> BackendReadResponse:
        buffer = (ctypes.c_ubyte * size)()
        bytes_read = ctypes.c_size_t(0)
        ok = self._kernel32.ReadProcessMemory(
            ctypes.c_void_p(process_handle),
            ctypes.c_void_p(address),
            ctypes.byref(buffer),
            size,
            ctypes.byref(bytes_read),
        )
        if not ok:
            return BackendReadResponse(data=b"", bytes_read=int(bytes_read.value), error_code=ctypes.get_last_error())

        length = int(bytes_read.value)
        return BackendReadResponse(data=bytes(buffer[:length]), bytes_read=length, error_code=None)


def _failure_from_issue(
    issue: ValidationIssue,
    *,
    address: int | None = None,
    size: int | None = None,
) -> ReadFailure:
    return ReadFailure(
        code=issue.code,
        message=issue.message,
        address=address,
        size=size,
    )


class ProcessMemoryReader:
    """Typed memory reader with address guards and structured failures."""

    def __init__(
        self,
        *,
        process_handle: int,
        backend: MemoryReadBackend | None = None,
        address_range: AddressRange | None = None,
        pointer_size: int = 8,
    ) -> None:
        if pointer_size not in (4, 8):
            raise ValueError("pointer_size must be 4 or 8.")
        self._process_handle = process_handle
        self._backend = backend or WindowsMemoryReadBackend()
        self._address_range = address_range or AddressRange()
        self._pointer_size = pointer_size

    @property
    def pointer_size(self) -> int:
        """Pointer size in bytes."""
        return self._pointer_size

    def _read_exact(self, address: int, size: int) -> ReadResult[bytes]:
        address_issue = validate_address(address, allowed_range=self._address_range)
        if address_issue is not None:
            return ReadResult.fail(_failure_from_issue(address_issue, address=address, size=size))

        size_issue = validate_read_size(size)
        if size_issue is not None:
            return ReadResult.fail(_failure_from_issue(size_issue, address=address, size=size))

        response = self._backend.read_memory(self._process_handle, address, size)
        if response.error_code is not None:
            return ReadResult.fail(
                ReadFailure(
                    code="read_failed",
                    message=f"Backend read failed for 0x{address:X}.",
                    address=address,
                    size=size,
                    detail=f"error_code={response.error_code}",
                )
            )

        if response.bytes_read != size or len(response.data) != size:
            return ReadResult.fail(
                ReadFailure(
                    code="short_read",
                    message=(
                        f"Short read for 0x{address:X}: requested {size}, got {response.bytes_read}."
                    ),
                    address=address,
                    size=size,
                    detail=f"bytes_len={len(response.data)}",
                )
            )

        return ReadResult.ok(response.data)

    def read_bytes(self, address: int, size: int) -> ReadResult[bytes]:
        """Read an exact byte slice from process memory."""
        return self._read_exact(address, size)

    def read_int32(self, address: int) -> ReadResult[int]:
        """Read a signed 32-bit integer."""
        raw = self._read_exact(address, 4)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok(struct.unpack("<i", raw.value or b"")[0])

    def read_int64(self, address: int) -> ReadResult[int]:
        """Read a signed 64-bit integer."""
        raw = self._read_exact(address, 8)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok(struct.unpack("<q", raw.value or b"")[0])

    def read_uint64(self, address: int) -> ReadResult[int]:
        """Read an unsigned 64-bit integer."""
        raw = self._read_exact(address, 8)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok(struct.unpack("<Q", raw.value or b"")[0])

    def read_float32(self, address: int) -> ReadResult[float]:
        """Read a 32-bit IEEE float."""
        raw = self._read_exact(address, 4)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok(struct.unpack("<f", raw.value or b"")[0])

    def read_bool(self, address: int) -> ReadResult[bool]:
        """Read a boolean represented by one byte."""
        raw = self._read_exact(address, 1)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok((raw.value or b"\x00")[0] != 0)

    def read_pointer(self, address: int) -> ReadResult[int]:
        """Read a pointer-sized unsigned integer."""
        if self._pointer_size == 8:
            return self.read_uint64(address)
        raw = self._read_exact(address, 4)
        if not raw.is_ok:
            return ReadResult.fail(raw.error or ReadFailure(code="unknown", message="Unknown read failure."))
        return ReadResult.ok(struct.unpack("<I", raw.value or b"")[0])

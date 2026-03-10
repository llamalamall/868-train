"""Typed process memory write primitives."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Protocol

from src.memory.validators import AddressRange, ValidationIssue, validate_address, validate_read_size


@dataclass(frozen=True)
class WriteFailure:
    """Structured write failure for diagnostics and control flow."""

    code: str
    message: str
    address: int | None = None
    size: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class WriteResult:
    """Memory write result with explicit success/failure state."""

    error: WriteFailure | None = None

    @property
    def is_ok(self) -> bool:
        """Return true when a write succeeded."""
        return self.error is None

    @classmethod
    def ok(cls) -> WriteResult:
        """Create a successful result."""
        return cls(error=None)

    @classmethod
    def fail(cls, failure: WriteFailure) -> WriteResult:
        """Create a failed result."""
        return cls(error=failure)


@dataclass(frozen=True)
class BackendWriteResponse:
    """Low-level backend write response."""

    bytes_written: int
    error_code: int | None = None
    flush_error_code: int | None = None


class MemoryWriteBackend(Protocol):
    """Backend contract for process memory writes."""

    def write_memory(self, process_handle: int, address: int, data: bytes) -> BackendWriteResponse:
        """Write raw bytes into target process memory."""


class WindowsMemoryWriteBackend:
    """Windows backend using WriteProcessMemory + FlushInstructionCache."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("WindowsMemoryWriteBackend is only supported on Windows.")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def write_memory(self, process_handle: int, address: int, data: bytes) -> BackendWriteResponse:
        if not data:
            return BackendWriteResponse(bytes_written=0, error_code=None, flush_error_code=None)

        write_buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        bytes_written = ctypes.c_size_t(0)
        ok = self._kernel32.WriteProcessMemory(
            ctypes.c_void_p(process_handle),
            ctypes.c_void_p(address),
            ctypes.byref(write_buffer),
            len(data),
            ctypes.byref(bytes_written),
        )
        if not ok:
            return BackendWriteResponse(bytes_written=int(bytes_written.value), error_code=ctypes.get_last_error())

        flush_ok = self._kernel32.FlushInstructionCache(
            ctypes.c_void_p(process_handle),
            ctypes.c_void_p(address),
            len(data),
        )
        flush_error = None if flush_ok else ctypes.get_last_error()
        return BackendWriteResponse(
            bytes_written=int(bytes_written.value),
            error_code=None,
            flush_error_code=flush_error,
        )


def _failure_from_issue(
    issue: ValidationIssue,
    *,
    address: int | None = None,
    size: int | None = None,
) -> WriteFailure:
    return WriteFailure(
        code=issue.code,
        message=issue.message,
        address=address,
        size=size,
    )


class ProcessMemoryWriter:
    """Typed memory writer with address guards and structured failures."""

    def __init__(
        self,
        *,
        process_handle: int,
        backend: MemoryWriteBackend | None = None,
        address_range: AddressRange | None = None,
    ) -> None:
        self._process_handle = process_handle
        self._backend = backend or WindowsMemoryWriteBackend()
        self._address_range = address_range or AddressRange()

    def write_bytes(self, address: int, data: bytes) -> WriteResult:
        """Write an exact byte slice into process memory."""
        address_issue = validate_address(address, allowed_range=self._address_range)
        if address_issue is not None:
            return WriteResult.fail(
                _failure_from_issue(address_issue, address=address, size=len(data))
            )

        size_issue = validate_read_size(len(data))
        if size_issue is not None:
            return WriteResult.fail(
                _failure_from_issue(size_issue, address=address, size=len(data))
            )

        response = self._backend.write_memory(self._process_handle, address, data)
        if response.error_code is not None:
            return WriteResult.fail(
                WriteFailure(
                    code="write_failed",
                    message=f"Backend write failed for 0x{address:X}.",
                    address=address,
                    size=len(data),
                    detail=f"error_code={response.error_code}",
                )
            )

        if response.bytes_written != len(data):
            return WriteResult.fail(
                WriteFailure(
                    code="short_write",
                    message=(
                        f"Short write for 0x{address:X}: requested {len(data)}, "
                        f"wrote {response.bytes_written}."
                    ),
                    address=address,
                    size=len(data),
                )
            )

        if response.flush_error_code is not None:
            return WriteResult.fail(
                WriteFailure(
                    code="flush_instruction_cache_failed",
                    message=f"FlushInstructionCache failed for 0x{address:X}.",
                    address=address,
                    size=len(data),
                    detail=f"error_code={response.flush_error_code}",
                )
            )

        return WriteResult.ok()

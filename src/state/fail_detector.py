"""Fail-state detection utilities."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

from src.config.offsets import OffsetEntry
from src.memory.pointer_chain import ModuleBaseResolver, resolve_offset_entry_address
from src.memory.reader import ProcessMemoryReader

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailDetectionResult:
    """Result from one fail-state check."""

    is_terminal: bool
    reason: str
    source: str
    timestamp_utc: str
    value: Any | None = None
    address: int | None = None
    error: str | None = None


class FallbackFailDetector(Protocol):
    """Fallback fail detector contract (e.g. UI/pixel-based)."""

    def check(self) -> FailDetectionResult:
        """Return terminal status for one fallback check."""


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_entry_value(reader: ProcessMemoryReader, entry: OffsetEntry, address: int) -> tuple[Any | None, str | None]:
    normalized_type = entry.data_type.strip().lower()
    if normalized_type == "bool":
        result = reader.read_bool(address)
    elif normalized_type == "int32":
        result = reader.read_int32(address)
    elif normalized_type == "int64":
        result = reader.read_int64(address)
    elif normalized_type == "uint64":
        result = reader.read_uint64(address)
    elif normalized_type == "uint32":
        raw = reader.read_bytes(address, 4)
        if not raw.is_ok:
            code = raw.error.code if raw.error is not None else "unknown_read_failure"
            return (None, code)
        return (int.from_bytes(raw.value or b"\x00\x00\x00\x00", "little", signed=False), None)
    elif normalized_type == "float":
        result = reader.read_float32(address)
    else:
        return (None, f"unsupported_fail_data_type:{entry.data_type}")

    if not result.is_ok:
        code = result.error.code if result.error is not None else "unknown_read_failure"
        return (None, code)
    return (result.value, None)


class MemoryFailDetector:
    """Primary fail detector that reads fail status from memory."""

    def __init__(
        self,
        *,
        reader: ProcessMemoryReader,
        fail_entry: OffsetEntry,
        module_base_resolver: ModuleBaseResolver | None = None,
        fallback_detectors: Sequence[FallbackFailDetector] = (),
        is_terminal_value: Callable[[Any], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._reader = reader
        self._fail_entry = fail_entry
        self._module_base_resolver = module_base_resolver
        self._fallback_detectors = tuple(fallback_detectors)
        self._is_terminal_value = is_terminal_value or bool
        self._logger = logger or LOGGER

    def check(self) -> FailDetectionResult:
        """Evaluate fail state once, with memory as primary and optional fallbacks."""
        timestamp = _now_iso_utc()
        resolve_result = resolve_offset_entry_address(
            reader=self._reader,
            entry=self._fail_entry,
            module_base_resolver=self._module_base_resolver,
        )

        if not resolve_result.is_ok or resolve_result.value is None:
            resolve_error = (
                resolve_result.error.code if resolve_result.error is not None else "resolve_failed"
            )
            self._logger.warning(
                "Fail-state memory resolution failed for '%s': %s",
                self._fail_entry.name,
                resolve_error,
            )
            return self._fallback_or_default(
                timestamp=timestamp,
                memory_error=resolve_error,
            )

        resolved_address = resolve_result.value
        value, read_error = _read_entry_value(self._reader, self._fail_entry, resolved_address)
        if read_error is not None:
            self._logger.warning(
                "Fail-state read failed for '%s' at 0x%X: %s",
                self._fail_entry.name,
                resolved_address,
                read_error,
            )
            return self._fallback_or_default(
                timestamp=timestamp,
                memory_error=read_error,
                resolved_address=resolved_address,
            )

        if self._is_terminal_value(value):
            reason = f"memory:{self._fail_entry.name}"
            event = FailDetectionResult(
                is_terminal=True,
                reason=reason,
                source="memory",
                timestamp_utc=timestamp,
                value=value,
                address=resolved_address,
            )
            self._logger.info(
                "Terminal fail-state detected from memory field '%s' (value=%r, address=0x%X).",
                self._fail_entry.name,
                value,
                resolved_address,
            )
            return event

        return self._fallback_or_default(
            timestamp=timestamp,
            resolved_address=resolved_address,
            memory_value=value,
        )

    def _fallback_or_default(
        self,
        *,
        timestamp: str,
        memory_error: str | None = None,
        resolved_address: int | None = None,
        memory_value: Any | None = None,
    ) -> FailDetectionResult:
        for fallback in self._fallback_detectors:
            fallback_result = fallback.check()
            if fallback_result.is_terminal:
                self._logger.info(
                    "Terminal fail-state detected by fallback source '%s' (reason=%s).",
                    fallback_result.source,
                    fallback_result.reason,
                )
                return fallback_result

        if memory_error is not None:
            return FailDetectionResult(
                is_terminal=False,
                reason="memory_unavailable",
                source="memory",
                timestamp_utc=timestamp,
                value=memory_value,
                address=resolved_address,
                error=memory_error,
            )

        return FailDetectionResult(
            is_terminal=False,
            reason="not_failed",
            source="memory",
            timestamp_utc=timestamp,
            value=memory_value,
            address=resolved_address,
        )


def poll_for_fail_state(
    *,
    detector: MemoryFailDetector,
    poll_interval_seconds: float = 0.25,
    timeout_seconds: float | None = None,
    max_polls: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    logger: logging.Logger | None = None,
) -> FailDetectionResult:
    """Poll until terminal failure is detected or limits are reached."""
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be > 0.")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0 when provided.")
    if max_polls is not None and max_polls < 1:
        raise ValueError("max_polls must be >= 1 when provided.")

    active_logger = logger or LOGGER
    started = monotonic_fn()
    polls = 0
    last_result: FailDetectionResult | None = None

    while True:
        result = detector.check()
        last_result = result
        polls += 1
        if result.is_terminal:
            active_logger.info(
                "Fail-state polling detected terminal event after %s poll(s): %s.",
                polls,
                result.reason,
            )
            return result

        if max_polls is not None and polls >= max_polls:
            break
        if timeout_seconds is not None and (monotonic_fn() - started) >= timeout_seconds:
            break
        sleep_fn(poll_interval_seconds)

    active_logger.info("Fail-state polling exited without terminal event after %s poll(s).", polls)
    return last_result or FailDetectionResult(
        is_terminal=False,
        reason="no_polls",
        source="poller",
        timestamp_utc=_now_iso_utc(),
        error="poll_loop_did_not_execute",
    )

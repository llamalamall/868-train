"""Tests for fail-state detection and polling behavior."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

import pytest

from src.config.offsets import OffsetBase, OffsetEntry
from src.memory.reader import BackendReadResponse, ProcessMemoryReader
from src.state.fail_detector import (
    FailDetectionResult,
    MemoryFailDetector,
    poll_for_fail_state,
)


@dataclass
class FakeMemoryBackend:
    """Simple fake memory backend for fail-detector tests."""

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


class FixedFallbackDetector:
    """Fixed fallback detector used for deterministic tests."""

    def __init__(self, result: FailDetectionResult) -> None:
        self._result = result

    def check(self) -> FailDetectionResult:
        return self._result


def _fail_entry(*, data_type: str = "bool") -> OffsetEntry:
    return OffsetEntry(
        name="fail_state",
        data_type=data_type,
        base=OffsetBase(kind="absolute", value="0x200000"),
        pointer_chain=(),
        confidence="high",
        notes="test",
        read_offset=0,
    )


def _pointer_fail_entry(*, data_type: str = "int32") -> OffsetEntry:
    return OffsetEntry(
        name="player_health",
        data_type=data_type,
        base=OffsetBase(kind="absolute", value="0x210000"),
        pointer_chain=(0,),
        confidence="high",
        notes="test",
        read_offset=0,
    )


def test_memory_fail_detector_returns_terminal_when_memory_flag_is_true() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: b"\x01"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry())

    result = detector.check()

    assert result.is_terminal
    assert result.source == "memory"
    assert result.reason == "memory:fail_state"
    assert result.value is True
    assert result.address == 0x200000


def test_memory_fail_detector_uses_fallback_when_memory_unavailable() -> None:
    backend = FakeMemoryBackend(error_by_address={0x200000: 299})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    fallback = FixedFallbackDetector(
        FailDetectionResult(
            is_terminal=True,
            reason="ui:game-over-banner",
            source="fallback-ui",
            timestamp_utc="2026-01-01T00:00:00+00:00",
        )
    )
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry(), fallback_detectors=(fallback,))

    result = detector.check()

    assert result.is_terminal
    assert result.source == "fallback-ui"
    assert result.reason == "ui:game-over-banner"


def test_memory_fail_detector_allows_fallback_override_when_memory_value_is_not_terminal() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: b"\x00"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    fallback = FixedFallbackDetector(
        FailDetectionResult(
            is_terminal=True,
            reason="ui:game-over-banner",
            source="fallback-ui",
            timestamp_utc="2026-01-01T00:00:00+00:00",
        )
    )
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry(), fallback_detectors=(fallback,))

    result = detector.check()

    assert result.is_terminal
    assert result.source == "fallback-ui"
    assert result.reason == "ui:game-over-banner"


def test_memory_fail_detector_reads_non_boolean_fail_field() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: struct.pack("<i", 2)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry(data_type="int32"))

    result = detector.check()

    assert result.is_terminal
    assert result.value == 2


def test_memory_fail_detector_reports_unsupported_data_type_as_memory_unavailable() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: b"\x01"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry(data_type="string"))

    result = detector.check()

    assert not result.is_terminal
    assert result.reason == "memory_unavailable"
    assert result.error == "unsupported_fail_data_type:string"


def test_memory_fail_detector_does_not_warn_for_null_pointer_resolution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = FakeMemoryBackend(memory_by_address={0x210000: struct.pack("<Q", 0)})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_pointer_fail_entry())

    with caplog.at_level(logging.WARNING):
        result = detector.check()

    assert not result.is_terminal
    assert result.reason == "memory_unavailable"
    assert result.error == "null_pointer"
    assert "Fail-state memory resolution failed" not in caplog.text


def test_memory_fail_detector_warns_once_for_repeated_identical_read_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = FakeMemoryBackend(error_by_address={0x200000: 299})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry())

    with caplog.at_level(logging.WARNING):
        first = detector.check()
        second = detector.check()

    assert not first.is_terminal
    assert not second.is_terminal
    assert first.error == "read_failed"
    assert second.error == "read_failed"
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warning_messages) == 1
    assert "Fail-state read failed for 'fail_state'" in warning_messages[0]


def test_poll_for_fail_state_detects_terminal_within_interval() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: b"\x00"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry())
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            backend.memory_by_address[0x200000] = b"\x01"

    result = poll_for_fail_state(
        detector=detector,
        poll_interval_seconds=0.1,
        timeout_seconds=1.0,
        sleep_fn=fake_sleep,
    )

    assert result.is_terminal
    assert sleep_calls == [0.1]


def test_poll_for_fail_state_returns_last_non_terminal_result_when_limited() -> None:
    backend = FakeMemoryBackend(memory_by_address={0x200000: b"\x00"})
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    detector = MemoryFailDetector(reader=reader, fail_entry=_fail_entry())

    result = poll_for_fail_state(
        detector=detector,
        poll_interval_seconds=0.05,
        max_polls=2,
        sleep_fn=lambda _: None,
    )

    assert not result.is_terminal
    assert result.reason == "not_failed"

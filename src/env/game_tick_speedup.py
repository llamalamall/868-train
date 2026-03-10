"""Runtime game tick speedup patching helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

from src.memory.reader import ProcessMemoryReader
from src.memory.writer import ProcessMemoryWriter

LOGGER = logging.getLogger(__name__)

_DEFAULT_GAME_TICK_MS = 16
_TICK_INSTRUCTION_RVA = 0xB08C
_TICK_INSTRUCTION_BYTES = bytes.fromhex("83 05 0D DF 06 00 10")
_TICK_IMMEDIATE_OFFSET = 6


class _ByteReader(Protocol):
    def read_bytes(self, address: int, size: int): ...


class _ByteWriter(Protocol):
    def write_bytes(self, address: int, data: bytes): ...


@dataclass(frozen=True)
class GameTickPatchPoint:
    """Static patch metadata for the known game loop instruction."""

    instruction_rva: int = _TICK_INSTRUCTION_RVA
    instruction_bytes: bytes = _TICK_INSTRUCTION_BYTES
    immediate_offset: int = _TICK_IMMEDIATE_OFFSET

    @property
    def original_tick_ms(self) -> int:
        return int(self.instruction_bytes[self.immediate_offset])


class GameTickSpeedupPatcher:
    """Patches in-memory loop tick constant from 16ms down to a faster cadence."""

    def __init__(
        self,
        *,
        game_tick_ms: int,
        logger: logging.Logger | None = None,
        patch_point: GameTickPatchPoint = GameTickPatchPoint(),
        reader_factory: Callable[[int], _ByteReader] | None = None,
        writer_factory: Callable[[int], _ByteWriter] | None = None,
    ) -> None:
        self._logger = logger or LOGGER
        self._patch_point = patch_point
        self._reader_factory = reader_factory or (lambda handle: ProcessMemoryReader(process_handle=handle))
        self._writer_factory = writer_factory or (lambda handle: ProcessMemoryWriter(process_handle=handle))

        self._target_tick_ms = int(game_tick_ms)
        if self._target_tick_ms < 1 or self._target_tick_ms > self._patch_point.original_tick_ms:
            raise ValueError(
                f"game_tick_ms must be in range [1, {self._patch_point.original_tick_ms}]."
            )
        self._saved_original_tick_ms: int | None = None
        self._patched_once = False

    @property
    def enabled(self) -> bool:
        return self._target_tick_ms < self._patch_point.original_tick_ms

    @property
    def target_tick_ms(self) -> int:
        return self._target_tick_ms

    def _instruction_address(self, module_base: int) -> int:
        return int(module_base) + int(self._patch_point.instruction_rva)

    def apply(self, *, process_handle: int, module_base: int) -> bool:
        """Apply patch to a process. Returns True only when target value is active."""
        if not self.enabled:
            return False

        instruction_address = self._instruction_address(module_base)
        reader = self._reader_factory(int(process_handle))
        writer = self._writer_factory(int(process_handle))
        read_result = reader.read_bytes(instruction_address, len(self._patch_point.instruction_bytes))
        if not read_result.is_ok:
            self._logger.warning(
                "game_tick_speedup_unavailable reason=instruction_read_failed address=0x%X detail=%s",
                instruction_address,
                getattr(read_result.error, "detail", None),
            )
            return False

        current_instruction = bytes(read_result.value or b"")
        if len(current_instruction) != len(self._patch_point.instruction_bytes):
            self._logger.warning(
                "game_tick_speedup_unavailable reason=instruction_size_mismatch address=0x%X size=%s",
                instruction_address,
                len(current_instruction),
            )
            return False

        expected_prefix = self._patch_point.instruction_bytes[: self._patch_point.immediate_offset]
        if current_instruction[: self._patch_point.immediate_offset] != expected_prefix:
            self._logger.warning(
                "game_tick_speedup_unavailable reason=instruction_prefix_mismatch address=0x%X",
                instruction_address,
            )
            return False

        current_tick_ms = int(current_instruction[self._patch_point.immediate_offset])
        if current_tick_ms == self._target_tick_ms:
            self._patched_once = True
            return True

        if current_tick_ms != self._patch_point.original_tick_ms:
            self._logger.warning(
                "game_tick_speedup_unavailable reason=unexpected_original_tick address=0x%X value=%s",
                instruction_address,
                current_tick_ms,
            )
            return False

        self._saved_original_tick_ms = current_tick_ms
        immediate_address = instruction_address + self._patch_point.immediate_offset
        write_result = writer.write_bytes(immediate_address, bytes([self._target_tick_ms]))
        if not write_result.is_ok:
            self._logger.warning(
                "game_tick_speedup_unavailable reason=instruction_write_failed address=0x%X detail=%s",
                immediate_address,
                getattr(write_result.error, "detail", None),
            )
            return False

        self._patched_once = True
        self._logger.info(
            "game_tick_speedup_applied instruction=0x%X tick_ms=%s",
            instruction_address,
            self._target_tick_ms,
        )
        return True

    def restore(self, *, process_handle: int, module_base: int) -> bool:
        """Restore original tick immediate when previously patched."""
        if not self.enabled or not self._patched_once:
            return False

        original_tick_ms = self._saved_original_tick_ms
        if original_tick_ms is None:
            original_tick_ms = self._patch_point.original_tick_ms

        immediate_address = self._instruction_address(module_base) + self._patch_point.immediate_offset
        writer = self._writer_factory(int(process_handle))
        write_result = writer.write_bytes(immediate_address, bytes([int(original_tick_ms)]))
        if not write_result.is_ok:
            self._logger.warning(
                "game_tick_speedup_restore_failed address=0x%X detail=%s",
                immediate_address,
                getattr(write_result.error, "detail", None),
            )
            return False

        self._logger.info(
            "game_tick_speedup_restored immediate=0x%X tick_ms=%s",
            immediate_address,
            original_tick_ms,
        )
        return True

"""Runtime enemy suppression helpers for no-enemies training mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from src.memory.writer import ProcessMemoryWriter

LOGGER = logging.getLogger(__name__)


class _ByteWriter(Protocol):
    def write_bytes(self, address: int, data: bytes): ...


@dataclass(frozen=True)
class EnemyEntityLayout:
    """Memory layout metadata for the runtime entity table."""

    table_base_offset: int = 0x0C
    entity_stride: int = 0x44
    entity_count: int = 64
    active_flag_offset: int = 0x00
    player_slot: int = 0


class EnemySpawnSuppressor:
    """Clears active enemy entity slots to keep no-enemies mode stable."""

    def __init__(
        self,
        *,
        enabled: bool,
        logger: logging.Logger | None = None,
        layout: EnemyEntityLayout = EnemyEntityLayout(),
        writer_factory: Callable[[int], _ByteWriter] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._logger = logger or LOGGER
        self._layout = layout
        self._writer_factory = writer_factory or (lambda handle: ProcessMemoryWriter(process_handle=handle))
        self._warned_addresses: set[int] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def suppress(
        self,
        *,
        process_handle: int,
        map_root_address: int,
        slots: Iterable[int] | None = None,
    ) -> int:
        """Write 0 to entity active flags. Returns successful slot writes."""
        if not self._enabled:
            return 0

        writer = self._writer_factory(int(process_handle))
        suppressed = 0
        for slot in self._iter_target_slots(slots):
            active_address = (
                int(map_root_address)
                + int(self._layout.table_base_offset)
                + int(slot) * int(self._layout.entity_stride)
                + int(self._layout.active_flag_offset)
            )
            write_result = writer.write_bytes(active_address, b"\x00")
            if write_result.is_ok:
                suppressed += 1
                continue

            if active_address in self._warned_addresses:
                continue
            self._warned_addresses.add(active_address)
            self._logger.warning(
                "enemy_spawn_suppression_write_failed slot=%s address=0x%X detail=%s",
                slot,
                active_address,
                getattr(write_result.error, "detail", None),
            )
        return suppressed

    def _iter_target_slots(self, slots: Iterable[int] | None) -> tuple[int, ...]:
        candidates: Iterable[int]
        if slots is None:
            candidates = range(int(self._layout.entity_count))
        else:
            candidates = slots

        normalized: list[int] = []
        seen: set[int] = set()
        for slot_value in candidates:
            slot = int(slot_value)
            if slot == int(self._layout.player_slot):
                continue
            if slot < 0 or slot >= int(self._layout.entity_count):
                continue
            if slot in seen:
                continue
            seen.add(slot)
            normalized.append(slot)

        return tuple(normalized)

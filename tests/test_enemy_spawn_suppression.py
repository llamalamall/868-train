"""Tests for no-enemies runtime suppression writes."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.env.enemy_spawn_suppression import EnemySpawnSuppressor
from src.memory.writer import WriteFailure, WriteResult


@dataclass
class FakeWriter:
    fail_addresses: set[int] = field(default_factory=set)
    writes: list[tuple[int, bytes]] = field(default_factory=list)

    def write_bytes(self, address: int, data: bytes) -> WriteResult:
        self.writes.append((int(address), bytes(data)))
        if int(address) in self.fail_addresses:
            return WriteResult.fail(
                WriteFailure(
                    code="write_failed",
                    message="forced write failure",
                    address=int(address),
                    size=len(data),
                )
            )
        return WriteResult.ok()


def _slot_active_address(*, map_root: int, slot: int) -> int:
    return int(map_root) + 0x0C + int(slot) * 0x44


def test_enemy_spawn_suppressor_noops_when_disabled() -> None:
    writer_factory_calls: list[int] = []

    def _writer_factory(handle: int) -> FakeWriter:
        writer_factory_calls.append(int(handle))
        return FakeWriter()

    suppressor = EnemySpawnSuppressor(
        enabled=False,
        writer_factory=_writer_factory,
    )

    suppressed = suppressor.suppress(
        process_handle=1234,
        map_root_address=0x1000,
    )

    assert suppressed == 0
    assert writer_factory_calls == []


def test_enemy_spawn_suppressor_targets_only_valid_non_player_slots() -> None:
    writer = FakeWriter()
    suppressor = EnemySpawnSuppressor(
        enabled=True,
        writer_factory=lambda _handle: writer,
    )

    suppressed = suppressor.suppress(
        process_handle=555,
        map_root_address=0x2000,
        slots=(0, 1, 1, 2, -1, 64),
    )

    assert suppressed == 2
    assert writer.writes == [
        (_slot_active_address(map_root=0x2000, slot=1), b"\x00"),
        (_slot_active_address(map_root=0x2000, slot=2), b"\x00"),
    ]


def test_enemy_spawn_suppressor_clears_all_enemy_slots_when_slots_unspecified() -> None:
    writer = FakeWriter()
    suppressor = EnemySpawnSuppressor(
        enabled=True,
        writer_factory=lambda _handle: writer,
    )

    suppressed = suppressor.suppress(
        process_handle=777,
        map_root_address=0x3000,
        slots=None,
    )

    assert suppressed == 63
    assert len(writer.writes) == 63
    assert writer.writes[0] == (_slot_active_address(map_root=0x3000, slot=1), b"\x00")
    assert writer.writes[-1] == (_slot_active_address(map_root=0x3000, slot=63), b"\x00")


def test_enemy_spawn_suppressor_logs_write_failure_once_per_address(caplog) -> None:
    failing_address = _slot_active_address(map_root=0x4000, slot=2)
    writer = FakeWriter(fail_addresses={failing_address})
    suppressor = EnemySpawnSuppressor(
        enabled=True,
        writer_factory=lambda _handle: writer,
    )

    with caplog.at_level("WARNING"):
        first = suppressor.suppress(
            process_handle=888,
            map_root_address=0x4000,
            slots=(2,),
        )
        second = suppressor.suppress(
            process_handle=888,
            map_root_address=0x4000,
            slots=(2,),
        )

    assert first == 0
    assert second == 0
    assert caplog.text.count("enemy_spawn_suppression_write_failed") == 1

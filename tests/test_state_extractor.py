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


def _chained_entry(
    name: str,
    data_type: str,
    *,
    base_address: int,
    pointer_chain: tuple[int, ...],
    read_offset: int,
) -> OffsetEntry:
    return OffsetEntry(
        name=name,
        data_type=data_type,
        base=OffsetBase(kind="absolute", value=f"0x{base_address:X}"),
        pointer_chain=pointer_chain,
        confidence="high",
        notes="test",
        read_offset=read_offset,
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


def test_extract_state_decodes_optional_action_availability_fields() -> None:
    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _entry("can_siphon_now", "bool", 0x20000C),
            _entry("prog_slots_available_mask", "uint32", 0x200010),
        ),
    )
    backend = FakeMemoryBackend(
        memory_by_address={
            0x200000: struct.pack("<i", 12),
            0x200004: struct.pack("<i", 5),
            0x200008: struct.pack("<i", 41),
            0x20000C: b"\x01",
            0x200010: struct.pack("<I", 0b101),
        }
    )
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.can_siphon_now is True
    assert snapshot.prog_slots_available_mask == 0b101


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


def test_extract_state_ignores_explicit_fail_state_and_uses_health_rule() -> None:
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
    assert snapshot.fail_state.value is False
    assert snapshot.fail_state.source_field == "derived:player_health==-1"


def test_extract_state_ignores_run_active_and_uses_health_rule() -> None:
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
    assert snapshot.fail_state.value is False
    assert snapshot.fail_state.source_field == "derived:player_health==-1"


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


def test_extract_state_decodes_map_cells_siphons_walls_resources_exit_and_enemies() -> None:
    root = 0x500000
    pointer_base = 0x100000
    player_x_offset = 0x19C8
    cell_base_offset = 0x11B8
    cell_stride = 0x38
    entity_base_offset = 0x0C
    entity_stride = 0x44

    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x200000),
            _entry("player_energy", "int32", 0x200004),
            _entry("player_credits", "int32", 0x200008),
            _chained_entry(
                "player_x",
                "int32",
                base_address=pointer_base,
                pointer_chain=(0,),
                read_offset=player_x_offset,
            ),
        ),
    )

    memory: dict[int, bytes] = {
        0x200000: struct.pack("<i", 10),
        0x200004: struct.pack("<i", 6),
        0x200008: struct.pack("<i", 21),
        pointer_base: struct.pack("<Q", root),
        root + player_x_offset: struct.pack("<i", 2),
    }

    # Initialize all map cells with deterministic defaults.
    default_cell_values = {
        0x00: 0,   # type
        0x04: 0,   # seed/variant A
        0x08: 0,   # variant B
        0x0C: 0,   # credits
        0x10: 0,   # energy
        0x14: -1,  # prog id
        0x18: 0,   # wall state
        0x1C: 0,   # threat
        0x20: 0,   # points
        0x24: 0,   # siphon flag
        0x28: 0,   # special state
        0x2C: 0,   # exit overlay
        0x30: 0,   # lock/hidden
        0x34: 0,   # marker
    }
    for index in range(36):
        base = root + cell_base_offset + index * cell_stride
        for offset, value in default_cell_values.items():
            memory[base + offset] = struct.pack("<i", value)

    def _cell_index(x: int, y: int) -> int:
        return x * 6 + y

    def _write_cell(x: int, y: int, values: dict[int, int]) -> None:
        base = root + cell_base_offset + _cell_index(x, y) * cell_stride
        for offset, value in values.items():
            memory[base + offset] = struct.pack("<i", value)

    # Non-wall resource cell.
    _write_cell(0, 1, {0x0C: 3, 0x10: 2, 0x20: 1})
    # Prog wall with one prog and threat metadata.
    _write_cell(1, 2, {0x00: 1, 0x14: 4, 0x18: 2, 0x1C: 5})
    # Points wall.
    _write_cell(2, 3, {0x00: 2, 0x20: 7, 0x18: 1})
    # Siphon marker.
    _write_cell(4, 1, {0x24: 1})
    # Exit.
    _write_cell(5, 5, {0x00: 3})

    # Initialize entity table as inactive.
    for slot in range(64):
        memory[root + entity_base_offset + slot * entity_stride] = b"\x00"

    # Entity slot 0 is the player character.
    player0 = root + entity_base_offset
    memory[player0] = b"\x01"
    memory[player0 + 0x08] = struct.pack("<i", 2)
    memory[player0 + 0x0C] = struct.pack("<i", 5)
    memory[player0 + 0x18] = struct.pack("<i", 1)
    memory[player0 + 0x34] = struct.pack("<i", 3)
    memory[player0 + 0x38] = struct.pack("<i", 4)

    # Active enemy 1 (out-of-bounds, still tracked).
    enemy1 = root + entity_base_offset + entity_stride
    memory[enemy1] = b"\x01"
    memory[enemy1 + 0x08] = struct.pack("<i", 7)
    memory[enemy1 + 0x0C] = struct.pack("<i", 2)
    memory[enemy1 + 0x18] = struct.pack("<i", 0)
    memory[enemy1 + 0x34] = struct.pack("<i", 8)
    memory[enemy1 + 0x38] = struct.pack("<i", 1)

    backend = FakeMemoryBackend(memory_by_address=memory)
    reader = ProcessMemoryReader(process_handle=1, backend=backend)

    snapshot = extract_state(reader=reader, registry=registry)

    assert snapshot.map.status == "ok"
    assert snapshot.map.address == root
    assert len(snapshot.map.cells) == 36
    assert tuple((pos.x, pos.y) for pos in snapshot.map.siphons) == ((4, 1),)
    assert snapshot.map.exit_position is not None
    assert (snapshot.map.exit_position.x, snapshot.map.exit_position.y) == (5, 5)

    walls_by_pos = {(wall.position.x, wall.position.y): wall for wall in snapshot.map.walls}
    assert walls_by_pos[(1, 2)].wall_type == "prog_wall"
    assert walls_by_pos[(1, 2)].prog_id == 4
    assert walls_by_pos[(2, 3)].wall_type == "point_wall"
    assert walls_by_pos[(2, 3)].points == 7

    resource_by_pos = {
        (resource.position.x, resource.position.y): resource for resource in snapshot.map.resource_cells
    }
    assert resource_by_pos[(0, 1)].credits == 3
    assert resource_by_pos[(0, 1)].energy == 2
    assert resource_by_pos[(0, 1)].points == 1

    assert snapshot.map.player_position is not None
    assert (snapshot.map.player_position.x, snapshot.map.player_position.y) == (3, 4)

    assert len(snapshot.map.enemies) == 1
    assert snapshot.map.enemies[0].type_id == 0
    assert (snapshot.map.enemies[0].position.x, snapshot.map.enemies[0].position.y) == (8, 1)
    assert snapshot.map.enemies[0].in_bounds is False

    layers = snapshot.map.layers
    assert layers.obstacle_map[2][1] == 1  # prog wall
    assert layers.obstacle_map[3][2] == 1  # points wall
    assert layers.player_position_map[4][3] == 1
    assert layers.goal_map[1][4] == 2  # siphon priority
    assert layers.goal_map[5][5] == 1  # exit priority
    assert layers.credits_map[1][0] == 3
    assert layers.energy_map[1][0] == 2
    assert layers.progs_map[1][1] == (4,)
    assert layers.points_map[2][2] == 7
    assert layers.siphon_penalty_map[1][1] == 5


def test_extract_state_layer_refresh_reuses_static_layers_when_only_player_moves() -> None:
    root = 0x600000
    pointer_base = 0x100100
    player_x_offset = 0x19C8
    cell_base_offset = 0x11B8
    cell_stride = 0x38
    entity_base_offset = 0x0C
    entity_stride = 0x44

    registry = OffsetRegistry(
        version=1,
        entries=(
            _entry("player_health", "int32", 0x210000),
            _entry("player_energy", "int32", 0x210004),
            _entry("player_credits", "int32", 0x210008),
            _chained_entry(
                "player_x",
                "int32",
                base_address=pointer_base,
                pointer_chain=(0,),
                read_offset=player_x_offset,
            ),
        ),
    )

    memory: dict[int, bytes] = {
        0x210000: struct.pack("<i", 10),
        0x210004: struct.pack("<i", 6),
        0x210008: struct.pack("<i", 21),
        pointer_base: struct.pack("<Q", root),
        root + player_x_offset: struct.pack("<i", 2),
    }

    default_cell_values = {
        0x00: 0,
        0x04: 0,
        0x08: 0,
        0x0C: 0,
        0x10: 0,
        0x14: -1,
        0x18: 0,
        0x1C: 0,
        0x20: 0,
        0x24: 0,
        0x28: 0,
        0x2C: 0,
        0x30: 0,
        0x34: 0,
    }
    for index in range(36):
        base = root + cell_base_offset + index * cell_stride
        for offset, value in default_cell_values.items():
            memory[base + offset] = struct.pack("<i", value)

    def _cell_index(x: int, y: int) -> int:
        return x * 6 + y

    def _write_cell(x: int, y: int, values: dict[int, int]) -> None:
        base = root + cell_base_offset + _cell_index(x, y) * cell_stride
        for offset, value in values.items():
            memory[base + offset] = struct.pack("<i", value)

    _write_cell(0, 1, {0x0C: 2, 0x10: 1})
    _write_cell(1, 2, {0x00: 1, 0x14: 4, 0x1C: 3})
    _write_cell(4, 1, {0x24: 1})
    _write_cell(5, 5, {0x00: 3})

    for slot in range(64):
        memory[root + entity_base_offset + slot * entity_stride] = b"\x00"

    player0 = root + entity_base_offset
    memory[player0] = b"\x01"
    memory[player0 + 0x08] = struct.pack("<i", 2)
    memory[player0 + 0x0C] = struct.pack("<i", 5)
    memory[player0 + 0x18] = struct.pack("<i", 1)
    memory[player0 + 0x34] = struct.pack("<i", 1)
    memory[player0 + 0x38] = struct.pack("<i", 1)

    backend = FakeMemoryBackend(memory_by_address=memory)
    reader = ProcessMemoryReader(process_handle=1, backend=backend)
    first_snapshot = extract_state(reader=reader, registry=registry)

    memory[player0 + 0x34] = struct.pack("<i", 2)
    second_snapshot = extract_state(
        reader=reader,
        registry=registry,
        previous_map_state=first_snapshot.map,
    )

    assert second_snapshot.map.layer_refresh.obstacles_updated is False
    assert second_snapshot.map.layer_refresh.siphon_outcomes_updated is False
    assert second_snapshot.map.layer_refresh.player_and_enemy_updated is True
    assert second_snapshot.map.layer_refresh.goals_updated is True
    assert second_snapshot.map.layers.obstacle_map is first_snapshot.map.layers.obstacle_map
    assert second_snapshot.map.layers.energy_map is first_snapshot.map.layers.energy_map

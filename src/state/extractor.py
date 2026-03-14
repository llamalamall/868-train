"""Normalized game-state extraction from configured memory offsets."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from src.config.offsets import OffsetEntry, OffsetRegistry
from src.memory.pointer_chain import ModuleBaseResolver, resolve_offset_entry_address
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult
from src.state.prog_catalog import PROG_NAME_BY_ID
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapLayerRefreshState,
    MapLayersState,
    MapCellState,
    MapState,
    ProgInventoryItem,
    ResourceCellState,
    WallCellState,
)

LOGGER = logging.getLogger(__name__)

_HEALTH_FIELD_CANDIDATES = ("player_health", "health")
_ENERGY_FIELD_CANDIDATES = ("player_energy", "energy")
_CURRENCY_FIELD_CANDIDATES = ("player_credits", "player_currency", "currency")
_COLLECTED_PROGS_FIELD_CANDIDATES = ("collected_progs",)
_CAN_SIPHON_NOW_FIELD_CANDIDATES = (
    "can_siphon_now",
    "siphon_prompt_active",
    "siphon_available",
)
_PROG_SLOTS_AVAILABLE_MASK_FIELD_CANDIDATES = (
    "prog_slots_available_mask",
    "prog_slot_available_mask",
    "available_prog_slots_mask",
)
_GAME_STATE_ROOT_FIELD_CANDIDATES = (
    "player_x",
    "player_y",
    "player_energy",
    "player_credits",
    "score",
    "run_active",
    "collected_progs",
    "siphons",
)
_MAX_VECTOR_INT32_ITEMS = 4096
_MAP_WIDTH = 6
_MAP_HEIGHT = 6
_MAP_CELL_COUNT = _MAP_WIDTH * _MAP_HEIGHT
_MAP_CELLS_BASE_OFFSET = 0x11B8
_MAP_CELL_STRIDE = 0x38
_MAP_ENTITIES_BASE_OFFSET = 0x0C
_MAP_ENTITY_STRIDE = 0x44
_MAP_ENTITY_COUNT = 64
_ENEMY_VISIBLE_INCUBATION_TIMER = 0
_CAN_SIPHON_NOW_DERIVED_OFFSET = 0x1BD2
_PROG_SLOTS_AVAILABLE_BASE_OFFSET = 0x1BC8
_PROG_SLOT_COUNT = 10
_SIPHON_REACH_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, 1),
    (0, -1),
)

_LOGGED_UNKNOWN_PROG_IDS: set[int] = set()


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _missing_field(error_code: str, *, source_field: str | None = None, error: str | None = None) -> FieldState:
    return FieldState(
        value=None,
        status="missing",
        error_code=error_code,
        error=error,
        source_field=source_field,
    )


def _invalid_field(
    error_code: str,
    *,
    address: int | None = None,
    source_field: str | None = None,
    error: str | None = None,
) -> FieldState:
    return FieldState(
        value=None,
        status="invalid",
        error_code=error_code,
        error=error,
        address=address,
        source_field=source_field,
    )


def _ok_field(value: Any, *, address: int | None, source_field: str) -> FieldState:
    return FieldState(
        value=value,
        status="ok",
        address=address,
        source_field=source_field,
    )


def _missing_inventory(
    error_code: str,
    *,
    source_field: str | None = None,
    error: str | None = None,
) -> InventoryState:
    return InventoryState(
        status="missing",
        error_code=error_code,
        error=error,
        source_field=source_field,
    )


def _invalid_inventory(
    error_code: str,
    *,
    source_field: str | None = None,
    address: int | None = None,
    error: str | None = None,
    raw_prog_ids: tuple[int, ...] = (),
) -> InventoryState:
    return InventoryState(
        status="invalid",
        error_code=error_code,
        error=error,
        source_field=source_field,
        address=address,
        raw_prog_ids=raw_prog_ids,
    )


def _ok_inventory(
    *,
    source_field: str,
    address: int | None,
    raw_prog_ids: tuple[int, ...],
    collected_progs: tuple[ProgInventoryItem, ...],
    unknown_prog_ids: tuple[int, ...],
) -> InventoryState:
    return InventoryState(
        status="ok",
        source_field=source_field,
        address=address,
        raw_prog_ids=raw_prog_ids,
        collected_progs=collected_progs,
        unknown_prog_ids=unknown_prog_ids,
    )


def _entry_by_name(registry: OffsetRegistry) -> dict[str, OffsetEntry]:
    return {entry.name: entry for entry in registry.entries}


def _select_entry(
    entries_by_name: dict[str, OffsetEntry],
    candidates: tuple[str, ...],
) -> OffsetEntry | None:
    for name in candidates:
        entry = entries_by_name.get(name)
        if entry is not None:
            return entry
    return None


def _read_uint32(reader: ProcessMemoryReader, address: int) -> ReadResult[int]:
    raw = reader.read_bytes(address, 4)
    if not raw.is_ok:
        return ReadResult.fail(
            raw.error or ReadFailure(code="read_failed", message="Read failed.", address=address, size=4)
        )
    return ReadResult.ok(int.from_bytes(raw.value or b"\x00\x00\x00\x00", "little", signed=False))


def _read_vector_int32(reader: ProcessMemoryReader, address: int) -> ReadResult[tuple[int, ...]]:
    begin_result = reader.read_pointer(address)
    if not begin_result.is_ok:
        return ReadResult.fail(
            begin_result.error
            or ReadFailure(
                code="vector_begin_read_failed",
                message="Failed reading vector begin pointer.",
                address=address,
                size=reader.pointer_size,
            )
        )
    end_address = address + reader.pointer_size
    end_result = reader.read_pointer(end_address)
    if not end_result.is_ok:
        return ReadResult.fail(
            end_result.error
            or ReadFailure(
                code="vector_end_read_failed",
                message="Failed reading vector end pointer.",
                address=end_address,
                size=reader.pointer_size,
            )
        )

    begin = int(begin_result.value or 0)
    end = int(end_result.value or 0)
    if begin == 0 and end == 0:
        return ReadResult.ok(())
    if begin == 0 or end == 0:
        return ReadResult.fail(
            ReadFailure(
                code="vector_null_boundary",
                message=f"Invalid vector boundary begin=0x{begin:X} end=0x{end:X}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )
    if end < begin:
        return ReadResult.fail(
            ReadFailure(
                code="vector_end_before_begin",
                message=f"Invalid vector range begin=0x{begin:X} end=0x{end:X}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )
    span = end - begin
    if span % 4 != 0:
        return ReadResult.fail(
            ReadFailure(
                code="vector_unaligned_span",
                message=f"Vector span {span} is not aligned to int32.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )

    count = span // 4
    if count > _MAX_VECTOR_INT32_ITEMS:
        return ReadResult.fail(
            ReadFailure(
                code="vector_too_large",
                message=f"Vector item count {count} exceeds limit {_MAX_VECTOR_INT32_ITEMS}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )

    values: list[int] = []
    for index in range(count):
        item_address = begin + index * 4
        item_result = reader.read_int32(item_address)
        if not item_result.is_ok:
            return ReadResult.fail(
                item_result.error
                or ReadFailure(
                    code="vector_item_read_failed",
                    message=f"Failed reading vector item at index {index}.",
                    address=item_address,
                    size=4,
                )
            )
        values.append(int(item_result.value or 0))
    return ReadResult.ok(tuple(values))


def _decode_value(reader: ProcessMemoryReader, entry: OffsetEntry, address: int) -> ReadResult[Any]:
    normalized = entry.data_type.strip().lower()
    if normalized == "int32":
        return reader.read_int32(address)
    if normalized == "int64":
        return reader.read_int64(address)
    if normalized == "uint32":
        return _read_uint32(reader, address)
    if normalized == "uint64":
        return reader.read_uint64(address)
    if normalized == "float":
        return reader.read_float32(address)
    if normalized == "bool":
        return reader.read_bool(address)
    if normalized == "array<int32>":
        return _read_vector_int32(reader, address)
    return ReadResult.fail(
        ReadFailure(
            code="unsupported_data_type",
            message=f"Unsupported data_type '{entry.data_type}'.",
            address=address,
        )
    )


def _extract_field(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> FieldState:
    if entry is None:
        return _missing_field("field_not_configured")

    resolved = resolve_offset_entry_address(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if not resolved.is_ok or resolved.value is None:
        error_code = resolved.error.code if resolved.error is not None else "resolve_failed"
        error_message = resolved.error.message if resolved.error is not None else "Address resolution failed."
        return _invalid_field(
            error_code,
            source_field=entry.name,
            error=error_message,
        )

    address = resolved.value
    decoded = _decode_value(reader, entry, address)
    if not decoded.is_ok:
        code = decoded.error.code if decoded.error is not None else "read_failed"
        message = decoded.error.message if decoded.error is not None else "Read failed."
        return _invalid_field(
            code,
            address=address,
            source_field=entry.name,
            error=message,
        )

    return _ok_field(decoded.value, address=address, source_field=entry.name)


def _extract_optional_bool(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> bool | None:
    field = _extract_field(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if field.status != "ok":
        return None

    value = field.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


def _extract_optional_uint32_mask(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> int | None:
    field = _extract_field(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if field.status != "ok":
        return None

    value = field.value
    if not isinstance(value, int):
        return None
    if value < 0:
        return None
    return int(value & 0xFFFFFFFF)


def _derive_optional_bool_from_root(
    *,
    reader: ProcessMemoryReader,
    root_address: int | None,
    offset: int,
) -> bool | None:
    if root_address is None:
        return None
    result = reader.read_bool(int(root_address) + int(offset))
    if not result.is_ok:
        return None
    return bool(result.value)


def _derive_prog_slots_available_mask_from_root(
    *,
    reader: ProcessMemoryReader,
    root_address: int | None,
) -> int | None:
    if root_address is None:
        return None
    mask = 0
    for slot_index in range(_PROG_SLOT_COUNT):
        result = reader.read_bool(int(root_address) + _PROG_SLOTS_AVAILABLE_BASE_OFFSET + slot_index)
        if not result.is_ok:
            return None
        if bool(result.value):
            mask |= 1 << slot_index
    return mask


def _log_unknown_prog_ids_once(unknown_prog_ids: tuple[int, ...]) -> None:
    for prog_id in unknown_prog_ids:
        if prog_id in _LOGGED_UNKNOWN_PROG_IDS:
            continue
        _LOGGED_UNKNOWN_PROG_IDS.add(prog_id)
        LOGGER.warning(
            "Unknown collected prog id detected: %s. Preserving id in snapshot for later mapping.",
            prog_id,
        )


def _build_prog_inventory(raw_prog_ids: tuple[int, ...]) -> tuple[tuple[ProgInventoryItem, ...], tuple[int, ...]]:
    counts = Counter(raw_prog_ids)
    seen: set[int] = set()
    ordered_ids: list[int] = []
    for prog_id in raw_prog_ids:
        if prog_id in seen:
            continue
        seen.add(prog_id)
        ordered_ids.append(prog_id)

    unknown: list[int] = []
    items: list[ProgInventoryItem] = []
    for prog_id in ordered_ids:
        name = PROG_NAME_BY_ID.get(prog_id)
        known = name is not None
        flags = () if known else ("unknown_id",)
        if not known:
            unknown.append(prog_id)
        items.append(
            ProgInventoryItem(
                prog_id=prog_id,
                count=counts[prog_id],
                name=name,
                known=known,
                flags=flags,
            )
        )
    unknown_ids = tuple(unknown)
    _log_unknown_prog_ids_once(unknown_ids)
    return (tuple(items), unknown_ids)


def _extract_inventory(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> InventoryState:
    if entry is None:
        return _missing_inventory("field_not_configured")

    resolved = resolve_offset_entry_address(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if not resolved.is_ok or resolved.value is None:
        error_code = resolved.error.code if resolved.error is not None else "resolve_failed"
        error_message = resolved.error.message if resolved.error is not None else "Address resolution failed."
        return _invalid_inventory(
            error_code,
            source_field=entry.name,
            error=error_message,
        )

    address = resolved.value
    decoded = _decode_value(reader, entry, address)
    if not decoded.is_ok:
        code = decoded.error.code if decoded.error is not None else "read_failed"
        message = decoded.error.message if decoded.error is not None else "Read failed."
        return _invalid_inventory(
            code,
            source_field=entry.name,
            address=address,
            error=message,
        )

    raw_prog_ids = tuple(int(value) for value in (decoded.value or ()))
    items, unknown_prog_ids = _build_prog_inventory(raw_prog_ids)
    return _ok_inventory(
        source_field=entry.name,
        address=address,
        raw_prog_ids=raw_prog_ids,
        collected_progs=items,
        unknown_prog_ids=unknown_prog_ids,
    )


def _missing_map(
    error_code: str,
    *,
    source_field: str | None = None,
    error: str | None = None,
) -> MapState:
    return MapState(
        status="missing",
        error_code=error_code,
        error=error,
        source_field=source_field,
    )


def _invalid_map(
    error_code: str,
    *,
    source_field: str | None = None,
    address: int | None = None,
    error: str | None = None,
) -> MapState:
    return MapState(
        status="invalid",
        error_code=error_code,
        error=error,
        source_field=source_field,
        address=address,
    )


def _ok_map(
    *,
    source_field: str,
    address: int,
    cells: tuple[MapCellState, ...],
    siphons: tuple[GridPosition, ...],
    walls: tuple[WallCellState, ...],
    resource_cells: tuple[ResourceCellState, ...],
    player_position: GridPosition | None,
    exit_position: GridPosition | None,
    enemies: tuple[EnemyState, ...],
    layers: MapLayersState,
    layer_refresh: MapLayerRefreshState,
) -> MapState:
    return MapState(
        status="ok",
        source_field=source_field,
        address=address,
        cells=cells,
        siphons=siphons,
        walls=walls,
        resource_cells=resource_cells,
        player_position=player_position,
        exit_position=exit_position,
        enemies=enemies,
        layers=layers,
        layer_refresh=layer_refresh,
    )


def _build_obstacle_signature(cells: tuple[MapCellState, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            (int(cell.position.x), int(cell.position.y))
            for cell in cells
            if cell.is_wall
        )
    )


def _build_goal_signature(
    *,
    siphons: tuple[GridPosition, ...],
    exit_position: GridPosition | None,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int] | None]:
    siphon_positions = tuple(sorted((int(position.x), int(position.y)) for position in siphons))
    exit_pos = (int(exit_position.x), int(exit_position.y)) if exit_position is not None else None
    return (siphon_positions, exit_pos)


def _build_enemy_signature(enemies: tuple[EnemyState, ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted(
            (
                int(enemy.position.x),
                int(enemy.position.y),
                int(enemy.type_id),
            )
            for enemy in enemies
            if enemy.in_bounds and int(enemy.type_id) > 0
        )
    )


def _build_siphon_outcome_signature(cells: tuple[MapCellState, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            int(cell.position.x),
            int(cell.position.y),
            1 if cell.is_wall else 0,
            int(cell.wall_state),
            max(int(cell.credits), 0),
            max(int(cell.energy), 0),
            max(int(cell.points), 0),
            int(cell.prog_id) if cell.prog_id is not None else -1,
            max(int(cell.threat), 0),
        )
        for cell in cells
    )


def _empty_int_grid(*, width: int, height: int) -> list[list[int]]:
    return [[0 for _ in range(width)] for _ in range(height)]


def _empty_prog_grid(*, width: int, height: int) -> list[list[list[int]]]:
    return [[[] for _ in range(width)] for _ in range(height)]


def _freeze_int_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def _freeze_prog_grid(grid: list[list[list[int]]]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(int(prog_id) for prog_id in cell) for cell in row) for row in grid)


def _int_grid_matches_dimensions(
    grid: tuple[tuple[int, ...], ...],
    *,
    width: int,
    height: int,
) -> bool:
    return len(grid) == height and all(len(row) == width for row in grid)


def _nested_grid_matches_dimensions(
    grid: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    width: int,
    height: int,
) -> bool:
    return len(grid) == height and all(len(row) == width for row in grid)


def _layer_dimensions_match(
    *,
    layers: MapLayersState,
    width: int,
    height: int,
) -> bool:
    return (
        _int_grid_matches_dimensions(layers.obstacle_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.player_position_map, width=width, height=height)
        and _nested_grid_matches_dimensions(layers.enemy_position_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.goal_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.energy_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.credits_map, width=width, height=height)
        and _nested_grid_matches_dimensions(layers.progs_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.points_map, width=width, height=height)
        and _int_grid_matches_dimensions(layers.siphon_penalty_map, width=width, height=height)
    )


def _build_obstacle_map(
    *,
    width: int,
    height: int,
    cells: tuple[MapCellState, ...],
) -> tuple[tuple[int, ...], ...]:
    obstacle_grid = _empty_int_grid(width=width, height=height)
    for cell in cells:
        x = int(cell.position.x)
        y = int(cell.position.y)
        if 0 <= x < width and 0 <= y < height and cell.is_wall:
            obstacle_grid[y][x] = 1
    return _freeze_int_grid(obstacle_grid)


def _build_player_position_map(
    *,
    width: int,
    height: int,
    player_position: GridPosition | None,
) -> tuple[tuple[int, ...], ...]:
    player_grid = _empty_int_grid(width=width, height=height)
    if player_position is None:
        return _freeze_int_grid(player_grid)
    x = int(player_position.x)
    y = int(player_position.y)
    if 0 <= x < width and 0 <= y < height:
        player_grid[y][x] = 1
    return _freeze_int_grid(player_grid)


def _build_enemy_position_map(
    *,
    width: int,
    height: int,
    enemies: tuple[EnemyState, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    enemy_grid = _empty_prog_grid(width=width, height=height)
    for enemy in enemies:
        enemy_type = int(enemy.type_id)
        x = int(enemy.position.x)
        y = int(enemy.position.y)
        if not enemy.in_bounds or enemy_type <= 0:
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        enemy_grid[y][x].append(enemy_type)
    for row in enemy_grid:
        for cell in row:
            cell.sort()
    return _freeze_prog_grid(enemy_grid)


def _build_goal_map(
    *,
    width: int,
    height: int,
    siphons: tuple[GridPosition, ...],
    exit_position: GridPosition | None,
) -> tuple[tuple[int, ...], ...]:
    goal_grid = _empty_int_grid(width=width, height=height)
    if exit_position is not None:
        exit_x = int(exit_position.x)
        exit_y = int(exit_position.y)
        if 0 <= exit_x < width and 0 <= exit_y < height:
            goal_grid[exit_y][exit_x] = 1
    for siphon in siphons:
        siphon_x = int(siphon.x)
        siphon_y = int(siphon.y)
        if 0 <= siphon_x < width and 0 <= siphon_y < height:
            goal_grid[siphon_y][siphon_x] = 2
    return _freeze_int_grid(goal_grid)


def _build_siphon_outcome_layers(
    *,
    width: int,
    height: int,
    cells: tuple[MapCellState, ...],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    cell_by_pos = {(int(cell.position.x), int(cell.position.y)): cell for cell in cells}
    energy_grid = _empty_int_grid(width=width, height=height)
    credits_grid = _empty_int_grid(width=width, height=height)
    points_grid = _empty_int_grid(width=width, height=height)
    penalty_grid = _empty_int_grid(width=width, height=height)
    progs_grid = _empty_prog_grid(width=width, height=height)

    for y in range(height):
        for x in range(width):
            for dx, dy in _SIPHON_REACH_DELTAS:
                target_x = x + dx
                target_y = y + dy
                if target_x < 0 or target_y < 0 or target_x >= width or target_y >= height:
                    continue
                target_cell = cell_by_pos.get((target_x, target_y))
                if target_cell is None:
                    continue
                if target_cell.is_wall:
                    points_grid[y][x] += max(int(target_cell.points), 0)
                    penalty_grid[y][x] += max(int(target_cell.threat), 0)
                    prog_id = int(target_cell.prog_id) if target_cell.prog_id is not None else -1
                    if prog_id > 0 and int(target_cell.wall_state) > 0:
                        progs_grid[y][x].append(prog_id)
                    continue
                credits_grid[y][x] += max(int(target_cell.credits), 0)
                energy_grid[y][x] += max(int(target_cell.energy), 0)

    for row in progs_grid:
        for cell in row:
            cell.sort()

    return (
        _freeze_int_grid(energy_grid),
        _freeze_int_grid(credits_grid),
        _freeze_prog_grid(progs_grid),
        _freeze_int_grid(points_grid),
        _freeze_int_grid(penalty_grid),
    )


def _build_map_layers(
    *,
    width: int,
    height: int,
    cells: tuple[MapCellState, ...],
    siphons: tuple[GridPosition, ...],
    player_position: GridPosition | None,
    exit_position: GridPosition | None,
    enemies: tuple[EnemyState, ...],
    previous_map_state: MapState | None,
) -> tuple[MapLayersState, MapLayerRefreshState]:
    reusable_previous = (
        previous_map_state is not None
        and previous_map_state.status == "ok"
        and int(previous_map_state.width) == width
        and int(previous_map_state.height) == height
        and _layer_dimensions_match(layers=previous_map_state.layers, width=width, height=height)
    )

    previous_layers = previous_map_state.layers if reusable_previous and previous_map_state is not None else None
    previous_cells = previous_map_state.cells if reusable_previous and previous_map_state is not None else ()
    previous_siphons = previous_map_state.siphons if reusable_previous and previous_map_state is not None else ()
    previous_exit = previous_map_state.exit_position if reusable_previous and previous_map_state is not None else None
    previous_enemies = previous_map_state.enemies if reusable_previous and previous_map_state is not None else ()
    previous_player = previous_map_state.player_position if reusable_previous and previous_map_state is not None else None

    player_moved = previous_player != player_position
    obstacle_signature = _build_obstacle_signature(cells)
    previous_obstacle_signature = _build_obstacle_signature(previous_cells)
    goals_signature = _build_goal_signature(siphons=siphons, exit_position=exit_position)
    previous_goals_signature = _build_goal_signature(siphons=previous_siphons, exit_position=previous_exit)
    enemy_signature = _build_enemy_signature(enemies)
    previous_enemy_signature = _build_enemy_signature(previous_enemies)
    siphon_outcome_signature = _build_siphon_outcome_signature(cells)
    previous_siphon_outcome_signature = _build_siphon_outcome_signature(previous_cells)

    obstacles_updated = (
        not reusable_previous
        or obstacle_signature != previous_obstacle_signature
    )
    goals_updated = (
        not reusable_previous
        or player_moved
        or goals_signature != previous_goals_signature
    )
    player_and_enemy_updated = (
        not reusable_previous
        or player_moved
        or enemy_signature != previous_enemy_signature
    )
    siphon_outcomes_updated = (
        not reusable_previous
        or siphon_outcome_signature != previous_siphon_outcome_signature
    )

    obstacle_map = (
        _build_obstacle_map(width=width, height=height, cells=cells)
        if obstacles_updated or previous_layers is None
        else previous_layers.obstacle_map
    )
    player_position_map = _build_player_position_map(
        width=width,
        height=height,
        player_position=player_position,
    )
    enemy_position_map = _build_enemy_position_map(
        width=width,
        height=height,
        enemies=enemies,
    )
    goal_map = (
        _build_goal_map(
            width=width,
            height=height,
            siphons=siphons,
            exit_position=exit_position,
        )
        if goals_updated or previous_layers is None
        else previous_layers.goal_map
    )
    if siphon_outcomes_updated or previous_layers is None:
        (
            energy_map,
            credits_map,
            progs_map,
            points_map,
            siphon_penalty_map,
        ) = _build_siphon_outcome_layers(
            width=width,
            height=height,
            cells=cells,
        )
    else:
        energy_map = previous_layers.energy_map
        credits_map = previous_layers.credits_map
        progs_map = previous_layers.progs_map
        points_map = previous_layers.points_map
        siphon_penalty_map = previous_layers.siphon_penalty_map

    return (
        MapLayersState(
            obstacle_map=obstacle_map,
            player_position_map=player_position_map,
            enemy_position_map=enemy_position_map,
            goal_map=goal_map,
            energy_map=energy_map,
            credits_map=credits_map,
            progs_map=progs_map,
            points_map=points_map,
            siphon_penalty_map=siphon_penalty_map,
        ),
        MapLayerRefreshState(
            obstacles_updated=obstacles_updated,
            player_and_enemy_updated=player_and_enemy_updated,
            goals_updated=goals_updated,
            siphon_outcomes_updated=siphon_outcomes_updated,
        ),
    )


def _resolve_game_state_root(
    *,
    reader: ProcessMemoryReader,
    entries_by_name: dict[str, OffsetEntry],
    module_base_resolver: ModuleBaseResolver | None,
) -> tuple[int | None, str | None, ReadFailure | None]:
    first_failure: ReadFailure | None = None
    found_candidate = False

    for name in _GAME_STATE_ROOT_FIELD_CANDIDATES:
        entry = entries_by_name.get(name)
        if entry is None:
            continue
        # We can only back-calculate root reliably from pointer-chain derived fields.
        if not entry.pointer_chain:
            continue
        found_candidate = True

        resolved = resolve_offset_entry_address(
            reader=reader,
            entry=entry,
            module_base_resolver=module_base_resolver,
        )
        if not resolved.is_ok or resolved.value is None:
            if first_failure is None and resolved.error is not None:
                first_failure = resolved.error
            continue

        root_address = resolved.value - entry.read_offset
        if root_address < 0:
            if first_failure is None:
                first_failure = ReadFailure(
                    code="invalid_root_address",
                    message=f"Derived negative root address from '{entry.name}'.",
                    address=resolved.value,
                )
            continue
        return (root_address, entry.name, None)

    if not found_candidate:
        return (None, None, None)
    return (None, None, first_failure)


def _read_required_int32(reader: ProcessMemoryReader, address: int) -> ReadResult[int]:
    result = reader.read_int32(address)
    if not result.is_ok:
        return ReadResult.fail(
            result.error
            or ReadFailure(code="read_failed", message="Failed reading int32.", address=address, size=4)
        )
    return ReadResult.ok(int(result.value or 0))


def _read_required_bool(reader: ProcessMemoryReader, address: int) -> ReadResult[bool]:
    result = reader.read_bool(address)
    if not result.is_ok:
        return ReadResult.fail(
            result.error
            or ReadFailure(code="read_failed", message="Failed reading bool.", address=address, size=1)
        )
    return ReadResult.ok(bool(result.value))


def _mask_enemy_type_for_visibility(*, raw_type_id: int, incubation_timer: int, in_bounds: bool) -> int:
    """Hide enemy type for off-board or incubating entities."""
    if raw_type_id <= 0:
        return 0
    if not in_bounds:
        return 0
    if incubation_timer > _ENEMY_VISIBLE_INCUBATION_TIMER:
        return 0
    return raw_type_id


def _extract_map_state(
    *,
    reader: ProcessMemoryReader,
    entries_by_name: dict[str, OffsetEntry],
    module_base_resolver: ModuleBaseResolver | None,
    previous_map_state: MapState | None = None,
) -> MapState:
    root_address, source_field, root_failure = _resolve_game_state_root(
        reader=reader,
        entries_by_name=entries_by_name,
        module_base_resolver=module_base_resolver,
    )
    if root_address is None:
        if root_failure is None:
            return _missing_map(
                "field_not_configured",
                error="No pointer-chain field available to derive game-state root.",
            )
        return _invalid_map(
            root_failure.code,
            source_field=source_field,
            address=root_failure.address,
            error=root_failure.message,
        )

    cell_states: list[MapCellState] = []
    siphons: list[GridPosition] = []
    walls: list[WallCellState] = []
    resource_cells: list[ResourceCellState] = []
    player_position: GridPosition | None = None
    exit_position: GridPosition | None = None

    for index in range(_MAP_CELL_COUNT):
        x = index // _MAP_HEIGHT
        y = index % _MAP_HEIGHT
        position = GridPosition(x=x, y=y)
        cell_base = root_address + _MAP_CELLS_BASE_OFFSET + index * _MAP_CELL_STRIDE

        read_offsets = (
            0x00,  # type
            0x04,  # seed/variant_a
            0x08,  # variant_b
            0x0C,  # credits
            0x10,  # energy
            0x14,  # prog id
            0x18,  # wall state / depleted-siphon marker
            0x1C,  # threat
            0x20,  # points
            0x24,  # siphon flag
            0x28,  # special state
            0x2C,  # exit overlay flag
            0x30,  # locked/hidden flag
            0x34,  # marker flag
        )
        values: list[int] = []
        for offset in read_offsets:
            read_result = _read_required_int32(reader, cell_base + offset)
            if not read_result.is_ok:
                failure = read_result.error or ReadFailure(
                    code="read_failed",
                    message="Failed reading map cell field.",
                    address=cell_base + offset,
                    size=4,
                )
                return _invalid_map(
                    failure.code,
                    source_field=source_field,
                    address=failure.address,
                    error=failure.message,
                )
            values.append(int(read_result.value or 0))

        cell_type = values[0]
        tile_variant = values[2]
        credits = values[3]
        energy = values[4]
        raw_prog_id = values[5]
        wall_state = values[6]
        threat = values[7]
        points = values[8]
        has_siphon = values[9] > 0
        special_state = values[10]
        has_exit_overlay = values[11] > 0
        is_wall = cell_type in (1, 2)
        is_exit = cell_type == 3
        prog_id = raw_prog_id if raw_prog_id >= 0 else None

        cell = MapCellState(
            position=position,
            cell_type=cell_type,
            tile_variant=tile_variant,
            wall_state=wall_state,
            prog_id=prog_id,
            credits=credits,
            energy=energy,
            points=points,
            threat=threat,
            special_state=special_state,
            has_siphon=has_siphon,
            has_exit_overlay=has_exit_overlay,
            is_wall=is_wall,
            is_exit=is_exit,
        )
        cell_states.append(cell)

        if has_siphon:
            siphons.append(position)
        if is_exit and exit_position is None:
            exit_position = position
        if is_wall:
            wall_type = "unknown_wall"
            if cell_type == 1:
                wall_type = "prog_wall"
            elif cell_type == 2:
                wall_type = "point_wall"
            walls.append(
                WallCellState(
                    position=position,
                    wall_type=wall_type,
                    wall_state=wall_state,
                    prog_id=prog_id,
                    points=points,
                    threat=threat,
                )
            )
        elif credits > 0 or energy > 0 or points > 0:
            resource_cells.append(
                ResourceCellState(
                    position=position,
                    credits=credits,
                    energy=energy,
                    points=points,
                )
            )

    enemies: list[EnemyState] = []
    for slot in range(_MAP_ENTITY_COUNT):
        entity_base = root_address + _MAP_ENTITIES_BASE_OFFSET + slot * _MAP_ENTITY_STRIDE
        active_result = _read_required_bool(reader, entity_base)
        if not active_result.is_ok:
            failure = active_result.error or ReadFailure(
                code="read_failed",
                message="Failed reading entity active flag.",
                address=entity_base,
                size=1,
            )
            return _invalid_map(
                failure.code,
                source_field=source_field,
                address=failure.address,
                error=failure.message,
            )
        if not bool(active_result.value):
            continue

        type_result = _read_required_int32(reader, entity_base + 0x08)
        hp_result = _read_required_int32(reader, entity_base + 0x0C)
        state_result = _read_required_int32(reader, entity_base + 0x18)
        x_result = _read_required_int32(reader, entity_base + 0x34)
        y_result = _read_required_int32(reader, entity_base + 0x38)
        for result in (type_result, hp_result, state_result, x_result, y_result):
            if result.is_ok:
                continue
            failure = result.error or ReadFailure(
                code="read_failed",
                message="Failed reading entity field.",
                address=entity_base,
                size=4,
            )
            return _invalid_map(
                failure.code,
                source_field=source_field,
                address=failure.address,
                error=failure.message,
            )

        enemy_x = int(x_result.value or 0)
        enemy_y = int(y_result.value or 0)
        enemy_position = GridPosition(x=enemy_x, y=enemy_y)
        in_bounds = 0 <= enemy_x < _MAP_WIDTH and 0 <= enemy_y < _MAP_HEIGHT
        incubation_timer = int(state_result.value or 0)
        if slot == 0:
            player_position = enemy_position
            continue

        enemies.append(
            EnemyState(
                slot=slot,
                type_id=_mask_enemy_type_for_visibility(
                    raw_type_id=int(type_result.value or 0),
                    incubation_timer=incubation_timer,
                    in_bounds=in_bounds,
                ),
                position=enemy_position,
                hp=int(hp_result.value or 0),
                state=incubation_timer,
                in_bounds=in_bounds,
                incubation_timer=incubation_timer,
            )
        )

    layers, layer_refresh = _build_map_layers(
        width=_MAP_WIDTH,
        height=_MAP_HEIGHT,
        cells=tuple(cell_states),
        siphons=tuple(siphons),
        player_position=player_position,
        exit_position=exit_position,
        enemies=tuple(enemies),
        previous_map_state=previous_map_state,
    )

    return _ok_map(
        source_field=source_field or "derived_root",
        address=root_address,
        cells=tuple(cell_states),
        siphons=tuple(siphons),
        walls=tuple(walls),
        resource_cells=tuple(resource_cells),
        player_position=player_position,
        exit_position=exit_position,
        enemies=tuple(enemies),
        layers=layers,
        layer_refresh=layer_refresh,
    )


def _derive_fail_state(
    health: FieldState,
    terminal_health_value: int,
) -> FieldState:
    if health.status != "ok":
        return _missing_field(
            "fail_state_unavailable",
            source_field=health.source_field,
            error="Health is unavailable; cannot derive fail_state.",
        )
    try:
        numeric_health = int(float(health.value))
    except (TypeError, ValueError):
        return _invalid_field(
            "health_not_numeric",
            source_field=health.source_field,
            address=health.address,
            error="Health value is non-numeric; cannot derive fail_state.",
        )

    return _ok_field(
        numeric_health == terminal_health_value,
        address=health.address,
        source_field=f"derived:{health.source_field}=={terminal_health_value}",
    )


def extract_state(
    *,
    reader: ProcessMemoryReader,
    registry: OffsetRegistry,
    module_base_resolver: ModuleBaseResolver | None = None,
    previous_map_state: MapState | None = None,
    terminal_health_value: int = -1,
    timestamp_fn: Callable[[], str] = _now_iso_utc,
) -> GameStateSnapshot:
    """Extract a normalized state snapshot with per-field status metadata."""
    entries = _entry_by_name(registry)
    health_entry = _select_entry(entries, _HEALTH_FIELD_CANDIDATES)
    energy_entry = _select_entry(entries, _ENERGY_FIELD_CANDIDATES)
    currency_entry = _select_entry(entries, _CURRENCY_FIELD_CANDIDATES)
    collected_progs_entry = _select_entry(entries, _COLLECTED_PROGS_FIELD_CANDIDATES)
    can_siphon_now_entry = _select_entry(entries, _CAN_SIPHON_NOW_FIELD_CANDIDATES)
    prog_slots_available_mask_entry = _select_entry(
        entries,
        _PROG_SLOTS_AVAILABLE_MASK_FIELD_CANDIDATES,
    )

    health = _extract_field(reader=reader, entry=health_entry, module_base_resolver=module_base_resolver)
    energy = _extract_field(reader=reader, entry=energy_entry, module_base_resolver=module_base_resolver)
    currency = _extract_field(reader=reader, entry=currency_entry, module_base_resolver=module_base_resolver)
    inventory = _extract_inventory(
        reader=reader,
        entry=collected_progs_entry,
        module_base_resolver=module_base_resolver,
    )
    map_state = _extract_map_state(
        reader=reader,
        entries_by_name=entries,
        module_base_resolver=module_base_resolver,
        previous_map_state=previous_map_state,
    )
    can_siphon_now = _extract_optional_bool(
        reader=reader,
        entry=can_siphon_now_entry,
        module_base_resolver=module_base_resolver,
    )
    prog_slots_available_mask = _extract_optional_uint32_mask(
        reader=reader,
        entry=prog_slots_available_mask_entry,
        module_base_resolver=module_base_resolver,
    )
    root_address = int(map_state.address) if map_state.address is not None else None
    if root_address is None:
        root_address, _root_source_field, _root_failure = _resolve_game_state_root(
            reader=reader,
            entries_by_name=entries,
            module_base_resolver=module_base_resolver,
        )
    if can_siphon_now is None:
        can_siphon_now = _derive_optional_bool_from_root(
            reader=reader,
            root_address=root_address,
            offset=_CAN_SIPHON_NOW_DERIVED_OFFSET,
        )
    if prog_slots_available_mask is None:
        prog_slots_available_mask = _derive_prog_slots_available_mask_from_root(
            reader=reader,
            root_address=root_address,
        )
    fail_state = _derive_fail_state(
        health,
        terminal_health_value,
    )

    known_names = {
        entry.name
        for entry in (
            health_entry,
            energy_entry,
            currency_entry,
            collected_progs_entry,
            can_siphon_now_entry,
            prog_slots_available_mask_entry,
        )
        if entry is not None
    }
    extra_fields: dict[str, FieldState] = {}
    for entry in registry.entries:
        if entry.name in known_names:
            continue
        extra_fields[entry.name] = _extract_field(
            reader=reader,
            entry=entry,
            module_base_resolver=module_base_resolver,
        )

    return GameStateSnapshot(
        timestamp_utc=timestamp_fn(),
        health=health,
        energy=energy,
        currency=currency,
        fail_state=fail_state,
        inventory=inventory,
        map=map_state,
        can_siphon_now=can_siphon_now,
        prog_slots_available_mask=prog_slots_available_mask,
        extra_fields=extra_fields,
    )

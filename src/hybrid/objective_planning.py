"""Pure helpers for Hybrid objective resolution and target scoring."""

from __future__ import annotations

from src.state.schema import GameStateSnapshot, GridPosition

_CARDINAL_DELTAS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))


def manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    walls = {cell.position for cell in state.map.cells if cell.is_wall}
    if walls:
        return walls
    return {wall.position for wall in state.map.walls}


def is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def adjacent_positions(position: GridPosition) -> tuple[GridPosition, ...]:
    return tuple(
        GridPosition(x=position.x + dx, y=position.y + dy)
        for dx, dy in _CARDINAL_DELTAS
    )


def siphon_reach_positions(position: GridPosition) -> tuple[GridPosition, ...]:
    return (position, *adjacent_positions(position))


def cell_index(state: GameStateSnapshot) -> dict[GridPosition, object]:
    if state.map.status != "ok":
        return {}
    return {cell.position: cell for cell in state.map.cells}


def cell_target_already_siphoned(cell: object) -> bool:
    wall_state = int(getattr(cell, "wall_state", 0))
    is_wall = bool(getattr(cell, "is_wall", False) or hasattr(cell, "wall_type"))
    if is_wall:
        has_payload = bool(
            (getattr(cell, "prog_id", None) is not None and int(getattr(cell, "prog_id", 0)) > 0)
            or max(int(getattr(cell, "points", 0)), 0) > 0
        )
        return has_payload and wall_state <= 0
    return wall_state > 0


def resource_value_index(state: GameStateSnapshot) -> dict[GridPosition, tuple[int, int, int]]:
    if state.map.status != "ok":
        return {}
    cells = cell_index(state)
    values: dict[GridPosition, tuple[int, int, int]] = {}
    for resource in state.map.resource_cells:
        raw_cell = cells.get(resource.position)
        if raw_cell is not None and cell_target_already_siphoned(raw_cell):
            continue
        values[resource.position] = (
            max(int(resource.credits), 0),
            max(int(resource.energy), 0),
            max(int(resource.points), 0),
        )
    return values


def read_layer_cell(
    layer: tuple[tuple[int, ...], ...],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> int | None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    if len(layer) != height:
        return None
    row = layer[y]
    if len(row) != width:
        return None
    try:
        return int(row[x])
    except (TypeError, ValueError):
        return None


def credit_energy_cluster_total(
    *,
    position: GridPosition,
    cell_index_map: dict[GridPosition, object],
    resource_values: dict[GridPosition, tuple[int, int, int]],
    width: int,
    height: int,
    credits_map: tuple[tuple[int, ...], ...],
    energy_map: tuple[tuple[int, ...], ...],
) -> int:
    x = int(position.x)
    y = int(position.y)
    layer_credits = read_layer_cell(
        credits_map,
        x=x,
        y=y,
        width=width,
        height=height,
    )
    layer_energy = read_layer_cell(
        energy_map,
        x=x,
        y=y,
        width=width,
        height=height,
    )
    if layer_credits is not None and layer_energy is not None:
        return max(layer_credits, 0) + max(layer_energy, 0)

    total = 0
    for candidate in (position, *adjacent_positions(position)):
        cell = cell_index_map.get(candidate)
        if cell is not None and cell_target_already_siphoned(cell):
            cell_credits = 0
            cell_energy = 0
        else:
            cell_credits = max(int(getattr(cell, "credits", 0)), 0) if cell is not None else 0
            cell_energy = max(int(getattr(cell, "energy", 0)), 0) if cell is not None else 0
        resource = resource_values.get(candidate)
        resource_credits = resource[0] if resource is not None else 0
        resource_energy = resource[1] if resource is not None else 0
        total += max(cell_credits, resource_credits)
        total += max(cell_energy, resource_energy)
    return total


def count_enemies(state: GameStateSnapshot) -> int:
    if state.map.status != "ok":
        return 0
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def resource_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    if state.map.status != "ok":
        return ()
    width = int(state.map.width)
    height = int(state.map.height)
    layers = state.map.layers
    walls = wall_positions(state)
    cells = cell_index(state)
    resource_values = resource_value_index(state)
    weighted_targets: dict[GridPosition, int] = {}

    def add_target(position: GridPosition, *, weight: int) -> None:
        if not is_in_bounds(position, width=width, height=height):
            return
        if position in walls:
            return
        cell = cells.get(position)
        if cell is not None and cell_target_already_siphoned(cell):
            return
        bounded = max(int(weight), 0)
        current = weighted_targets.get(position)
        if current is None or bounded > current:
            weighted_targets[position] = bounded

    for cell in state.map.resource_cells:
        raw_cell = cells.get(cell.position)
        if raw_cell is not None and cell_target_already_siphoned(raw_cell):
            continue
        cluster_total = credit_energy_cluster_total(
            position=cell.position,
            cell_index_map=cells,
            resource_values=resource_values,
            width=width,
            height=height,
            credits_map=layers.credits_map,
            energy_map=layers.energy_map,
        )
        points_total = max(int(cell.points), 0)
        if cluster_total <= 0 and points_total <= 0:
            continue
        add_target(cell.position, weight=cluster_total + points_total)

    for cell in state.map.cells:
        if cell_target_already_siphoned(cell):
            continue
        cluster_total = credit_energy_cluster_total(
            position=cell.position,
            cell_index_map=cells,
            resource_values=resource_values,
            width=width,
            height=height,
            credits_map=layers.credits_map,
            energy_map=layers.energy_map,
        )
        has_prog = bool(cell.prog_id is not None and cell.prog_id > 0)
        points_total = max(int(cell.points), 0)
        if not cell.is_wall:
            if cluster_total <= 0 and not has_prog and points_total <= 0:
                continue
            add_target(
                cell.position,
                weight=cluster_total + points_total + (25 if has_prog else 0),
            )
            continue
        if not has_prog and points_total <= 0:
            continue
        wall_weight = points_total + (25 if has_prog else 0)
        for adjacent in adjacent_positions(cell.position):
            adjacent_cluster = credit_energy_cluster_total(
                position=adjacent,
                cell_index_map=cells,
                resource_values=resource_values,
                width=width,
                height=height,
                credits_map=layers.credits_map,
                energy_map=layers.energy_map,
            )
            add_target(adjacent, weight=adjacent_cluster + wall_weight)

    for wall in state.map.walls:
        if cell_target_already_siphoned(wall):
            continue
        has_prog = bool(wall.prog_id is not None and wall.prog_id > 0)
        points_total = max(int(wall.points), 0)
        if not has_prog and points_total <= 0:
            continue
        wall_weight = points_total + (25 if has_prog else 0)
        for adjacent in adjacent_positions(wall.position):
            adjacent_cluster = credit_energy_cluster_total(
                position=adjacent,
                cell_index_map=cells,
                resource_values=resource_values,
                width=width,
                height=height,
                credits_map=layers.credits_map,
                energy_map=layers.energy_map,
            )
            add_target(adjacent, weight=adjacent_cluster + wall_weight)

    ordered = sorted(
        weighted_targets.items(),
        key=lambda item: (-item[1], item[0].y, item[0].x),
    )
    return tuple(position for position, _ in ordered)

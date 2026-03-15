"""Tests for hybrid decision coordination behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.hybrid.coordinator import HybridCoordinator, HybridCoordinatorConfig
from src.hybrid.types import ObjectivePhase, ThreatOverride
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapLayersState,
    MapState,
    ResourceCellState,
    WallCellState,
)


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _build_state(
    *,
    player: GridPosition,
    siphons: tuple[GridPosition, ...] = (),
    exit_position: GridPosition | None = None,
    enemies: tuple[EnemyState, ...] = (),
    cells: tuple[MapCellState, ...] | None = None,
    resource_cells: tuple[ResourceCellState, ...] = (),
    walls: tuple[WallCellState, ...] = (),
    map_layers: MapLayersState | None = None,
    can_siphon_now: bool | None = True,
    player_siphons: int | None = None,
) -> GameStateSnapshot:
    map_cells = cells or tuple(
        MapCellState(
            position=GridPosition(x=x, y=y),
            cell_type=0,
            tile_variant=0,
            wall_state=0,
            is_wall=False,
            is_exit=(exit_position == GridPosition(x=x, y=y)),
        )
        for y in range(6)
        for x in range(6)
    )
    return GameStateSnapshot(
        timestamp_utc="2026-03-11T00:00:00Z",
        health=_ok_field(10),
        energy=_ok_field(8),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        inventory=InventoryState(status="ok", raw_prog_ids=()),
        map=MapState(
            status="ok",
            width=6,
            height=6,
            cells=map_cells,
            siphons=siphons,
            walls=walls,
            resource_cells=resource_cells,
            player_position=player,
            exit_position=exit_position,
            enemies=enemies,
            layers=map_layers or MapLayersState(),
        ),
        can_siphon_now=can_siphon_now,
        extra_fields={"siphons": _ok_field(player_siphons)} if player_siphons is not None else {},
    )


@dataclass
class _StubMetaController:
    feature_count: int = 18
    epsilon: float = 0.0

    def start_episode(self) -> None:
        return

    def select_objective(
        self,
        *,
        features: Sequence[float],
        allowed_phases: Sequence[ObjectivePhase] | None = None,
        explore: bool = True,
    ) -> tuple[ObjectivePhase, str, float | None]:
        assert len(tuple(features)) == self.feature_count
        assert allowed_phases is not None
        return (allowed_phases[0], "stub_meta", 0.0)


@dataclass
class _StubThreatController:
    feature_count: int = 20
    epsilon: float = 0.0

    def start_episode(self) -> None:
        return

    def select_override(
        self,
        *,
        features: Sequence[float],
        threat_active: bool,
        allowed_overrides: Sequence[ThreatOverride] | None = None,
        explore: bool = True,
    ) -> tuple[ThreatOverride, str, float | None]:
        assert len(tuple(features)) == self.feature_count
        if not threat_active:
            return (ThreatOverride.ROUTE_DEFAULT, "inactive", None)
        return (ThreatOverride.WAIT, "force_wait", 0.0)


@dataclass
class _SequenceMetaController:
    selected_phases: list[ObjectivePhase]
    feature_count: int = 18
    epsilon: float = 0.0

    def start_episode(self) -> None:
        return

    def select_objective(
        self,
        *,
        features: Sequence[float],
        allowed_phases: Sequence[ObjectivePhase] | None = None,
        explore: bool = True,
    ) -> tuple[ObjectivePhase, str, float | None]:
        assert len(tuple(features)) == self.feature_count
        assert allowed_phases is not None
        phase = self.selected_phases.pop(0) if self.selected_phases else allowed_phases[0]
        return (phase, "sequence_meta", 0.0)


def test_allowed_meta_phases_prioritize_siphons() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        siphons=(GridPosition(1, 0),),
        exit_position=GridPosition(5, 5),
    )

    phases = coordinator.allowed_meta_phases(state)
    assert phases == (ObjectivePhase.COLLECT_SIPHONS,)


def test_allowed_meta_phases_exit_after_siphons_when_configured() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
        config=HybridCoordinatorConfig(exit_after_siphons_when_scripted=True),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        siphons=(),
        resource_cells=(ResourceCellState(position=GridPosition(1, 0), credits=3),),
        exit_position=GridPosition(5, 5),
    )

    phases = coordinator.allowed_meta_phases(state)
    assert phases == (ObjectivePhase.EXIT_SECTOR,)


def test_allowed_meta_phases_progress_to_resources_then_exit_after_siphons() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        siphons=(),
        resource_cells=(ResourceCellState(position=GridPosition(1, 0), credits=3),),
        exit_position=GridPosition(5, 5),
    )

    phases = coordinator.allowed_meta_phases(state)
    assert phases == (
        ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        ObjectivePhase.EXIT_SECTOR,
    )


def test_invalid_threat_override_falls_back_to_route_action() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        siphons=(GridPosition(2, 0),),
        exit_position=GridPosition(5, 5),
        enemies=(
            EnemyState(
                slot=1,
                type_id=2,
                position=GridPosition(0, 1),
                hp=2,
                state=1,
                in_bounds=True,
            ),
        ),
    )
    trace = coordinator.decide(
        state=state,
        available_actions=("move_up", "move_right"),
        use_meta_controller=False,
        use_threat_controller=True,
        explore_meta=False,
        explore_threat=False,
    )

    assert trace.decision.threat_override == ThreatOverride.WAIT
    assert trace.decision.used_fallback is True
    assert trace.decision.action == "move_right"


def test_phase_lock_holds_resource_phase_until_min_steps_elapsed() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_SequenceMetaController(
            selected_phases=[
                ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
                ObjectivePhase.EXIT_SECTOR,
                ObjectivePhase.EXIT_SECTOR,
                ObjectivePhase.EXIT_SECTOR,
                ObjectivePhase.EXIT_SECTOR,
                ObjectivePhase.EXIT_SECTOR,
                ObjectivePhase.EXIT_SECTOR,
            ]
        ),
        threat_controller=_StubThreatController(),
        config=HybridCoordinatorConfig(phase_lock_min_steps=6, target_stall_release_steps=4),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        resource_cells=(ResourceCellState(position=GridPosition(2, 0), credits=3),),
        exit_position=GridPosition(5, 5),
    )

    traces = [
        coordinator.decide(
            state=state,
            available_actions=("move_up", "move_right"),
            use_meta_controller=True,
            use_threat_controller=False,
            explore_meta=False,
            explore_threat=False,
        )
        for _ in range(7)
    ]

    assert traces[0].decision.objective.phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
    assert all(
        trace.decision.objective.phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
        for trace in traces[1:6]
    )
    assert traces[6].decision.objective.phase == ObjectivePhase.EXIT_SECTOR
    assert coordinator.phase_lock_overrides == 5


def test_phase_lock_releases_after_configured_stall_steps() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_SequenceMetaController(
            selected_phases=[
                ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
                ObjectivePhase.EXIT_SECTOR,
            ]
        ),
        threat_controller=_StubThreatController(),
        config=HybridCoordinatorConfig(phase_lock_min_steps=6, target_stall_release_steps=4),
    )
    state = _build_state(
        player=GridPosition(0, 0),
        resource_cells=(ResourceCellState(position=GridPosition(2, 0), credits=3),),
        exit_position=GridPosition(5, 5),
    )

    trace = coordinator.decide(
        state=state,
        available_actions=("move_up", "move_right"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    for _ in range(4):
        coordinator.observe_step_result(trace=trace, info={"action_effective": False}, next_state=state)

    released_trace = coordinator.decide(
        state=state,
        available_actions=("move_up", "move_right"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )

    assert coordinator.stall_releases == 1
    assert released_trace.decision.objective.phase == ObjectivePhase.EXIT_SECTOR


def test_resource_target_lock_persists_until_target_invalidates() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    resource_cells = (
        ResourceCellState(position=GridPosition(2, 0), credits=4),
        ResourceCellState(position=GridPosition(0, 2), credits=4),
    )
    first_state = _build_state(
        player=GridPosition(0, 0),
        resource_cells=resource_cells,
    )
    first_trace = coordinator.decide(
        state=first_state,
        available_actions=("move_up", "move_right", "space"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    first_target = first_trace.decision.objective.target_position
    assert first_target is not None

    moved_state = _build_state(
        player=GridPosition(0, 1),
        resource_cells=resource_cells,
    )
    second_trace = coordinator.decide(
        state=moved_state,
        available_actions=("move_up", "move_right", "move_down", "space"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )

    assert second_trace.decision.objective.target_position == first_target


def test_effective_resource_siphon_blacklists_completed_target_in_meta_mode() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        resource_cells=(
            ResourceCellState(position=GridPosition(1, 1), credits=3),
            ResourceCellState(position=GridPosition(4, 1), credits=3),
        ),
        exit_position=GridPosition(5, 5),
        player_siphons=1,
    )

    first_trace = coordinator.decide(
        state=state,
        available_actions=("move_right", "space"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )

    assert first_trace.decision.objective.phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
    assert first_trace.decision.objective.target_position == GridPosition(1, 1)
    assert first_trace.decision.action == "space"

    coordinator.observe_step_result(trace=first_trace, info={"action_effective": True})

    second_trace = coordinator.decide(
        state=state,
        available_actions=("move_right", "space"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )

    assert second_trace.decision.objective.target_position == GridPosition(3, 1)


def test_effective_resource_siphon_removes_collect_resources_phase_when_no_targets_remain() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        exit_position=GridPosition(5, 5),
        player_siphons=1,
    )

    first_trace = coordinator.decide(
        state=state,
        available_actions=("move_right", "space"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    coordinator.observe_step_result(trace=first_trace, info={"action_effective": True})

    assert coordinator.allowed_meta_phases(state) == (ObjectivePhase.EXIT_SECTOR,)


def test_ineffective_resource_siphon_keeps_target_available() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        resource_cells=(
            ResourceCellState(position=GridPosition(1, 1), credits=3),
            ResourceCellState(position=GridPosition(4, 1), credits=3),
        ),
        exit_position=GridPosition(5, 5),
        player_siphons=1,
    )

    first_trace = coordinator.decide(
        state=state,
        available_actions=("move_right", "space"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    coordinator.observe_step_result(trace=first_trace, info={"action_effective": False})

    second_trace = coordinator.decide(
        state=state,
        available_actions=("move_right", "space"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )

    assert second_trace.decision.objective.target_position == GridPosition(1, 1)


def test_resource_targets_use_precomputed_energy_credit_layers() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    zero_row = (0, 0, 0, 0, 0, 0)
    credits_grid = (
        zero_row,
        (0, 0, 5, 0, 0, 0),
        zero_row,
        zero_row,
        zero_row,
        zero_row,
    )
    energy_grid = (
        zero_row,
        (0, 0, 2, 0, 0, 0),
        zero_row,
        zero_row,
        zero_row,
        zero_row,
    )
    state = _build_state(
        player=GridPosition(0, 0),
        resource_cells=(),
        map_layers=MapLayersState(
            credits_map=credits_grid,
            energy_map=energy_grid,
        ),
    )

    target = coordinator.resolve_target_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
    )
    assert target == GridPosition(2, 1)


def test_resource_targets_skip_previously_siphoned_non_wall_cells() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    cells = tuple(
        MapCellState(
            position=GridPosition(x=x, y=y),
            cell_type=0,
            tile_variant=0,
            wall_state=1 if (x == 1 and y == 1) else 0,
            is_wall=False,
        )
        for y in range(6)
        for x in range(6)
    )
    state = _build_state(
        player=GridPosition(0, 1),
        cells=cells,
        resource_cells=(
            ResourceCellState(position=GridPosition(1, 1), credits=3),
            ResourceCellState(position=GridPosition(4, 1), credits=2),
        ),
    )

    target = coordinator.resolve_target_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
    )
    assert target == GridPosition(3, 1)


def test_resource_targets_skip_depleted_prog_walls() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    walls = (WallCellState(position=GridPosition(2, 2), wall_type="prog_wall", wall_state=0, prog_id=8),)
    cells = tuple(
        MapCellState(
            position=GridPosition(x=x, y=y),
            cell_type=0,
            tile_variant=0,
            wall_state=0,
            is_wall=(x == 2 and y == 2),
            prog_id=8 if (x == 2 and y == 2) else None,
            points=0,
        )
        for y in range(6)
        for x in range(6)
    )
    state = _build_state(
        player=GridPosition(0, 0),
        cells=cells,
        walls=walls,
    )

    target = coordinator.resolve_target_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
    )
    assert target is None


def test_resource_target_arrival_avoids_siphon_when_siphons_depleted() -> None:
    space_coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    resource_cells = (ResourceCellState(position=GridPosition(1, 1), credits=3),)
    at_target_state = _build_state(
        player=GridPosition(1, 1),
        resource_cells=resource_cells,
        player_siphons=0,
    )
    space_trace = space_coordinator.decide(
        state=at_target_state,
        available_actions=("move_up", "space"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert space_trace.decision.action == "move_up"

    z_coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    z_trace = z_coordinator.decide(
        state=at_target_state,
        available_actions=("move_up", "z"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert z_trace.decision.action == "move_up"


def test_collect_siphon_target_arrival_prefers_space_then_z() -> None:
    space_coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    at_target_state = _build_state(
        player=GridPosition(1, 1),
        siphons=(GridPosition(1, 1),),
        player_siphons=2,
    )
    space_trace = space_coordinator.decide(
        state=at_target_state,
        available_actions=("move_up", "space"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert space_trace.decision.action == "space"

    z_coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    z_trace = z_coordinator.decide(
        state=at_target_state,
        available_actions=("move_up", "z"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert z_trace.decision.action == "z"


def test_prog_and_point_wall_targets_resolve_to_adjacent_walkable_tiles() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    walls = (WallCellState(position=GridPosition(2, 2), wall_type="prog_wall", wall_state=1, prog_id=8),)
    cells = tuple(
        MapCellState(
            position=GridPosition(x=x, y=y),
            cell_type=0,
            tile_variant=0,
            wall_state=0,
            is_wall=(x == 2 and y == 2),
            prog_id=8 if (x == 2 and y == 2) else None,
            points=0,
        )
        for y in range(6)
        for x in range(6)
    )
    state = _build_state(
        player=GridPosition(0, 0),
        cells=cells,
        walls=walls,
    )

    target = coordinator.resolve_target_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
    )
    assert target is not None
    assert target != GridPosition(2, 2)
    assert target in {
        GridPosition(2, 1),
        GridPosition(3, 2),
        GridPosition(2, 3),
        GridPosition(1, 2),
    }


def test_collect_resources_resolution_reverts_to_collect_siphons_when_inventory_is_empty() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        siphons=(GridPosition(3, 1),),
        resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        exit_position=GridPosition(5, 5),
        player_siphons=0,
    )

    resolved_phase, resolved_target = coordinator.resolve_objective_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        scripted_mode=True,
    )

    assert resolved_phase == ObjectivePhase.COLLECT_SIPHONS
    assert resolved_target == GridPosition(3, 1)
    assert (
        coordinator.resolve_target_for_phase(
            state=state,
            phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
            scripted_mode=True,
        )
        == GridPosition(3, 1)
    )


def test_collect_resources_decision_exits_when_siphons_are_depleted_and_none_remain_on_map() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        siphons=(),
        resource_cells=(
            ResourceCellState(position=GridPosition(1, 1), credits=3),
            ResourceCellState(position=GridPosition(2, 1), credits=3),
        ),
        exit_position=GridPosition(5, 5),
        player_siphons=0,
    )

    resolved_phase, resolved_target = coordinator.resolve_objective_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        scripted_mode=True,
    )

    assert resolved_phase == ObjectivePhase.EXIT_SECTOR
    assert resolved_target == GridPosition(5, 5)

    first_trace = coordinator.decide(
        state=state,
        available_actions=("space", "move_right", "move_up"),
        use_meta_controller=True,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert first_trace.decision.objective.phase == ObjectivePhase.EXIT_SECTOR
    assert "phase_resolved_to_exit_sector" in first_trace.decision.objective.reason
    assert first_trace.decision.objective.target_position == GridPosition(5, 5)
    assert first_trace.decision.action == "move_right"

    second_target = coordinator.resolve_target_for_phase(
        state=state,
        phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        scripted_mode=True,
    )
    assert second_target == GridPosition(5, 5)


def test_resource_target_arrival_can_siphon_when_player_has_siphons_even_if_map_has_none() -> None:
    coordinator = HybridCoordinator(
        meta_controller=_StubMetaController(),
        threat_controller=_StubThreatController(),
    )
    state = _build_state(
        player=GridPosition(1, 1),
        siphons=(),
        resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        player_siphons=1,
    )
    trace = coordinator.decide(
        state=state,
        available_actions=("move_up", "space"),
        use_meta_controller=False,
        use_threat_controller=False,
        explore_meta=False,
        explore_threat=False,
    )
    assert trace.decision.action == "space"

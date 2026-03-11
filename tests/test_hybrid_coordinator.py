"""Tests for hybrid decision coordination behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.hybrid.coordinator import HybridCoordinator
from src.hybrid.types import ObjectivePhase, ThreatOverride
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapState,
)


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _build_state(
    *,
    player: GridPosition,
    siphons: tuple[GridPosition, ...] = (),
    exit_position: GridPosition | None = None,
    enemies: tuple[EnemyState, ...] = (),
) -> GameStateSnapshot:
    cells = tuple(
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
            cells=cells,
            siphons=siphons,
            player_position=player,
            exit_position=exit_position,
            enemies=enemies,
        ),
        can_siphon_now=True,
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


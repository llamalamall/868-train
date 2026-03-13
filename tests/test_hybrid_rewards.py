"""Tests for hybrid reward shaping."""

from __future__ import annotations

import pytest

from src.hybrid.rewards import (
    HybridMetaRewardWeights,
    HybridRewardSuite,
    HybridThreatRewardWeights,
)
from src.hybrid.types import ObjectivePhase, ThreatOverride
from src.state.schema import (
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapState,
    ResourceCellState,
)


def _field(value: object, *, status: str = "ok") -> FieldState:
    return FieldState(value=value, status=status)  # type: ignore[arg-type]


def _snapshot(
    *,
    map_state: MapState | None = None,
    extra_fields: dict[str, FieldState] | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-13T00:00:00+00:00",
        health=_field(10),
        energy=_field(5),
        currency=_field(3),
        fail_state=_field(False),
        inventory=InventoryState(status="missing"),
        map=map_state or MapState(status="missing"),
        extra_fields=extra_fields or {},
    )


def _exit_map() -> MapState:
    return MapState(
        status="ok",
        player_position=GridPosition(5, 5),
        exit_position=GridPosition(5, 5),
    )


def test_compute_meta_reward_adds_final_sector_win_bonus_when_victory_flag_is_active() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=30.0,
        )
    )

    previous = _snapshot(extra_fields={"current_sector": _field(7)})
    current = _snapshot(
        extra_fields={
            "current_sector": _field(7),
            "victory_active": _field(True),
        }
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=True,
        info={"terminal_reason": "state:victory"},
    )

    assert result.final_sector_win == pytest.approx(30.0)
    assert result.total == pytest.approx(30.0)


def test_compute_meta_reward_adds_final_sector_win_bonus_on_final_exit_fallback() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=30.0,
        )
    )

    previous = _snapshot(
        map_state=_exit_map(),
        extra_fields={"current_sector": _field(7)},
    )
    current = _snapshot()

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=True,
        info={"terminal_reason": "state:start_screen"},
    )

    assert result.final_sector_win == pytest.approx(30.0)
    assert result.total == pytest.approx(30.0)


def test_compute_meta_reward_does_not_add_final_sector_win_bonus_before_sector_8() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=30.0,
        )
    )

    previous = _snapshot(
        map_state=_exit_map(),
        extra_fields={"current_sector": _field(6)},
    )
    current = _snapshot()

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=True,
        info={"terminal_reason": "state:start_screen"},
    )

    assert result.final_sector_win == 0.0
    assert result.total == 0.0


def test_compute_meta_reward_does_not_add_final_sector_win_bonus_for_premature_exit() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=30.0,
        )
    )

    previous = _snapshot(
        map_state=_exit_map(),
        extra_fields={"current_sector": _field(7)},
    )
    current = _snapshot()

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=True,
        info={
            "terminal_reason": "state:start_screen",
            "premature_exit_attempt": True,
        },
    )

    assert result.final_sector_win == 0.0
    assert result.total == 0.0


def test_compute_meta_reward_penalizes_exit_when_resources_remain() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=2.0,
            sector_advance=0.0,
            final_sector_win=0.0,
        )
    )
    previous = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(5, 5),
            exit_position=GridPosition(5, 5),
            resource_cells=(),
        )
    )
    current = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(5, 5),
            exit_position=GridPosition(5, 5),
            resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        ),
        extra_fields={"siphons": _field(1)},
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=False,
        info={},
    )

    assert result.premature_exit_penalty == pytest.approx(-2.0)
    assert result.total == pytest.approx(-2.0)


def test_compute_meta_reward_does_not_penalize_exit_when_resources_remain_but_player_has_no_siphons() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=2.0,
            sector_advance=0.0,
            final_sector_win=0.0,
        )
    )
    previous = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(5, 5),
            exit_position=GridPosition(5, 5),
            siphons=(),
            resource_cells=(),
        ),
        extra_fields={"siphons": _field(0)},
    )
    current = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(5, 5),
            exit_position=GridPosition(5, 5),
            siphons=(),
            resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        ),
        extra_fields={"siphons": _field(0)},
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=False,
        info={},
    )

    assert result.premature_exit_penalty == 0.0
    assert result.total == 0.0


def test_compute_meta_reward_exit_completion_allows_uncollectable_resources_with_zero_siphons() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=3.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=2.0,
            sector_advance=0.0,
            final_sector_win=0.0,
        )
    )
    previous = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(4, 5),
            exit_position=GridPosition(5, 5),
            siphons=(),
            resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        ),
        extra_fields={"siphons": _field(0)},
    )
    current = _snapshot(
        map_state=MapState(
            status="ok",
            player_position=GridPosition(5, 5),
            exit_position=GridPosition(5, 5),
            siphons=(),
            resource_cells=(ResourceCellState(position=GridPosition(1, 1), credits=3),),
        ),
        extra_fields={"siphons": _field(0)},
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.EXIT_SECTOR,
        done=False,
        info={"objective_target": GridPosition(5, 5)},
    )

    assert result.objective_complete == pytest.approx(3.0)
    assert result.premature_exit_penalty == 0.0
    assert result.total == pytest.approx(3.0)


def test_compute_meta_reward_uses_route_distance_overrides_from_info() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=1.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
        )
    )

    previous = _snapshot(map_state=MapState(status="ok", player_position=GridPosition(0, 0)))
    current = _snapshot(map_state=MapState(status="ok", player_position=GridPosition(0, 0)))

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={
            "objective_target": GridPosition(5, 5),
            "objective_distance_before": 6,
            "objective_distance_after": 2,
        },
    )

    assert result.phase_progress == pytest.approx(4.0)
    assert result.total == pytest.approx(4.0)


def test_compute_threat_reward_treats_memory_player_health_as_fail_terminal() -> None:
    reward_suite = HybridRewardSuite(
        threat_weights=HybridThreatRewardWeights(
            survival=0.0,
            damage_taken_penalty=0.0,
            fail_penalty=4.5,
            route_rejoin_bonus=0.0,
            invalid_override_penalty=0.0,
        )
    )
    previous = _snapshot()
    current = _snapshot()

    result = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=True,
        threat_override=ThreatOverride.EVADE,
        info={
            "terminal_reason": "memory:player_health",
            "threat_reward_active": True,
        },
    )

    assert result.fail_penalty == pytest.approx(-4.5)
    assert result.total == pytest.approx(-4.5)


def test_compute_threat_reward_rejoin_bonus_requires_transition_and_active_window() -> None:
    reward_suite = HybridRewardSuite(
        threat_weights=HybridThreatRewardWeights(
            survival=0.0,
            damage_taken_penalty=0.0,
            fail_penalty=0.0,
            route_rejoin_bonus=2.25,
            invalid_override_penalty=0.0,
        )
    )
    previous = _snapshot()
    current = _snapshot()

    no_transition = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ROUTE_DEFAULT,
        info={
            "threat_reward_active": True,
            "rejoined_route_transition": False,
        },
    )
    inactive_window = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ROUTE_DEFAULT,
        info={
            "threat_reward_active": False,
            "rejoined_route_transition": True,
        },
    )
    transitioned = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ROUTE_DEFAULT,
        info={
            "threat_reward_active": True,
            "rejoined_route_transition": True,
        },
    )

    assert no_transition.route_rejoin_bonus == 0.0
    assert inactive_window.route_rejoin_bonus == 0.0
    assert transitioned.route_rejoin_bonus == pytest.approx(2.25)

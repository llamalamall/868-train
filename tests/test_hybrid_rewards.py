"""Tests for hybrid reward shaping."""

from __future__ import annotations

import pytest

from src.hybrid.rewards import HybridMetaRewardWeights, HybridRewardSuite, HybridThreatRewardWeights
from src.hybrid.types import ObjectivePhase, ThreatOverride
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapLayersState,
    MapState,
)


def _field(value: object, *, status: str = "ok") -> FieldState:
    return FieldState(value=value, status=status)  # type: ignore[arg-type]


def _snapshot(
    *,
    map_state: MapState | None = None,
    extra_fields: dict[str, FieldState] | None = None,
    energy: object = 5,
    currency: object = 3,
    inventory: InventoryState | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-13T00:00:00+00:00",
        health=_field(10),
        energy=_field(energy),
        currency=_field(currency),
        fail_state=_field(False),
        inventory=inventory or InventoryState(status="missing"),
        map=map_state or MapState(status="missing"),
        extra_fields=extra_fields or {},
    )


def _exit_map() -> MapState:
    return MapState(
        status="ok",
        player_position=GridPosition(5, 5),
        exit_position=GridPosition(5, 5),
    )


def _combat_map(
    *,
    player: GridPosition = GridPosition(1, 1),
    enemies: tuple[EnemyState, ...],
    layers: MapLayersState | None = None,
) -> MapState:
    return MapState(
        status="ok",
        width=6,
        height=6,
        player_position=player,
        enemies=enemies,
        layers=layers or MapLayersState(),
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


def test_compute_meta_reward_adds_positive_loot_gain_terms() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
            currency_gain=0.5,
            energy_gain=0.25,
            score_gain=0.1,
            prog_gain=2.0,
        )
    )

    previous = _snapshot(
        currency=3,
        energy=5,
        inventory=InventoryState(status="ok", raw_prog_ids=(11,)),
        extra_fields={"score": _field(100)},
    )
    current = _snapshot(
        currency=7,
        energy=8,
        inventory=InventoryState(status="ok", raw_prog_ids=(11, 17)),
        extra_fields={"score": _field(130)},
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={},
    )

    assert result.currency_gain == pytest.approx(2.0)
    assert result.energy_gain == pytest.approx(0.75)
    assert result.score_gain == pytest.approx(3.0)
    assert result.prog_gain == pytest.approx(2.0)
    assert result.total == pytest.approx(7.75)


def test_compute_meta_reward_ignores_negative_loot_deltas() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
            currency_gain=0.5,
            energy_gain=0.25,
            score_gain=0.1,
            prog_gain=2.0,
        )
    )

    previous = _snapshot(
        currency=8,
        energy=6,
        inventory=InventoryState(status="ok", raw_prog_ids=(11, 17)),
        extra_fields={"score": _field(140)},
    )
    current = _snapshot(
        currency=3,
        energy=4,
        inventory=InventoryState(status="ok", raw_prog_ids=(11,)),
        extra_fields={"score": _field(100)},
    )

    result = reward_suite.compute_meta_reward(
        previous_state=previous,
        current_state=current,
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={},
    )

    assert result.currency_gain == 0.0
    assert result.energy_gain == 0.0
    assert result.score_gain == 0.0
    assert result.prog_gain == 0.0
    assert result.total == 0.0


def test_compute_meta_reward_applies_step_limit_penalty() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
            currency_gain=0.0,
            energy_gain=0.0,
            score_gain=0.0,
            prog_gain=0.0,
            step_limit_penalty=6.5,
            stagnation_penalty=0.0,
        )
    )

    result = reward_suite.compute_meta_reward(
        previous_state=_snapshot(),
        current_state=_snapshot(),
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={"hit_step_limit": True},
    )

    assert result.step_limit_penalty == pytest.approx(-6.5)
    assert result.total == pytest.approx(-6.5)


def test_compute_meta_reward_applies_stagnation_penalty_after_grace_window() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
            currency_gain=0.0,
            energy_gain=0.0,
            score_gain=0.0,
            prog_gain=0.0,
            step_limit_penalty=0.0,
            stagnation_penalty=0.2,
            stagnation_grace_steps=3,
        )
    )

    result = reward_suite.compute_meta_reward(
        previous_state=_snapshot(),
        current_state=_snapshot(),
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={"objective_stagnation_steps": 3},
    )

    assert result.stagnation_penalty == pytest.approx(-0.2)
    assert result.total == pytest.approx(-0.2)


def test_compute_meta_reward_does_not_apply_stagnation_penalty_before_grace_window() -> None:
    reward_suite = HybridRewardSuite(
        meta_weights=HybridMetaRewardWeights(
            objective_complete=0.0,
            phase_progress=0.0,
            step_cost=0.0,
            premature_exit_penalty=0.0,
            sector_advance=0.0,
            final_sector_win=0.0,
            currency_gain=0.0,
            energy_gain=0.0,
            score_gain=0.0,
            prog_gain=0.0,
            step_limit_penalty=0.0,
            stagnation_penalty=0.2,
            stagnation_grace_steps=3,
        )
    )

    result = reward_suite.compute_meta_reward(
        previous_state=_snapshot(),
        current_state=_snapshot(),
        objective_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
        done=False,
        info={"objective_stagnation_steps": 2},
    )

    assert result.stagnation_penalty == 0.0
    assert result.total == 0.0


def test_compute_threat_reward_adds_enemy_damage_and_clear_rewards() -> None:
    reward_suite = HybridRewardSuite(
        threat_weights=HybridThreatRewardWeights(
            survival=0.0,
            damage_taken_penalty=0.0,
            fail_penalty=0.0,
            route_rejoin_bonus=0.0,
            invalid_override_penalty=0.0,
            enemy_damaged=0.5,
            enemy_cleared=2.0,
            spawn_debt_penalty=0.0,
        )
    )
    previous = _snapshot(
        map_state=_combat_map(
            enemies=(
                EnemyState(slot=1, type_id=3, position=GridPosition(2, 1), hp=4, state=1, in_bounds=True),
                EnemyState(slot=2, type_id=2, position=GridPosition(4, 1), hp=2, state=1, in_bounds=True),
            )
        )
    )
    current = _snapshot(
        map_state=_combat_map(
            enemies=(EnemyState(slot=1, type_id=3, position=GridPosition(2, 1), hp=1, state=1, in_bounds=True),)
        )
    )

    result = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ENGAGE,
        info={},
    )

    assert result.enemy_damaged == pytest.approx(1.5)
    assert result.enemy_cleared == pytest.approx(2.0)
    assert result.total == pytest.approx(3.5)


def test_compute_threat_reward_applies_spawn_debt_penalty_from_siphon_cost() -> None:
    reward_suite = HybridRewardSuite(
        threat_weights=HybridThreatRewardWeights(
            survival=0.0,
            damage_taken_penalty=0.0,
            fail_penalty=0.0,
            route_rejoin_bonus=0.0,
            invalid_override_penalty=0.0,
            enemy_damaged=0.0,
            enemy_cleared=0.0,
            spawn_debt_penalty=0.25,
        )
    )
    layers = MapLayersState(
        siphon_penalty_map=(
            (0, 1, 0, 0, 0, 0),
            (1, 2, 1, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        )
    )
    previous = _snapshot(
        map_state=_combat_map(
            player=GridPosition(1, 1),
            enemies=(),
            layers=layers,
        )
    )
    current = _snapshot(
        map_state=_combat_map(
            player=GridPosition(1, 1),
            enemies=(),
            layers=layers,
        )
    )

    result = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ROUTE_DEFAULT,
        info={"action": "space", "action_effective": True},
    )

    assert result.spawn_debt_penalty == pytest.approx(-1.5)
    assert result.total == pytest.approx(-1.5)


def test_compute_threat_reward_applies_spawn_debt_penalty_from_enemy_growth() -> None:
    reward_suite = HybridRewardSuite(
        threat_weights=HybridThreatRewardWeights(
            survival=0.0,
            damage_taken_penalty=0.0,
            fail_penalty=0.0,
            route_rejoin_bonus=0.0,
            invalid_override_penalty=0.0,
            enemy_damaged=0.0,
            enemy_cleared=0.0,
            spawn_debt_penalty=0.5,
        )
    )
    previous = _snapshot(
        map_state=_combat_map(
            enemies=(EnemyState(slot=1, type_id=3, position=GridPosition(2, 1), hp=2, state=1, in_bounds=True),)
        )
    )
    current = _snapshot(
        map_state=_combat_map(
            enemies=(
                EnemyState(slot=1, type_id=3, position=GridPosition(2, 1), hp=2, state=1, in_bounds=True),
                EnemyState(slot=2, type_id=2, position=GridPosition(3, 1), hp=2, state=1, in_bounds=True),
            )
        )
    )

    result = reward_suite.compute_threat_reward(
        previous_state=previous,
        current_state=current,
        done=False,
        threat_override=ThreatOverride.ROUTE_DEFAULT,
        info={},
    )

    assert result.spawn_debt_penalty == pytest.approx(-0.5)
    assert result.total == pytest.approx(-0.5)

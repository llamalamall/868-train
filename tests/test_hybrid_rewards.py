"""Tests for hybrid reward shaping."""

from __future__ import annotations

import pytest

from src.hybrid.rewards import HybridMetaRewardWeights, HybridRewardSuite
from src.hybrid.types import ObjectivePhase
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, InventoryState, MapState


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

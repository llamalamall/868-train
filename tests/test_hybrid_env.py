"""Tests for hybrid environment action-availability adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.hybrid.env import HybridLiveEnv
from src.state.schema import (
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapState,
)


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _build_snapshot(
    *,
    player: GridPosition,
    cells: tuple[MapCellState, ...] = (),
    siphons: tuple[GridPosition, ...] = (),
    can_siphon_now: bool | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-11T00:00:00Z",
        health=_ok_field(10),
        energy=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        inventory=InventoryState(status="ok", raw_prog_ids=()),
        map=MapState(
            status="ok",
            width=6,
            height=6,
            player_position=player,
            cells=cells,
            siphons=siphons,
        ),
        can_siphon_now=can_siphon_now,
    )


@dataclass
class _StubGameEnv:
    action_space: tuple[str, ...]
    current_episode_id: str | None = None
    attached_pid: int | None = None
    _actions: tuple[str, ...] = ()
    runtime_callbacks: list[Any] | None = None

    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: str) -> tuple[Any, float, bool, dict[str, Any]]:
        raise NotImplementedError

    def available_actions(self, state: Any | None = None) -> tuple[str, ...]:
        return self._actions

    def close(self) -> None:
        return

    def add_runtime_binding_callback(self, callback: Any) -> None:
        if self.runtime_callbacks is None:
            self.runtime_callbacks = []
        self.runtime_callbacks.append(callback)


def test_hybrid_available_actions_adds_space_when_adjacent_siphon_marker_exists() -> None:
    game_env = _StubGameEnv(
        action_space=("move_up", "space"),
        _actions=("move_up",),
    )
    env = HybridLiveEnv(game_env=game_env, no_enemies_mode=True)
    state = _build_snapshot(
        player=GridPosition(1, 1),
        siphons=(GridPosition(2, 1),),
    )

    assert env.available_actions(state) == ("move_up", "space")


def test_hybrid_available_actions_does_not_add_space_for_previously_siphoned_tile() -> None:
    game_env = _StubGameEnv(
        action_space=("move_up", "space"),
        _actions=("move_up",),
    )
    env = HybridLiveEnv(game_env=game_env, no_enemies_mode=True)
    state = _build_snapshot(
        player=GridPosition(1, 1),
        cells=(
            MapCellState(
                position=GridPosition(2, 1),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                is_wall=True,
                prog_id=7,
            ),
        ),
        can_siphon_now=False,
    )

    assert env.available_actions(state) == ("move_up",)


def test_hybrid_env_exposes_attached_pid_and_runtime_callback_proxy() -> None:
    game_env = _StubGameEnv(
        action_space=("move_up",),
        attached_pid=4242,
    )
    env = HybridLiveEnv(game_env=game_env, no_enemies_mode=True)

    def callback(pid: int) -> int:
        return pid

    env.add_runtime_binding_callback(callback)

    assert env.attached_pid == 4242
    assert game_env.runtime_callbacks == [callback]

"""Tests for gym-like game environment wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.env.game_env import GameEnv, GameEnvConfig, ResetTimeoutError, StepTimeoutError, run_random_policy
from src.env.reset_manager import NoopResetManager
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, MapCellState, MapState


def _field(value: object, *, status: str = "ok") -> FieldState:
    return FieldState(value=value, status=status)  # type: ignore[arg-type]


def _snapshot(
    *,
    health: int = 10,
    energy: int = 5,
    credits: int = 3,
    failed: bool = False,
    map_state: MapState | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=_field(health),
        energy=_field(energy),
        currency=_field(credits),
        fail_state=_field(failed),
        map=map_state or MapState(status="missing"),
    )


@dataclass
class FakeActionAPI:
    """Minimal fake action API for env tests."""

    delay_seconds: float = 0.0
    actions: list[str] = field(default_factory=list)

    def perform_action(self, action_name: str) -> None:
        self.actions.append(action_name)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)


class QueueStateProvider:
    """Return queued states; once exhausted, keep returning last state."""

    def __init__(self, states: list[GameStateSnapshot]) -> None:
        self._states = list(states)
        self.calls = 0

    def __call__(self) -> GameStateSnapshot:
        self.calls += 1
        if len(self._states) > 1:
            return self._states.pop(0)
        return self._states[0]


def test_game_env_reset_and_step_contract() -> None:
    initial = _snapshot(health=10, failed=False)
    after_step = _snapshot(health=9, failed=False)
    state_provider = QueueStateProvider([initial, after_step])
    actions = FakeActionAPI()

    def reward_fn(previous: GameStateSnapshot, current: GameStateSnapshot, done: bool, _info: dict) -> float:
        assert previous.health.value == 10
        assert current.health.value == 9
        assert done is False
        return 1.5

    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        reward_fn=reward_fn,
        action_space=("move_up", "wait"),
    )

    reset_state = env.reset()
    assert reset_state.health.value == 10
    assert env.current_episode_id is not None

    state, reward, done, info = env.step("move_up")
    assert state.health.value == 9
    assert reward == 1.5
    assert done is False
    assert info["step_index"] == 0
    assert info["action"] == "move_up"
    assert actions.actions == ["move_up"]


def test_game_env_step_sets_done_from_fail_detector_result() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _snapshot(failed=False)])
    actions = FakeActionAPI()
    fail_detector = SimpleNamespace(
        check=lambda: SimpleNamespace(
            is_terminal=True,
            reason="memory:fail_state",
            source="memory",
            error=None,
        )
    )
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        fail_detector=fail_detector,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
    )

    env.reset()
    _, _, done, info = env.step("wait")

    assert done is True
    assert info["terminal_reason"] == "memory:fail_state"


def test_game_env_reset_timeout_when_terminal_state_persists() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=True)])
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(
            reset_timeout_seconds=0.05,
            state_timeout_seconds=0.05,
            state_poll_interval_seconds=0.01,
            require_non_terminal_on_reset=True,
        ),
    )

    with pytest.raises(ResetTimeoutError):
        env.reset()


def test_game_env_step_timeout_when_action_blocks() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(delay_seconds=0.20),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(
            step_timeout_seconds=0.05,
            state_timeout_seconds=0.05,
            reset_timeout_seconds=0.05,
            require_non_terminal_on_reset=False,
        ),
    )
    env.reset()
    with pytest.raises(StepTimeoutError):
        env.step("wait")


def test_run_random_policy_returns_episode_results() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    results = run_random_policy(env=env, episodes=2, max_steps_per_episode=2, seed=7)

    assert len(results) == 2
    assert all(result.steps == 2 for result in results)
    assert all(result.done is False for result in results)


def test_run_random_policy_uses_supplied_action_subset() -> None:
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("wait", "cancel"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    run_random_policy(env=env, episodes=1, max_steps_per_episode=3, seed=9, actions=("cancel",))

    assert actions.actions == ["cancel", "cancel", "cancel"]


def test_game_env_available_actions_filters_edges_and_walls() -> None:
    map_state = MapState(
        status="ok",
        width=3,
        height=3,
        player_position=GridPosition(0, 0),
        cells=(
            MapCellState(
                position=GridPosition(1, 0),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                is_wall=True,
            ),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(map_state=map_state)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "move_left", "move_right", "move_down", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    assert env.available_actions() == ("move_up",)


def test_run_random_policy_avoids_blocked_edge_moves() -> None:
    map_state = MapState(
        status="ok",
        width=2,
        height=1,
        player_position=GridPosition(0, 0),
    )
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=QueueStateProvider([_snapshot(map_state=map_state)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_left", "move_right"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    run_random_policy(env=env, episodes=1, max_steps_per_episode=4, seed=3)

    assert actions.actions == ["move_right", "move_right", "move_right", "move_right"]

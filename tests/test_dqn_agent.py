"""Tests for Task 14 DQN agent training and checkpoint behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.dqn_agent import DQNAgent, DQNConfig
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, MapState
from src.training.train import run_dqn_training


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


@dataclass
class TinyLineWorldEnv:
    """Small deterministic environment for offline DQN tests."""

    action_space: tuple[str, ...] = ("move_left", "move_right", "wait")

    def __post_init__(self) -> None:
        self._episode_counter = 0
        self.current_episode_id: str | None = None
        self._x = 0
        self._steps = 0

    def reset(self) -> GameStateSnapshot:
        self._episode_counter += 1
        self.current_episode_id = f"episode-{self._episode_counter:05d}"
        self._x = 0
        self._steps = 0
        return self._snapshot(failed=False)

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, object]]:
        if action == "move_right":
            self._x = min(3, self._x + 1)
            reward = 0.6
        elif action == "move_left":
            self._x = max(0, self._x - 1)
            reward = -0.3
        else:
            reward = -0.1

        self._steps += 1
        reached_exit = self._x == 3
        timed_out = self._steps >= 6 and not reached_exit
        done = reached_exit or timed_out
        if reached_exit:
            reward += 2.0

        terminal_reason = "goal" if reached_exit else ("timeout" if timed_out else None)
        return (self._snapshot(failed=timed_out), reward, done, {"terminal_reason": terminal_reason})

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        del state
        return self.action_space

    def close(self) -> None:
        return

    def _snapshot(self, *, failed: bool) -> GameStateSnapshot:
        return GameStateSnapshot(
            timestamp_utc="2026-03-06T00:00:00+00:00",
            health=_field(10),
            energy=_field(5),
            currency=_field(self._x),
            fail_state=_field(failed),
            map=MapState(
                status="ok",
                width=4,
                height=1,
                player_position=GridPosition(self._x, 0),
                exit_position=GridPosition(3, 0),
            ),
        )


def _test_config() -> DQNConfig:
    return DQNConfig(
        gamma=0.95,
        learning_rate=0.05,
        replay_capacity=128,
        min_replay_size=8,
        batch_size=8,
        target_sync_interval=4,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_steps=24,
    )


def test_dqn_training_runs_multiple_episodes_and_updates() -> None:
    env = TinyLineWorldEnv()
    agent = DQNAgent(action_space=env.action_space, config=_test_config(), seed=7)

    results = run_dqn_training(
        env=env,
        agent=agent,
        episodes=6,
        max_steps_per_episode=10,
        explore=True,
    )

    assert len(results) == 6
    assert all(result.steps >= 1 for result in results)
    assert any(result.updates_applied > 0 for result in results)
    assert results[0].epsilon > results[-1].epsilon

    expected_steps = sum(result.steps for result in results)
    state = agent.training_state
    assert state.episodes_seen == 6
    assert state.total_env_steps == expected_steps
    assert state.optimization_steps > 0
    assert state.last_loss is not None


def test_dqn_checkpoint_round_trip_and_resume_training(tmp_path: Path) -> None:
    env = TinyLineWorldEnv()
    agent = DQNAgent(action_space=env.action_space, config=_test_config(), seed=11)

    _ = run_dqn_training(
        env=env,
        agent=agent,
        episodes=4,
        max_steps_per_episode=8,
        explore=True,
    )
    before = agent.training_state

    checkpoint = agent.save_checkpoint(
        tmp_path / "dqn_checkpoint.json",
        metadata={"phase": "unit-test"},
    )
    loaded = DQNAgent.load_checkpoint(checkpoint)

    assert loaded.training_state == before
    assert loaded.checkpoint_metadata == {"phase": "unit-test"}
    assert loaded.action_space == agent.action_space
    assert loaded.epsilon == agent.epsilon

    _ = run_dqn_training(
        env=TinyLineWorldEnv(),
        agent=loaded,
        episodes=2,
        max_steps_per_episode=8,
        explore=True,
    )

    resumed = loaded.training_state
    assert resumed.episodes_seen == before.episodes_seen + 2
    assert resumed.total_env_steps > before.total_env_steps
    assert resumed.optimization_steps >= before.optimization_steps


def test_dqn_training_eval_mode_does_not_mutate_training_state() -> None:
    env = TinyLineWorldEnv()
    agent = DQNAgent(action_space=env.action_space, config=_test_config(), seed=17)

    _ = run_dqn_training(
        env=env,
        agent=agent,
        episodes=3,
        max_steps_per_episode=8,
        explore=True,
        learn=True,
    )
    before = agent.training_state

    results = run_dqn_training(
        env=TinyLineWorldEnv(),
        agent=agent,
        episodes=2,
        max_steps_per_episode=8,
        explore=False,
        learn=False,
    )

    assert len(results) == 2
    assert all(item.updates_applied == 0 for item in results)
    assert all(item.last_loss is None for item in results)
    assert agent.training_state == before

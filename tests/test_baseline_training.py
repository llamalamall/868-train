"""Tests for baseline rollout/evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.baseline_heuristic import HeuristicBaselineAgent
from src.agent.baseline_random import RandomBaselineAgent
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, MapState
from src.training.baselines import compare_baselines
from src.training.rollouts import run_agent_policy


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


@dataclass
class TinyLineWorldEnv:
    """Deterministic toy env for baseline-comparison tests."""

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
            self._x = min(2, self._x + 1)
            reward = 1.0
        elif action == "move_left":
            self._x = max(0, self._x - 1)
            reward = -0.2
        else:
            reward = -0.1

        self._steps += 1
        reached_exit = self._x == 2
        timed_out = self._steps >= 4 and not reached_exit
        done = reached_exit or timed_out
        terminal_reason = "goal" if reached_exit else ("timeout" if timed_out else None)
        if reached_exit:
            reward += 3.0

        return (
            self._snapshot(failed=timed_out),
            reward,
            done,
            {"terminal_reason": terminal_reason},
        )

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        del state
        return self.action_space

    def close(self) -> None:
        return

    def _snapshot(self, *, failed: bool) -> GameStateSnapshot:
        return GameStateSnapshot(
            timestamp_utc="2026-03-06T00:00:00+00:00",
            health=_field(10),
            energy=_field(0),
            currency=_field(0),
            fail_state=_field(failed),
            map=MapState(
                status="ok",
                width=3,
                height=1,
                player_position=GridPosition(self._x, 0),
                exit_position=GridPosition(2, 0),
            ),
        )


@dataclass
class NoAvailableActionsEnv:
    """Env that reports no available actions to exercise random fallback path."""

    action_space: tuple[str, ...] = ("move_left", "move_right")

    def __post_init__(self) -> None:
        self.current_episode_id = "episode-00001"
        self.actions_seen: list[str] = []
        self._step_count = 0

    def reset(self) -> GameStateSnapshot:
        self._step_count = 0
        return GameStateSnapshot(
            timestamp_utc="2026-03-06T00:00:00+00:00",
            health=_field(10),
            energy=_field(0),
            currency=_field(0),
            fail_state=_field(False),
            map=MapState(status="missing"),
        )

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, object]]:
        self.actions_seen.append(action)
        self._step_count += 1
        done = self._step_count >= 1
        return (
            GameStateSnapshot(
                timestamp_utc="2026-03-06T00:00:00+00:00",
                health=_field(10),
                energy=_field(0),
                currency=_field(0),
                fail_state=_field(False),
                map=MapState(status="missing"),
            ),
            0.0,
            done,
            {"terminal_reason": "done" if done else None},
        )

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        del state
        return ()

    def close(self) -> None:
        return


def test_run_agent_policy_completes_full_episodes() -> None:
    env = TinyLineWorldEnv()
    results = run_agent_policy(
        env=env,
        agent=HeuristicBaselineAgent(),
        episodes=3,
        max_steps_per_episode=5,
        seed=123,
    )

    assert len(results) == 3
    assert all(result.done for result in results)
    assert all(result.steps == 2 for result in results)
    assert all(result.terminal_reason == "goal" for result in results)


def test_compare_baselines_reports_heuristic_outperforming_random() -> None:
    metrics = compare_baselines(
        env_factory=TinyLineWorldEnv,
        baselines=(
            ("random", RandomBaselineAgent()),
            ("heuristic", HeuristicBaselineAgent()),
        ),
        episodes=50,
        max_steps_per_episode=4,
        seed=9,
    )

    by_name = {item.policy_name: item for item in metrics}
    random_metrics = by_name["random"]
    heuristic_metrics = by_name["heuristic"]

    assert heuristic_metrics.avg_reward > random_metrics.avg_reward
    assert heuristic_metrics.avg_steps < random_metrics.avg_steps
    assert heuristic_metrics.terminal_reasons.get("goal", 0) >= random_metrics.terminal_reasons.get("goal", 0)


def test_run_agent_policy_falls_back_to_random_action_when_none_available() -> None:
    env = NoAvailableActionsEnv()

    results = run_agent_policy(
        env=env,
        agent=HeuristicBaselineAgent(),
        episodes=1,
        max_steps_per_episode=5,
        seed=11,
    )

    assert len(results) == 1
    assert len(env.actions_seen) == 1
    assert env.actions_seen[0] in env.action_space


def test_run_agent_policy_step_callback_reports_action_and_reason() -> None:
    env = TinyLineWorldEnv()
    events: list[dict[str, object]] = []

    _ = run_agent_policy(
        env=env,
        agent=HeuristicBaselineAgent(),
        episodes=1,
        max_steps_per_episode=5,
        seed=3,
        step_callback=lambda event: events.append(event),
    )

    assert events
    first = events[0]
    assert isinstance(first["action"], str)
    assert isinstance(first["action_reason"], str)
    assert first["action_reason"] != ""


def test_run_agent_policy_before_step_callback_reports_pending_action() -> None:
    env = TinyLineWorldEnv()
    events: list[dict[str, object]] = []

    _ = run_agent_policy(
        env=env,
        agent=HeuristicBaselineAgent(),
        episodes=1,
        max_steps_per_episode=2,
        seed=5,
        before_step_callback=lambda event: events.append(event),
    )

    assert events
    first = events[0]
    assert isinstance(first["action"], str)
    assert isinstance(first["action_reason"], str)
    assert isinstance(first["total_reward"], float)

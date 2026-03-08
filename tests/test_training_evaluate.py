"""Tests for Task-15 DQN evaluation harness and KPI comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.dqn_agent import DQNAgent, DQNConfig
from src.state.schema import FieldState, GameStateSnapshot, MapState
from src.training.evaluate import (
    _build_parser,
    compare_dqn_checkpoints,
    comparison_report_to_json,
    evaluate_dqn_checkpoint,
    evaluation_report_to_json,
    format_comparison_table,
    format_evaluation_table,
)


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


@dataclass
class SeededLineWorldEnv:
    """Deterministic env with seed-aware resets for evaluation reproducibility tests."""

    action_space: tuple[str, ...] = ("move_left", "move_right")

    def __post_init__(self) -> None:
        self.current_episode_id: str | None = None
        self._episode_counter = 0
        self._rng_state = 0
        self._x = 0
        self._steps = 0
        self._health = 0
        self._currency = 0

    def seed(self, seed: int) -> None:
        self._rng_state = int(seed)

    def reset(self, seed: int | None = None) -> GameStateSnapshot:
        if seed is not None:
            self.seed(seed)
        self._episode_counter += 1
        self.current_episode_id = f"episode-{self._episode_counter:05d}"
        self._steps = 0
        self._x = self._rng_state % 2
        self._health = 4 + (self._rng_state % 3)
        self._currency = self._rng_state % 2
        return self._snapshot(failed=False)

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, object]]:
        if action == "move_right":
            self._x = min(3, self._x + 1)
            self._currency += 1
        elif action == "move_left":
            self._x = max(0, self._x - 1)
            self._health -= 1
        else:
            raise ValueError(f"Unsupported action for SeededLineWorldEnv: {action}")

        self._steps += 1
        reached_goal = self._x == 3
        failed = self._health <= 0
        timed_out = self._steps >= 5 and not reached_goal and not failed
        done = reached_goal or failed or timed_out
        terminal_reason = "goal" if reached_goal else ("fail_state" if done else None)
        failed_state = failed or timed_out
        return self._snapshot(failed=failed_state), 0.0, done, {"terminal_reason": terminal_reason}

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        del state
        return self.action_space

    def close(self) -> None:
        return

    def _snapshot(self, *, failed: bool) -> GameStateSnapshot:
        return GameStateSnapshot(
            timestamp_utc="2026-03-07T00:00:00+00:00",
            health=_field(self._health),
            energy=_field(0),
            currency=_field(self._currency),
            fail_state=_field(failed),
            map=MapState(status="missing"),
        )


def _build_checkpoint(tmp_path: Path, *, name: str, prefer: str) -> Path:
    if prefer not in {"move_left", "move_right"}:
        raise ValueError("prefer must be move_left or move_right.")

    config = DQNConfig(
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    agent = DQNAgent(action_space=("move_left", "move_right"), config=config, seed=7)
    left_index = agent.action_space.index("move_left")
    right_index = agent.action_space.index("move_right")
    if prefer == "move_right":
        agent._online_model.bias[left_index] = -0.5
        agent._online_model.bias[right_index] = 1.5
    else:
        agent._online_model.bias[left_index] = 1.5
        agent._online_model.bias[right_index] = -0.5
    agent._target_model.copy_from(agent._online_model)

    checkpoint_path = tmp_path / f"{name}.json"
    return agent.save_checkpoint(checkpoint_path, metadata={"name": name, "preferred_action": prefer})


def test_evaluate_dqn_checkpoint_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    checkpoint = _build_checkpoint(tmp_path, name="prefer-right", prefer="move_right")

    report_a = evaluate_dqn_checkpoint(
        env_factory=SeededLineWorldEnv,
        checkpoint_path=checkpoint,
        episodes=6,
        max_steps_per_episode=5,
        seed=123,
    )
    report_b = evaluate_dqn_checkpoint(
        env_factory=SeededLineWorldEnv,
        checkpoint_path=checkpoint,
        episodes=6,
        max_steps_per_episode=5,
        seed=123,
    )

    assert report_a == report_b
    assert report_a.fail_rate <= 0.5
    assert report_a.avg_currency_gain > 0.0
    assert report_a.avg_energy_gain == 0.0
    assert report_a.avg_score_gain == 0.0
    assert report_a.avg_prog_gains == 0.0


def test_compare_dqn_checkpoints_reports_direct_kpi_deltas(tmp_path: Path) -> None:
    checkpoint_a = _build_checkpoint(tmp_path, name="prefer-left", prefer="move_left")
    checkpoint_b = _build_checkpoint(tmp_path, name="prefer-right", prefer="move_right")

    comparison = compare_dqn_checkpoints(
        env_factory=SeededLineWorldEnv,
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        episodes=8,
        max_steps_per_episode=5,
        seed=99,
        label_a="left",
        label_b="right",
    )

    assert comparison.fail_rate_delta < 0.0
    assert comparison.avg_episode_length_delta < 0.0
    assert comparison.avg_health_delta_delta > 0.0
    assert comparison.avg_currency_gain_delta > 0.0
    assert comparison.avg_energy_gain_delta == 0.0
    assert comparison.avg_score_gain_delta == 0.0
    assert comparison.avg_prog_gains_delta == 0.0


def test_task15_outputs_include_json_and_table_formats(tmp_path: Path) -> None:
    checkpoint_a = _build_checkpoint(tmp_path, name="left", prefer="move_left")
    checkpoint_b = _build_checkpoint(tmp_path, name="right", prefer="move_right")

    report = evaluate_dqn_checkpoint(
        env_factory=SeededLineWorldEnv,
        checkpoint_path=checkpoint_b,
        episodes=4,
        max_steps_per_episode=5,
        seed=5,
        label="checkpoint-right",
    )
    comparison = compare_dqn_checkpoints(
        env_factory=SeededLineWorldEnv,
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        episodes=4,
        max_steps_per_episode=5,
        seed=5,
    )

    report_json = evaluation_report_to_json(report)
    comparison_json = comparison_report_to_json(comparison)
    report_table = format_evaluation_table(report)
    comparison_table = format_comparison_table(comparison)

    assert report_json["kind"] == "dqn_evaluation"
    assert report_json["fail_rate"] == report.fail_rate
    assert report_json["avg_episode_length"] == report.avg_episode_length
    assert comparison_json["kind"] == "dqn_checkpoint_comparison"
    assert comparison_json["avg_currency_gain_delta"] == comparison.avg_currency_gain_delta
    assert "avg_currency_gain" in report_table
    assert "avg_score_gain" in report_table
    assert "safe_step_rate" in report_table
    assert "delta_b_minus_a" in comparison_table
    assert "avg_prog_gains" in comparison_table
    assert "harvest_progress_rate" in comparison_table


def test_compare_parser_defaults_window_targeted_input_without_focus() -> None:
    parser = _build_parser()
    args = parser.parse_args(["compare", "--checkpoint-a", "a.json", "--checkpoint-b", "b.json"])

    assert args.focus_window is False
    assert args.window_input is True
    assert args.tui is True
    assert args.reward_energy_delta == 0.02
    assert args.reward_score_delta == 0.01

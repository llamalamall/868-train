"""Evaluation helpers for baseline and DQN policy checkpoints."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Protocol

from src.agent.dqn_agent import DQNAgent
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import _build_action_config, _build_reward_config, _build_reward_fn
from src.env.runner_tui import RunnerTuiSession
from src.state.schema import FieldState, GameStateSnapshot, GridPosition
from src.training.rewards import RewardWeights
from src.training.train import BaselineAgent, EpisodeEnv, EpisodeRolloutResult, run_agent_policy

EvaluationEventCallback = Callable[[dict[str, Any]], None]
_ENEMY_TYPE_VIRUS = 2


def _format_monitor_actions(actions: object, *, limit: int = 8) -> str:
    if not isinstance(actions, (tuple, list)):
        return "-"
    normalized = tuple(str(item).strip() for item in actions if str(item).strip())
    if not normalized:
        return "-"
    if len(normalized) <= limit:
        return ",".join(normalized)
    remaining = len(normalized) - limit
    return "{head},...(+{remaining})".format(
        head=",".join(normalized[:limit]),
        remaining=remaining,
    )


class ClosableEpisodeEnv(EpisodeEnv, Protocol):
    """Episode environment with optional close hook."""

    def close(self) -> None:
        """Release environment resources."""


@dataclass(frozen=True)
class BaselineMetrics:
    """Comparable KPI summary for one baseline policy."""

    policy_name: str
    episodes: int
    avg_steps: float
    avg_reward: float
    done_rate: float
    terminal_reasons: dict[str, int]


@dataclass(frozen=True)
class DQNEpisodeKpi:
    """Per-episode KPI values used in checkpoint-level evaluation reports."""

    episode_id: str
    steps: int
    done: bool
    terminal_reason: str | None
    failed: bool
    health_delta: float
    currency_gain: float
    energy_gain: float
    score_gain: float
    prog_gains: int
    map_clear_count: int
    premature_exit_attempts: int
    invalid_action_count: int
    safe_steps: int
    threatened_steps: int
    harvest_progress_steps: int
    siphons_collected: int
    enemies_cleared: int


@dataclass(frozen=True)
class DQNEvaluationReport:
    """Aggregate KPI report for a single DQN checkpoint evaluation."""

    label: str
    checkpoint_path: str
    episodes: int
    max_steps_per_episode: int
    seed: int | None
    fail_rate: float
    avg_episode_length: float
    avg_health_delta: float
    avg_currency_gain: float
    avg_energy_gain: float
    avg_score_gain: float
    avg_prog_gains: float
    map_clear_rate: float
    premature_exit_rate: float
    invalid_action_rate: float
    safe_step_rate: float
    threatened_step_rate: float
    harvest_progress_rate: float
    avg_siphons_collected_per_map: float
    avg_enemies_cleared_per_map: float
    terminal_reasons: dict[str, int]
    episode_results: tuple[DQNEpisodeKpi, ...]


@dataclass(frozen=True)
class DQNCheckpointComparison:
    """Side-by-side KPI comparison for two evaluated checkpoints."""

    checkpoint_a: DQNEvaluationReport
    checkpoint_b: DQNEvaluationReport
    fail_rate_delta: float
    avg_episode_length_delta: float
    avg_health_delta_delta: float
    avg_currency_gain_delta: float
    avg_energy_gain_delta: float
    avg_score_gain_delta: float
    avg_prog_gains_delta: float
    map_clear_rate_delta: float
    premature_exit_rate_delta: float
    invalid_action_rate_delta: float
    safe_step_rate_delta: float
    threatened_step_rate_delta: float
    harvest_progress_rate_delta: float
    avg_siphons_collected_per_map_delta: float
    avg_enemies_cleared_per_map_delta: float


def summarize_rollouts(*, policy_name: str, results: tuple[EpisodeRolloutResult, ...]) -> BaselineMetrics:
    """Convert episode-level rollout output into aggregate metrics."""
    if not results:
        raise ValueError("results must not be empty.")

    terminal_counts: dict[str, int] = {}
    for result in results:
        if result.terminal_reason:
            terminal_counts[result.terminal_reason] = terminal_counts.get(result.terminal_reason, 0) + 1

    return BaselineMetrics(
        policy_name=policy_name,
        episodes=len(results),
        avg_steps=mean(result.steps for result in results),
        avg_reward=mean(result.total_reward for result in results),
        done_rate=sum(1 for result in results if result.done) / len(results),
        terminal_reasons=terminal_counts,
    )


def evaluate_baseline(
    *,
    env: EpisodeEnv,
    policy_name: str,
    agent: BaselineAgent,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
) -> BaselineMetrics:
    """Run one baseline policy and return comparable metrics."""
    results = run_agent_policy(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )
    return summarize_rollouts(policy_name=policy_name, results=results)


def compare_baselines(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    baselines: tuple[tuple[str, BaselineAgent], ...],
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
) -> tuple[BaselineMetrics, ...]:
    """Evaluate multiple baselines under matching rollout settings."""
    if not baselines:
        raise ValueError("baselines must include at least one policy.")

    metrics: list[BaselineMetrics] = []
    for baseline_index, (policy_name, agent) in enumerate(baselines):
        env = env_factory()
        try:
            # Offset seed so each policy has deterministic but independent action sampling.
            policy_seed = None if seed is None else seed + baseline_index
            metrics.append(
                evaluate_baseline(
                    env=env,
                    policy_name=policy_name,
                    agent=agent,
                    episodes=episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    seed=policy_seed,
                )
            )
        finally:
            env.close()

    return tuple(metrics)


def _validate_rollout_settings(*, episodes: int, max_steps_per_episode: int) -> None:
    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be >= 1.")


def _field_to_float(field: FieldState) -> float | None:
    if field.status != "ok":
        return None
    value = field.value
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_or_zero(*, initial: float | None, final: float | None) -> float:
    if initial is None or final is None:
        return 0.0
    return float(final - initial)


def _count_siphons(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return len(state.map.siphons)


def _count_live_enemies(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def _enemy_identity_counts(state: GameStateSnapshot) -> Counter[int] | None:
    if state.map.status != "ok":
        return None
    return Counter(int(enemy.slot) for enemy in state.map.enemies if enemy.in_bounds)


def _enemy_cleared_delta(*, previous: GameStateSnapshot, current: GameStateSnapshot) -> int:
    previous_counts = _enemy_identity_counts(previous)
    current_counts = _enemy_identity_counts(current)
    if previous_counts is None or current_counts is None:
        return 0

    cleared = 0
    for enemy_id, previous_count in previous_counts.items():
        current_count = current_counts.get(enemy_id, 0)
        if previous_count > current_count:
            cleared += previous_count - current_count
    return int(cleared)


def _state_extra_float(state: GameStateSnapshot, *, key: str) -> float | None:
    field = state.extra_fields.get(key)
    if field is None:
        return None
    return _field_to_float(field)


def _inventory_counts(state: GameStateSnapshot) -> Counter[int]:
    if state.inventory.status != "ok":
        return Counter()
    return Counter(int(prog_id) for prog_id in state.inventory.raw_prog_ids)


def _prog_gain_delta(*, previous: GameStateSnapshot, current: GameStateSnapshot) -> int:
    previous_counts = _inventory_counts(previous)
    current_counts = _inventory_counts(current)
    if not previous_counts and not current_counts:
        return 0

    gained = 0
    for prog_id, current_count in current_counts.items():
        previous_count = previous_counts.get(prog_id, 0)
        if current_count > previous_count:
            gained += current_count - previous_count
    return gained


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _chebyshev(a: GridPosition, b: GridPosition) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def _harvest_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    if state.map.status != "ok":
        return ()

    targets: set[GridPosition] = set()
    for resource in state.map.resource_cells:
        if resource.credits > 0 or resource.energy > 0 or resource.points > 0:
            targets.add(resource.position)
    for wall in state.map.walls:
        if wall.prog_id is not None or wall.points > 0:
            targets.add(wall.position)
    if not state.map.walls:
        for cell in state.map.cells:
            if not cell.is_wall:
                continue
            if cell.prog_id is not None or cell.points > 0:
                targets.add(cell.position)
    return tuple(targets)


def _nearest_harvest_target_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None
    targets = _harvest_targets(state)
    if not targets:
        return None
    player = state.map.player_position
    return min(_manhattan(player, target) for target in targets)


def _enemy_can_attack_position(*, enemy_position: GridPosition, enemy_type_id: int, player_position: GridPosition) -> bool:
    if enemy_type_id == _ENEMY_TYPE_VIRUS:
        return _chebyshev(enemy_position, player_position) <= 1
    return _manhattan(enemy_position, player_position) <= 1


def _player_tile_threatened(state: GameStateSnapshot) -> bool | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None
    player = state.map.player_position
    enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
    if not enemies:
        return False
    return any(
        _enemy_can_attack_position(
            enemy_position=enemy.position,
            enemy_type_id=enemy.type_id,
            player_position=player,
        )
        for enemy in enemies
    )


def _player_on_exit(state: GameStateSnapshot) -> bool:
    if state.map.status != "ok":
        return False
    player = state.map.player_position
    exit_position = state.map.exit_position
    return player is not None and exit_position is not None and player == exit_position


def _episode_failed(*, final_state: GameStateSnapshot, terminal_reason: str | None) -> bool:
    if final_state.fail_state.status == "ok" and bool(final_state.fail_state.value):
        return True
    if terminal_reason is None:
        return False
    reason = terminal_reason.strip().lower()
    return any(token in reason for token in ("fail", "loss", "dead", "start_screen"))


def _reset_episode(env: EpisodeEnv, *, episode_seed: int | None) -> GameStateSnapshot:
    if episode_seed is None:
        return env.reset()

    reset_signature: inspect.Signature | None = None
    try:
        reset_signature = inspect.signature(env.reset)
    except (TypeError, ValueError):
        reset_signature = None
    if reset_signature is not None and "seed" in reset_signature.parameters:
        return env.reset(seed=episode_seed)  # type: ignore[call-arg]

    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        seed_fn(episode_seed)
    return env.reset()


def _resolve_eval_actions(*, env: EpisodeEnv, agent: DQNAgent, state: GameStateSnapshot) -> tuple[str, ...]:
    action_space = set(agent.action_space)
    available = tuple(action for action in env.available_actions(state) if action in action_space)
    if available:
        return available

    fallback = tuple(
        action
        for action in env.action_space
        if action in action_space and action not in {"cancel"}
    )
    if fallback:
        return fallback

    overlap = tuple(action for action in env.action_space if action in action_space)
    if overlap:
        return overlap
    raise ValueError("No overlapping actions between DQN checkpoint and environment action space.")


def summarize_dqn_episodes(
    *,
    label: str,
    checkpoint_path: str | Path,
    episodes: tuple[DQNEpisodeKpi, ...],
    max_steps_per_episode: int,
    seed: int | None,
) -> DQNEvaluationReport:
    """Summarize per-episode KPI rows into one checkpoint report."""
    if not episodes:
        raise ValueError("episodes must not be empty.")

    terminal_counts: dict[str, int] = {}
    for episode in episodes:
        if episode.terminal_reason:
            terminal_counts[episode.terminal_reason] = (
                terminal_counts.get(episode.terminal_reason, 0) + 1
            )
    total_steps = sum(max(episode.steps, 0) for episode in episodes)
    total_map_clears = sum(max(episode.map_clear_count, 0) for episode in episodes)
    total_premature_exit = sum(max(episode.premature_exit_attempts, 0) for episode in episodes)
    total_invalid_actions = sum(max(episode.invalid_action_count, 0) for episode in episodes)
    total_safe_steps = sum(max(episode.safe_steps, 0) for episode in episodes)
    total_threatened_steps = sum(max(episode.threatened_steps, 0) for episode in episodes)
    total_harvest_progress_steps = sum(max(episode.harvest_progress_steps, 0) for episode in episodes)
    total_siphons_collected = sum(max(episode.siphons_collected, 0) for episode in episodes)
    total_enemies_cleared = sum(max(episode.enemies_cleared, 0) for episode in episodes)

    return DQNEvaluationReport(
        label=label,
        checkpoint_path=str(checkpoint_path),
        episodes=len(episodes),
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        fail_rate=sum(1 for episode in episodes if episode.failed) / len(episodes),
        avg_episode_length=mean(episode.steps for episode in episodes),
        avg_health_delta=mean(episode.health_delta for episode in episodes),
        avg_currency_gain=mean(episode.currency_gain for episode in episodes),
        avg_energy_gain=mean(episode.energy_gain for episode in episodes),
        avg_score_gain=mean(episode.score_gain for episode in episodes),
        avg_prog_gains=mean(float(episode.prog_gains) for episode in episodes),
        map_clear_rate=sum(1 for episode in episodes if episode.map_clear_count > 0) / len(episodes),
        premature_exit_rate=(
            float(total_premature_exit) / float(total_steps)
            if total_steps > 0
            else 0.0
        ),
        invalid_action_rate=(
            float(total_invalid_actions) / float(total_steps)
            if total_steps > 0
            else 0.0
        ),
        safe_step_rate=(
            float(total_safe_steps) / float(total_steps)
            if total_steps > 0
            else 0.0
        ),
        threatened_step_rate=(
            float(total_threatened_steps) / float(total_steps)
            if total_steps > 0
            else 0.0
        ),
        harvest_progress_rate=(
            float(total_harvest_progress_steps) / float(total_steps)
            if total_steps > 0
            else 0.0
        ),
        avg_siphons_collected_per_map=(
            float(total_siphons_collected) / float(total_map_clears)
            if total_map_clears > 0
            else 0.0
        ),
        avg_enemies_cleared_per_map=(
            float(total_enemies_cleared) / float(total_map_clears)
            if total_map_clears > 0
            else 0.0
        ),
        terminal_reasons=terminal_counts,
        episode_results=episodes,
    )


def evaluate_dqn_checkpoint(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    checkpoint_path: str | Path,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = 0,
    label: str | None = None,
    step_callback: EvaluationEventCallback | None = None,
    episode_callback: EvaluationEventCallback | None = None,
) -> DQNEvaluationReport:
    """Run fixed-seed, no-exploration episodes and compute Task-15 KPIs."""
    _validate_rollout_settings(episodes=episodes, max_steps_per_episode=max_steps_per_episode)

    checkpoint = Path(checkpoint_path)
    agent = DQNAgent.load_checkpoint(checkpoint)
    report_label = label or checkpoint.name
    per_episode: list[DQNEpisodeKpi] = []
    env = env_factory()
    try:
        if not (set(agent.action_space) & set(env.action_space)):
            raise ValueError(
                "No overlapping actions between loaded DQN checkpoint and environment action_space."
            )

        for episode_index in range(episodes):
            episode_seed = None if seed is None else int(seed) + episode_index
            state = _reset_episode(env, episode_seed=episode_seed)
            episode_id = env.current_episode_id or f"episode-{episode_index + 1:05d}"
            if episode_callback is not None:
                episode_callback(
                    {
                        "event_type": "episode_start",
                        "checkpoint_label": report_label,
                        "checkpoint_path": str(checkpoint),
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "episodes_total": episodes,
                        "max_steps_per_episode": max_steps_per_episode,
                        "seed": episode_seed,
                    }
                )

            initial_health = _field_to_float(state.health)
            initial_currency = _field_to_float(state.currency)
            initial_energy = _field_to_float(state.energy)
            initial_score = _state_extra_float(state, key="score")

            steps = 0
            done = False
            terminal_reason: str | None = None
            map_clear_count = 0
            premature_exit_attempts = 0
            invalid_action_count = 0
            safe_steps = 0
            threatened_steps = 0
            harvest_progress_steps = 0
            prog_gains = 0
            siphons_collected = 0
            enemies_cleared = 0
            while steps < max_steps_per_episode and not done:
                previous_state = state
                allowed_actions = _resolve_eval_actions(env=env, agent=agent, state=state)
                action = agent.select_action(
                    state=state,
                    available_actions=allowed_actions,
                    explore=False,
                )
                state, reward, done, info = env.step(action)
                steps += 1

                previous_siphons = _count_siphons(previous_state)
                current_siphons = _count_siphons(state)
                if previous_siphons is not None and current_siphons is not None:
                    siphons_collected += max(previous_siphons - current_siphons, 0)

                previous_enemies = _count_live_enemies(previous_state)
                current_enemies = _count_live_enemies(state)
                enemies_cleared += _enemy_cleared_delta(previous=previous_state, current=state)

                prog_gains += _prog_gain_delta(previous=previous_state, current=state)

                if bool(info.get("premature_exit_attempt", False)):
                    premature_exit_attempts += 1
                if info.get("action_effective") is False:
                    invalid_action_count += 1

                tile_threat = _player_tile_threatened(state)
                if tile_threat is True:
                    threatened_steps += 1
                elif tile_threat is False:
                    safe_steps += 1

                previous_harvest_distance = _nearest_harvest_target_distance(previous_state)
                current_harvest_distance = _nearest_harvest_target_distance(state)
                if (
                    previous_harvest_distance is not None
                    and current_harvest_distance is not None
                    and current_harvest_distance < previous_harvest_distance
                ):
                    harvest_progress_steps += 1

                current_map_clear = (
                    _player_on_exit(state)
                    and current_siphons is not None
                    and current_siphons == 0
                    and current_enemies is not None
                    and current_enemies == 0
                )
                previous_map_clear = (
                    _player_on_exit(previous_state)
                    and previous_siphons is not None
                    and previous_siphons == 0
                    and previous_enemies is not None
                    and previous_enemies == 0
                )
                if current_map_clear and not previous_map_clear:
                    map_clear_count += 1

                reason = info.get("terminal_reason")
                if isinstance(reason, str) and reason.strip():
                    terminal_reason = reason
                next_available_actions = tuple(env.available_actions(state))
                if step_callback is not None:
                    step_callback(
                        {
                            "event_type": "step",
                            "checkpoint_label": report_label,
                            "checkpoint_path": str(checkpoint),
                            "episode_id": episode_id,
                            "episode_index": episode_index,
                            "episodes_total": episodes,
                            "step_index": steps - 1,
                            "max_steps_per_episode": max_steps_per_episode,
                            "action": action,
                            "action_reason": getattr(agent, "last_decision_reason", None) or "greedy_q",
                            "reward": float(reward),
                            "done": bool(done),
                            "terminal_reason": terminal_reason,
                            "map_clear_count": map_clear_count,
                            "premature_exit_attempts": premature_exit_attempts,
                            "invalid_action_count": invalid_action_count,
                            "safe_steps": safe_steps,
                            "threatened_steps": threatened_steps,
                            "harvest_progress_steps": harvest_progress_steps,
                            "prog_gains": prog_gains,
                            "siphons_collected": siphons_collected,
                            "enemies_cleared": enemies_cleared,
                            "next_available_actions": next_available_actions,
                        }
                    )

            final_health = _field_to_float(state.health)
            final_currency = _field_to_float(state.currency)
            final_energy = _field_to_float(state.energy)
            final_score = _state_extra_float(state, key="score")
            episode_result = DQNEpisodeKpi(
                episode_id=episode_id,
                steps=steps,
                done=done,
                terminal_reason=terminal_reason,
                failed=_episode_failed(final_state=state, terminal_reason=terminal_reason),
                health_delta=_delta_or_zero(initial=initial_health, final=final_health),
                currency_gain=_delta_or_zero(initial=initial_currency, final=final_currency),
                energy_gain=_delta_or_zero(initial=initial_energy, final=final_energy),
                score_gain=_delta_or_zero(initial=initial_score, final=final_score),
                prog_gains=prog_gains,
                map_clear_count=map_clear_count,
                premature_exit_attempts=premature_exit_attempts,
                invalid_action_count=invalid_action_count,
                safe_steps=safe_steps,
                threatened_steps=threatened_steps,
                harvest_progress_steps=harvest_progress_steps,
                siphons_collected=siphons_collected,
                enemies_cleared=enemies_cleared,
            )
            per_episode.append(episode_result)
            if episode_callback is not None:
                episode_callback(
                    {
                        "event_type": "episode_end",
                        "checkpoint_label": report_label,
                        "checkpoint_path": str(checkpoint),
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "episodes_total": episodes,
                        "steps": episode_result.steps,
                        "done": episode_result.done,
                        "failed": episode_result.failed,
                        "terminal_reason": episode_result.terminal_reason,
                        "health_delta": episode_result.health_delta,
                        "currency_gain": episode_result.currency_gain,
                        "energy_gain": episode_result.energy_gain,
                        "score_gain": episode_result.score_gain,
                        "prog_gains": episode_result.prog_gains,
                        "map_clear_count": episode_result.map_clear_count,
                        "premature_exit_attempts": episode_result.premature_exit_attempts,
                        "invalid_action_count": episode_result.invalid_action_count,
                        "safe_steps": episode_result.safe_steps,
                        "threatened_steps": episode_result.threatened_steps,
                        "harvest_progress_steps": episode_result.harvest_progress_steps,
                        "siphons_collected": episode_result.siphons_collected,
                        "enemies_cleared": episode_result.enemies_cleared,
                    }
                )
    finally:
        env.close()

    return summarize_dqn_episodes(
        label=report_label,
        checkpoint_path=checkpoint,
        episodes=tuple(per_episode),
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )


def compare_dqn_checkpoints(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = 0,
    label_a: str | None = None,
    label_b: str | None = None,
) -> DQNCheckpointComparison:
    """Evaluate two checkpoints under matching fixed-seed settings and report KPI deltas."""
    report_a = evaluate_dqn_checkpoint(
        env_factory=env_factory,
        checkpoint_path=checkpoint_a,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        label=label_a,
    )
    report_b = evaluate_dqn_checkpoint(
        env_factory=env_factory,
        checkpoint_path=checkpoint_b,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        label=label_b,
    )
    return summarize_dqn_comparison(report_a=report_a, report_b=report_b)


def summarize_dqn_comparison(
    *,
    report_a: DQNEvaluationReport,
    report_b: DQNEvaluationReport,
) -> DQNCheckpointComparison:
    """Build comparison KPI deltas from two per-checkpoint evaluation reports."""
    return DQNCheckpointComparison(
        checkpoint_a=report_a,
        checkpoint_b=report_b,
        fail_rate_delta=report_b.fail_rate - report_a.fail_rate,
        avg_episode_length_delta=report_b.avg_episode_length - report_a.avg_episode_length,
        avg_health_delta_delta=report_b.avg_health_delta - report_a.avg_health_delta,
        avg_currency_gain_delta=report_b.avg_currency_gain - report_a.avg_currency_gain,
        avg_energy_gain_delta=report_b.avg_energy_gain - report_a.avg_energy_gain,
        avg_score_gain_delta=report_b.avg_score_gain - report_a.avg_score_gain,
        avg_prog_gains_delta=report_b.avg_prog_gains - report_a.avg_prog_gains,
        map_clear_rate_delta=report_b.map_clear_rate - report_a.map_clear_rate,
        premature_exit_rate_delta=report_b.premature_exit_rate - report_a.premature_exit_rate,
        invalid_action_rate_delta=report_b.invalid_action_rate - report_a.invalid_action_rate,
        safe_step_rate_delta=report_b.safe_step_rate - report_a.safe_step_rate,
        threatened_step_rate_delta=report_b.threatened_step_rate - report_a.threatened_step_rate,
        harvest_progress_rate_delta=report_b.harvest_progress_rate - report_a.harvest_progress_rate,
        avg_siphons_collected_per_map_delta=(
            report_b.avg_siphons_collected_per_map - report_a.avg_siphons_collected_per_map
        ),
        avg_enemies_cleared_per_map_delta=(
            report_b.avg_enemies_cleared_per_map - report_a.avg_enemies_cleared_per_map
        ),
    )


def evaluation_report_to_json(report: DQNEvaluationReport) -> dict[str, Any]:
    """Convert one checkpoint report to machine-readable JSON payload."""
    payload = asdict(report)
    payload["kind"] = "dqn_evaluation"
    return payload


def comparison_report_to_json(report: DQNCheckpointComparison) -> dict[str, Any]:
    """Convert checkpoint comparison report to machine-readable JSON payload."""
    payload = asdict(report)
    payload["kind"] = "dqn_checkpoint_comparison"
    return payload


def format_evaluation_table(report: DQNEvaluationReport) -> str:
    """Render one checkpoint report as a compact human-readable table."""
    lines = [
        (
            "label\tepisodes\tfail_rate\tavg_episode_length\tavg_health_delta\t"
            "avg_currency_gain\tavg_energy_gain\tavg_score_gain\tavg_prog_gains\t"
            "map_clear_rate\tpremature_exit_rate\tinvalid_action_rate\t"
            "safe_step_rate\tthreatened_step_rate\tharvest_progress_rate\t"
            "avg_siphons_collected_per_map\tavg_enemies_cleared_per_map"
        ),
        (
            f"{report.label}\t{report.episodes}\t{report.fail_rate:.2%}\t"
            f"{report.avg_episode_length:.2f}\t{report.avg_health_delta:.3f}\t"
            f"{report.avg_currency_gain:.3f}\t{report.avg_energy_gain:.3f}\t"
            f"{report.avg_score_gain:.3f}\t{report.avg_prog_gains:.3f}\t"
            f"{report.map_clear_rate:.2%}\t{report.premature_exit_rate:.2%}\t"
            f"{report.invalid_action_rate:.2%}\t{report.safe_step_rate:.2%}\t"
            f"{report.threatened_step_rate:.2%}\t{report.harvest_progress_rate:.2%}\t"
            f"{report.avg_siphons_collected_per_map:.3f}\t"
            f"{report.avg_enemies_cleared_per_map:.3f}"
        ),
    ]
    if report.terminal_reasons:
        reasons = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(report.terminal_reasons.items())
        )
        lines.append(f"terminal_reasons\t{reasons}")
    return "\n".join(lines)


def format_comparison_table(report: DQNCheckpointComparison) -> str:
    """Render checkpoint-vs-checkpoint KPI deltas in tabular form."""
    a = report.checkpoint_a
    b = report.checkpoint_b
    lines = [
        "metric\tcheckpoint_a\tcheckpoint_b\tdelta_b_minus_a",
        f"label\t{a.label}\t{b.label}\t-",
        f"fail_rate\t{a.fail_rate:.2%}\t{b.fail_rate:.2%}\t{report.fail_rate_delta:+.2%}",
        (
            f"avg_episode_length\t{a.avg_episode_length:.2f}\t{b.avg_episode_length:.2f}\t"
            f"{report.avg_episode_length_delta:+.2f}"
        ),
        (
            f"avg_health_delta\t{a.avg_health_delta:.3f}\t{b.avg_health_delta:.3f}\t"
            f"{report.avg_health_delta_delta:+.3f}"
        ),
        (
            f"avg_currency_gain\t{a.avg_currency_gain:.3f}\t{b.avg_currency_gain:.3f}\t"
            f"{report.avg_currency_gain_delta:+.3f}"
        ),
        (
            f"avg_energy_gain\t{a.avg_energy_gain:.3f}\t{b.avg_energy_gain:.3f}\t"
            f"{report.avg_energy_gain_delta:+.3f}"
        ),
        (
            f"avg_score_gain\t{a.avg_score_gain:.3f}\t{b.avg_score_gain:.3f}\t"
            f"{report.avg_score_gain_delta:+.3f}"
        ),
        (
            f"avg_prog_gains\t{a.avg_prog_gains:.3f}\t{b.avg_prog_gains:.3f}\t"
            f"{report.avg_prog_gains_delta:+.3f}"
        ),
        (
            f"map_clear_rate\t{a.map_clear_rate:.2%}\t{b.map_clear_rate:.2%}\t"
            f"{report.map_clear_rate_delta:+.2%}"
        ),
        (
            f"premature_exit_rate\t{a.premature_exit_rate:.2%}\t{b.premature_exit_rate:.2%}\t"
            f"{report.premature_exit_rate_delta:+.2%}"
        ),
        (
            f"invalid_action_rate\t{a.invalid_action_rate:.2%}\t{b.invalid_action_rate:.2%}\t"
            f"{report.invalid_action_rate_delta:+.2%}"
        ),
        (
            f"safe_step_rate\t{a.safe_step_rate:.2%}\t{b.safe_step_rate:.2%}\t"
            f"{report.safe_step_rate_delta:+.2%}"
        ),
        (
            f"threatened_step_rate\t{a.threatened_step_rate:.2%}\t{b.threatened_step_rate:.2%}\t"
            f"{report.threatened_step_rate_delta:+.2%}"
        ),
        (
            f"harvest_progress_rate\t{a.harvest_progress_rate:.2%}\t{b.harvest_progress_rate:.2%}\t"
            f"{report.harvest_progress_rate_delta:+.2%}"
        ),
        (
            "avg_siphons_collected_per_map\t"
            f"{a.avg_siphons_collected_per_map:.3f}\t{b.avg_siphons_collected_per_map:.3f}\t"
            f"{report.avg_siphons_collected_per_map_delta:+.3f}"
        ),
        (
            "avg_enemies_cleared_per_map\t"
            f"{a.avg_enemies_cleared_per_map:.3f}\t{b.avg_enemies_cleared_per_map:.3f}\t"
            f"{report.avg_enemies_cleared_per_map_delta:+.3f}"
        ),
    ]
    return "\n".join(lines)


def _add_shared_eval_args(
    parser: argparse.ArgumentParser,
    *,
    focus_window_default: bool,
    window_input_default: bool,
) -> None:
    default_weights = RewardWeights()
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument(
        "--launch-exe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch --exe when not already running before attempting attach.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of fixed-seed episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic eval runs.")
    parser.add_argument(
        "--movement-keys",
        choices=("arrows", "wasd", "numpad"),
        default="arrows",
        help="Movement key mapping profile.",
    )
    parser.add_argument(
        "--prog-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prog-slot actions (prog_slot_1..prog_slot_10 mapped to 1..0).",
    )
    parser.add_argument(
        "--reset-sequence",
        default="confirm",
        nargs="?",
        const="",
        help="Comma-separated reset actions; pass without value to disable reset actions.",
    )
    parser.add_argument(
        "--focus-window",
        action=argparse.BooleanOptionalAction,
        default=focus_window_default,
        help="Focus game window during attach/reacquire.",
    )
    parser.add_argument(
        "--window-input",
        action=argparse.BooleanOptionalAction,
        default=window_input_default,
        help="Use window-targeted PostMessage input instead of global SendInput.",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=3.0,
        help="Per-step watchdog timeout in seconds.",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=15.0,
        help="Reset watchdog timeout in seconds.",
    )
    parser.add_argument(
        "--prog-backoff-steps",
        type=int,
        default=3,
        help="Fallback prog-slot backoff steps after ineffective attempts.",
    )
    parser.add_argument(
        "--require-non-terminal-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reset() to observe a non-terminal state before stepping.",
    )
    parser.add_argument(
        "--reward-survival",
        type=float,
        default=default_weights.survival,
        help="Survival reward applied for non-terminal steps.",
    )
    parser.add_argument(
        "--reward-step-penalty",
        type=float,
        default=default_weights.step_penalty,
        help="Per-step penalty magnitude applied each transition.",
    )
    parser.add_argument(
        "--reward-health-delta",
        type=float,
        default=default_weights.health_delta,
        help="Weight multiplied by (current_health - previous_health).",
    )
    parser.add_argument(
        "--reward-currency-delta",
        type=float,
        default=default_weights.currency_delta,
        help="Weight multiplied by (current_currency - previous_currency).",
    )
    parser.add_argument(
        "--reward-energy-delta",
        type=float,
        default=default_weights.energy_delta,
        help="Weight multiplied by (current_energy - previous_energy).",
    )
    parser.add_argument(
        "--reward-score-delta",
        type=float,
        default=default_weights.score_delta,
        help="Weight multiplied by (current_score - previous_score) when score is available.",
    )
    parser.add_argument(
        "--reward-siphon-collected",
        type=float,
        default=default_weights.siphon_collected,
        help="Reward per siphon removed from map.",
    )
    parser.add_argument(
        "--reward-enemy-damaged",
        type=float,
        default=default_weights.enemy_damaged,
        help="Reward per enemy HP point reduced when an enemy survives the step.",
    )
    parser.add_argument(
        "--reward-enemy-cleared",
        type=float,
        default=default_weights.enemy_cleared,
        help="Reward per live enemy removed from map.",
    )
    parser.add_argument(
        "--reward-phase-progress",
        type=float,
        default=default_weights.phase_progress,
        help="Weight for progress toward active objective (siphon->enemy->exit).",
    )
    parser.add_argument(
        "--reward-backtrack-penalty",
        type=float,
        default=default_weights.backtrack_penalty,
        help="Penalty weight for increased distance from the active objective.",
    )
    parser.add_argument(
        "--reward-map-clear-bonus",
        type=float,
        default=default_weights.map_clear_bonus,
        help="Bonus when player reaches exit after all siphons/enemies are cleared.",
    )
    parser.add_argument(
        "--reward-premature-exit-penalty",
        type=float,
        default=default_weights.premature_exit_penalty,
        help="Penalty when stepping onto exit before objectives are complete.",
    )
    parser.add_argument(
        "--reward-invalid-action-penalty",
        type=float,
        default=default_weights.invalid_action_penalty,
        help="Penalty when action appears ineffective/invalid.",
    )
    parser.add_argument(
        "--reward-fail-penalty",
        type=float,
        default=default_weights.fail_penalty,
        help="Terminal fail penalty magnitude (applied as negative).",
    )
    parser.add_argument(
        "--reward-safe-tile-bonus",
        type=float,
        default=default_weights.safe_tile_bonus,
        help="Bonus on steps where the current tile is not threatened by enemies.",
    )
    parser.add_argument(
        "--reward-danger-tile-penalty",
        type=float,
        default=default_weights.danger_tile_penalty,
        help="Penalty on steps where the current tile is threatened by enemies.",
    )
    parser.add_argument(
        "--reward-resource-proximity",
        type=float,
        default=default_weights.resource_proximity,
        help="Reward per tile of progress toward nearest harvest target.",
    )
    parser.add_argument(
        "--reward-prog-collected-base",
        type=float,
        default=default_weights.prog_collected_base,
        help="Base reward for each newly collected prog before priority bonus.",
    )
    parser.add_argument(
        "--reward-points-collected",
        type=float,
        default=default_weights.points_collected,
        help="Reward per map-point unit removed from available board points.",
    )
    parser.add_argument(
        "--reward-damage-taken-penalty",
        type=float,
        default=default_weights.damage_taken_penalty,
        help="Penalty multiplier applied to negative health deltas.",
    )
    parser.add_argument(
        "--reward-sector-advance",
        type=float,
        default=default_weights.sector_advance,
        help="Reward per positive sector index transition.",
    )
    parser.add_argument(
        "--reward-clip-abs",
        type=float,
        default=5.0,
        help="Absolute value used to clip final reward per step.",
    )
    parser.add_argument(
        "--print-reward-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-step reward component breakdown during evaluation.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON summary output.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task-15 DQN evaluation harness with fixed-seed KPI reporting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Evaluate one DQN checkpoint.",
        description="Run fixed-seed evaluation for one checkpoint and print KPIs.",
    )
    _add_shared_eval_args(
        run_parser,
        focus_window_default=True,
        window_input_default=True,
    )
    run_parser.add_argument("--checkpoint", required=True, help="Path to DQN checkpoint JSON.")
    run_parser.add_argument("--label", default=None, help="Optional display label in KPI table.")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two DQN checkpoints under identical evaluation settings.",
        description="Run fixed-seed KPI evaluation for checkpoint A and B, then print deltas.",
    )
    _add_shared_eval_args(
        compare_parser,
        focus_window_default=False,
        window_input_default=True,
    )
    compare_parser.add_argument("--checkpoint-a", required=True, help="Baseline checkpoint path.")
    compare_parser.add_argument("--checkpoint-b", required=True, help="Candidate checkpoint path.")
    compare_parser.add_argument("--label-a", default=None, help="Optional table label for A.")
    compare_parser.add_argument("--label-b", default=None, help="Optional table label for B.")
    compare_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch live state monitor TUI during compare runs.",
    )
    compare_parser.add_argument(
        "--tui-interval",
        type=float,
        default=0.5,
        help="Polling interval for compare TUI state monitor (seconds).",
    )
    compare_parser.add_argument(
        "--external-status-file",
        default=None,
        help=(
            "Optional JSON file path to receive compare status updates for GUI monitoring "
            "without opening the external TUI."
        ),
    )
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.episodes) < 1:
        parser.error("--episodes must be >= 1.")
    if int(args.max_steps) < 1:
        parser.error("--max-steps must be >= 1.")
    if args.command == "compare" and float(args.tui_interval) <= 0:
        parser.error("--tui-interval must be > 0.")


def _build_live_env_factory(args: argparse.Namespace) -> Callable[[], GameEnv]:
    reset_sequence = tuple(
        action.strip() for action in str(args.reset_sequence).split(",") if action.strip()
    )
    reward_config = _build_reward_config(args)
    reward_fn = _build_reward_fn(
        reward_config=reward_config,
        print_breakdown=bool(args.print_reward_breakdown),
    )

    def _factory() -> GameEnv:
        return GameEnv.from_live_process(
            executable_name=str(args.exe),
            config=GameEnvConfig(
                step_timeout_seconds=float(args.step_timeout),
                reset_timeout_seconds=float(args.reset_timeout),
                prog_slot_backoff_steps=max(int(args.prog_backoff_steps), 0),
                require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
            ),
            reset_sequence=reset_sequence if reset_sequence else None,
            launch_process_if_missing=bool(args.launch_exe),
            focus_window_on_attach=bool(args.focus_window),
            window_targeted_input=bool(args.window_input),
            action_config=_build_action_config(
                str(args.movement_keys),
                include_prog_actions=bool(args.prog_actions),
            ),
            reward_fn=reward_fn,
        )

    return _factory


def _emit_json_output(payload: dict[str, Any], *, output_path: str | None) -> None:
    json_text = json.dumps(payload, indent=2, sort_keys=True)
    print()
    print(json_text)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{json_text}\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(parser, args)
    env_factory = _build_live_env_factory(args)

    if args.command == "run":
        report = evaluate_dqn_checkpoint(
            env_factory=env_factory,
            checkpoint_path=args.checkpoint,
            episodes=int(args.episodes),
            max_steps_per_episode=int(args.max_steps),
            seed=args.seed,
            label=args.label,
        )
        print(format_evaluation_table(report))
        _emit_json_output(evaluation_report_to_json(report), output_path=args.json_out)
        return

    if args.command == "compare":
        label_a = args.label_a or Path(str(args.checkpoint_a)).name
        label_b = args.label_b or Path(str(args.checkpoint_b)).name
        monitor_enabled = bool(args.tui) or bool(args.external_status_file)
        tui = RunnerTuiSession(
            executable_name=str(args.exe),
            enabled=monitor_enabled,
            interval_seconds=float(args.tui_interval),
            launch_monitor=bool(args.tui),
            external_status_file=(str(args.external_status_file) if args.external_status_file else None),
        )

        def _on_step(event: dict[str, Any]) -> None:
            tui.update(
                training_line=(
                    "compare checkpoint={label} episode={episode}/{episode_total} "
                    "step={step}/{step_max} reward={reward:.3f} done={done} terminal={terminal}".format(
                        label=event.get("checkpoint_label"),
                        episode=int(event.get("episode_index", 0)) + 1,
                        episode_total=int(event.get("episodes_total", 0)),
                        step=int(event.get("step_index", 0)) + 1,
                        step_max=int(event.get("max_steps_per_episode", 0)),
                        reward=float(event.get("reward", 0.0)),
                        done=bool(event.get("done", False)),
                        terminal=event.get("terminal_reason") or "-",
                    )
                ),
                action_line="action={action} reason={reason}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "greedy_q",
                ),
                next_available_actions_line="next_available_actions={actions}".format(
                    actions=_format_monitor_actions(event.get("next_available_actions")),
                ),
            )

        def _on_episode(event: dict[str, Any]) -> None:
            event_type = str(event.get("event_type", ""))
            if event_type == "episode_start":
                tui.update(
                    training_line=(
                        "compare checkpoint={label} episode={episode}/{episode_total} "
                        "status=running seed={seed}".format(
                            label=event.get("checkpoint_label"),
                            episode=int(event.get("episode_index", 0)) + 1,
                            episode_total=int(event.get("episodes_total", 0)),
                            seed=event.get("seed"),
                        )
                    ),
                    action_line="action=idle reason=episode_start",
                    next_available_actions_line="next_available_actions=-",
                )
                return
            if event_type == "episode_end":
                tui.update(
                    training_line=(
                        "compare checkpoint={label} episode={episode}/{episode_total} "
                        "steps={steps} failed={failed} health_delta={health:+.3f} "
                        "currency_gain={currency:+.3f} terminal={terminal}".format(
                            label=event.get("checkpoint_label"),
                            episode=int(event.get("episode_index", 0)) + 1,
                            episode_total=int(event.get("episodes_total", 0)),
                            steps=int(event.get("steps", 0)),
                            failed=bool(event.get("failed", False)),
                            health=float(event.get("health_delta", 0.0)),
                            currency=float(event.get("currency_gain", 0.0)),
                            terminal=event.get("terminal_reason") or "-",
                        )
                    ),
                    action_line="action=idle reason=episode_complete",
                    next_available_actions_line="next_available_actions=-",
                )

        try:
            tui.start()
            tui.update(
                training_line="compare checkpoint={label} status=starting".format(label=label_a),
                action_line="action=idle reason=checkpoint_start",
                next_available_actions_line="next_available_actions=-",
            )
            report_a = evaluate_dqn_checkpoint(
                env_factory=env_factory,
                checkpoint_path=args.checkpoint_a,
                episodes=int(args.episodes),
                max_steps_per_episode=int(args.max_steps),
                seed=args.seed,
                label=label_a,
                step_callback=_on_step if monitor_enabled else None,
                episode_callback=_on_episode if monitor_enabled else None,
            )
            tui.update(
                training_line="compare checkpoint={label} status=complete".format(label=label_a),
                action_line="action=idle reason=checkpoint_complete",
                next_available_actions_line="next_available_actions=-",
            )
            tui.update(
                training_line="compare checkpoint={label} status=starting".format(label=label_b),
                action_line="action=idle reason=checkpoint_start",
                next_available_actions_line="next_available_actions=-",
            )
            report_b = evaluate_dqn_checkpoint(
                env_factory=env_factory,
                checkpoint_path=args.checkpoint_b,
                episodes=int(args.episodes),
                max_steps_per_episode=int(args.max_steps),
                seed=args.seed,
                label=label_b,
                step_callback=_on_step if monitor_enabled else None,
                episode_callback=_on_episode if monitor_enabled else None,
            )
            tui.update(
                training_line="compare status=complete checkpoint_a={a} checkpoint_b={b}".format(
                    a=label_a,
                    b=label_b,
                ),
                action_line="action=idle reason=compare_complete",
                next_available_actions_line="next_available_actions=-",
            )
        finally:
            tui.close()

        report = summarize_dqn_comparison(report_a=report_a, report_b=report_b)
        print(format_comparison_table(report))
        _emit_json_output(comparison_report_to_json(report), output_path=args.json_out)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

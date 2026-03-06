"""Baseline evaluation helpers and comparable metrics summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Protocol

from src.training.train import BaselineAgent, EpisodeEnv, EpisodeRolloutResult, run_agent_policy


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

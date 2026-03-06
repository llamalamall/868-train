"""Training/evaluation rollout helpers for baseline agents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.state.schema import GameStateSnapshot


class BaselineAgent(Protocol):
    """Action-selection contract shared by baseline agents."""

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Select one action from the given action space."""


class EpisodeEnv(Protocol):
    """Minimal environment contract needed for baseline rollouts."""

    action_space: tuple[str, ...]
    current_episode_id: str | None

    def reset(self) -> GameStateSnapshot:
        """Reset and return initial state."""

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, Any]]:
        """Apply one action and return Gym-style step tuple."""


@dataclass(frozen=True)
class EpisodeRolloutResult:
    """Episode summary produced by baseline rollouts."""

    episode_id: str
    steps: int
    done: bool
    total_reward: float
    terminal_reason: str | None


def run_agent_policy(
    *,
    env: EpisodeEnv,
    agent: BaselineAgent,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
) -> tuple[EpisodeRolloutResult, ...]:
    """Run an action-selecting agent through full episodes."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be >= 1.")

    rng = random.Random(seed)
    results: list[EpisodeRolloutResult] = []

    for _ in range(episodes):
        state = env.reset()
        episode_id = env.current_episode_id or f"episode-{len(results) + 1:05d}"
        steps = 0
        done = False
        total_reward = 0.0
        terminal_reason: str | None = None

        while steps < max_steps_per_episode and not done:
            action = agent.select_action(state=state, action_space=env.action_space, rng=rng)
            if action not in env.action_space:
                raise ValueError(
                    f"Agent selected unknown action '{action}'. Allowed actions: {', '.join(env.action_space)}."
                )

            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            reason = info.get("terminal_reason")
            if isinstance(reason, str):
                terminal_reason = reason

        results.append(
            EpisodeRolloutResult(
                episode_id=episode_id,
                steps=steps,
                done=done,
                total_reward=total_reward,
                terminal_reason=terminal_reason,
            )
        )

    return tuple(results)

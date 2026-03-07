"""Training/evaluation rollout helpers for baseline agents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.agent.dqn_agent import DQNAgent
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

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        """Return currently valid actions, optionally based on provided state."""


@dataclass(frozen=True)
class EpisodeRolloutResult:
    """Episode summary produced by baseline rollouts."""

    episode_id: str
    steps: int
    done: bool
    total_reward: float
    terminal_reason: str | None


@dataclass(frozen=True)
class LearningEpisodeRolloutResult:
    """Episode summary emitted by the learning-agent rollout loop."""

    episode_id: str
    steps: int
    done: bool
    total_reward: float
    terminal_reason: str | None
    updates_applied: int
    last_loss: float | None
    epsilon: float


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
            available_actions = env.available_actions(state)
            if not available_actions:
                fallback_actions = tuple(action for action in env.action_space if action != "wait")
                if not fallback_actions:
                    raise ValueError("No available actions and env.action_space is empty.")
                action = str(rng.choice(fallback_actions))
            else:
                action = agent.select_action(state=state, action_space=available_actions, rng=rng)
                if action not in available_actions:
                    raise ValueError(
                        "Agent selected action '{action}' outside available_actions. Allowed actions: "
                        "{allowed}.".format(action=action, allowed=", ".join(available_actions))
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


def run_dqn_training(
    *,
    env: EpisodeEnv,
    agent: DQNAgent,
    episodes: int,
    max_steps_per_episode: int = 200,
    explore: bool = True,
) -> tuple[LearningEpisodeRolloutResult, ...]:
    """Train (or evaluate) the DQN agent through full episodes."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be >= 1.")

    results: list[LearningEpisodeRolloutResult] = []

    for _ in range(episodes):
        state = env.reset()
        episode_id = env.current_episode_id or f"episode-{len(results) + 1:05d}"
        steps = 0
        done = False
        total_reward = 0.0
        updates_applied = 0
        last_loss: float | None = None
        terminal_reason: str | None = None

        agent.start_episode()

        while steps < max_steps_per_episode and not done:
            available_actions = env.available_actions(state)
            if not available_actions:
                fallback_actions = tuple(action for action in env.action_space if action != "wait")
                if not fallback_actions:
                    raise ValueError("No available actions and env.action_space is empty.")
                available_actions = fallback_actions

            action = agent.select_action(
                state=state,
                available_actions=available_actions,
                explore=explore,
            )
            if action not in available_actions:
                raise ValueError(
                    "Agent selected action '{action}' outside available_actions. Allowed actions: "
                    "{allowed}.".format(action=action, allowed=", ".join(available_actions))
                )

            next_state, reward, done, info = env.step(action)
            update = agent.observe(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_available_actions=env.available_actions(next_state),
            )
            if update.did_update:
                updates_applied += 1
                last_loss = update.loss

            total_reward += float(reward)
            steps += 1
            state = next_state

            reason = info.get("terminal_reason")
            if isinstance(reason, str):
                terminal_reason = reason

        results.append(
            LearningEpisodeRolloutResult(
                episode_id=episode_id,
                steps=steps,
                done=done,
                total_reward=total_reward,
                terminal_reason=terminal_reason,
                updates_applied=updates_applied,
                last_loss=last_loss,
                epsilon=agent.epsilon,
            )
        )

    return tuple(results)

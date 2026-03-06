"""Random-action baseline agent."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.state.schema import GameStateSnapshot


@dataclass(frozen=True)
class RandomBaselineAgent:
    """Policy that samples uniformly from available actions."""

    preferred_actions: tuple[str, ...] | None = None

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Choose one action uniformly at random from allowed actions."""
        del state
        if not action_space:
            raise ValueError("action_space must include at least one action.")

        if self.preferred_actions is None:
            candidates = tuple(action_space)
        else:
            allowed = set(action_space)
            candidates = tuple(action for action in self.preferred_actions if action in allowed)
            if not candidates:
                raise ValueError("preferred_actions produced no valid actions for current action_space.")

        return str(rng.choice(candidates))

"""Top-level objective DQN for the hybrid hierarchical agent."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.hybrid.types import ObjectivePhase


class HybridTorchUnavailableError(RuntimeError):
    """Raised when a torch-dependent controller is used without torch installed."""


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as error:  # pragma: no cover - depends on local runtime.
        raise HybridTorchUnavailableError(
            "PyTorch is required for hybrid controllers. Install with `pip install torch`."
        ) from error
    return (torch, nn, optim)


def _linear_epsilon(*, start: float, end: float, decay_steps: int, steps: int) -> float:
    if decay_steps <= 0 or start <= end:
        return max(min(start, 1.0), 0.0)
    clipped_steps = min(max(steps, 0), decay_steps)
    ratio = float(clipped_steps) / float(decay_steps)
    return start + ((end - start) * ratio)


@dataclass(frozen=True)
class MetaDQNConfig:
    """Core hyperparameters for objective-level DQN."""

    gamma: float = 0.99
    learning_rate: float = 0.001
    replay_capacity: int = 12_000
    min_replay_size: int = 256
    batch_size: int = 64
    target_sync_interval: int = 250
    epsilon_start: float = 0.60
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    hidden_size: int = 64
    feature_count: int = 32


@dataclass(frozen=True)
class MetaTransition:
    """Replay transition for meta-controller updates."""

    features: tuple[float, ...]
    action_index: int
    reward: float
    next_features: tuple[float, ...]
    done: bool
    next_valid_action_indices: tuple[int, ...]


@dataclass(frozen=True)
class MetaUpdateResult:
    """Summary returned by one meta-controller observe call."""

    did_update: bool
    loss: float | None
    epsilon: float
    replay_size: int
    optimization_step: int


class _MetaQNetwork:
    """Compact MLP Q-network wrapper."""

    def __init__(
        self,
        *,
        torch_module: Any,
        nn_module: Any,
        feature_count: int,
        hidden_size: int,
        action_count: int,
    ) -> None:
        self._torch = torch_module
        self.model = nn_module.Sequential(
            nn_module.Linear(feature_count, hidden_size),
            nn_module.ReLU(),
            nn_module.Linear(hidden_size, hidden_size),
            nn_module.ReLU(),
            nn_module.Linear(hidden_size, action_count),
        )

    def q_values(self, features: tuple[float, ...]) -> tuple[float, ...]:
        with self._torch.no_grad():
            tensor = self._torch.tensor(features, dtype=self._torch.float32).unsqueeze(0)
            q_tensor = self.model(tensor).squeeze(0)
            return tuple(float(value) for value in q_tensor.tolist())


class MetaControllerDQN:
    """Objective-level controller trained with DQN."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        *,
        config: MetaDQNConfig = MetaDQNConfig(),
        seed: int | None = None,
    ) -> None:
        if config.feature_count < 1:
            raise ValueError("feature_count must be >= 1.")
        if config.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if config.replay_capacity < config.batch_size:
            raise ValueError("replay_capacity must be >= batch_size.")
        if config.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be >= 1.")
        if config.epsilon_decay_steps < 1:
            raise ValueError("epsilon_decay_steps must be >= 1.")
        if not (0.0 <= config.epsilon_end <= 1.0 and 0.0 <= config.epsilon_start <= 1.0):
            raise ValueError("epsilon values must be in [0, 1].")

        torch_module, nn_module, optim_module = _require_torch()
        self._torch = torch_module
        self._nn = nn_module
        self._optim = optim_module

        self.config = config
        self._rng = random.Random(seed)
        self._seed = seed
        self._phase_order = tuple(ObjectivePhase)
        self._phase_to_index = {phase: index for index, phase in enumerate(self._phase_order)}

        self._online = _MetaQNetwork(
            torch_module=self._torch,
            nn_module=self._nn,
            feature_count=self.config.feature_count,
            hidden_size=self.config.hidden_size,
            action_count=len(self._phase_order),
        )
        self._target = _MetaQNetwork(
            torch_module=self._torch,
            nn_module=self._nn,
            feature_count=self.config.feature_count,
            hidden_size=self.config.hidden_size,
            action_count=len(self._phase_order),
        )
        self._target.model.load_state_dict(self._online.model.state_dict())
        self._optimizer = self._optim.Adam(
            self._online.model.parameters(),
            lr=float(self.config.learning_rate),
        )
        self._loss_fn = self._nn.MSELoss()

        self._replay: list[MetaTransition] = []
        self._replay_cursor = 0
        self._total_env_steps = 0
        self._optimization_steps = 0
        self._episodes_seen = 0
        self.last_decision_reason: str | None = None

    @property
    def feature_count(self) -> int:
        return int(self.config.feature_count)

    @property
    def epsilon(self) -> float:
        return _linear_epsilon(
            start=float(self.config.epsilon_start),
            end=float(self.config.epsilon_end),
            decay_steps=int(self.config.epsilon_decay_steps),
            steps=int(self._total_env_steps),
        )

    def start_episode(self) -> None:
        self._episodes_seen += 1

    def select_objective(
        self,
        *,
        features: Sequence[float],
        allowed_phases: Sequence[ObjectivePhase] | None = None,
        explore: bool = True,
    ) -> tuple[ObjectivePhase, str, float | None]:
        vector = self._normalize_features(features)
        allowed = self._normalize_allowed_phases(allowed_phases)
        allowed_indices = tuple(self._phase_to_index[phase] for phase in allowed)
        if not allowed_indices:
            raise ValueError("No allowed phases supplied.")

        if explore and self._rng.random() < self.epsilon:
            phase = self._phase_order[self._rng.choice(allowed_indices)]
            self.last_decision_reason = "meta_epsilon_explore"
            return (phase, self.last_decision_reason, None)

        q_values = self._online.q_values(vector)
        best_index = max(allowed_indices, key=lambda index: q_values[index])
        phase = self._phase_order[best_index]
        self.last_decision_reason = "meta_greedy_q"
        return (phase, self.last_decision_reason, float(q_values[best_index]))

    def observe(
        self,
        *,
        features: Sequence[float],
        chosen_phase: ObjectivePhase,
        reward: float,
        next_features: Sequence[float],
        done: bool,
        next_allowed_phases: Sequence[ObjectivePhase] | None,
    ) -> MetaUpdateResult:
        if chosen_phase not in self._phase_to_index:
            raise ValueError(f"Unknown phase: {chosen_phase}")
        current_vector = self._normalize_features(features)
        next_vector = self._normalize_features(next_features)
        next_allowed = self._normalize_allowed_phases(next_allowed_phases)
        next_indices = tuple(self._phase_to_index[phase] for phase in next_allowed)
        transition = MetaTransition(
            features=current_vector,
            action_index=self._phase_to_index[chosen_phase],
            reward=float(reward),
            next_features=next_vector,
            done=bool(done),
            next_valid_action_indices=next_indices,
        )
        self._append_transition(transition)
        self._total_env_steps += 1

        minimum_replay = max(int(self.config.min_replay_size), int(self.config.batch_size))
        if len(self._replay) < minimum_replay:
            return MetaUpdateResult(
                did_update=False,
                loss=None,
                epsilon=self.epsilon,
                replay_size=len(self._replay),
                optimization_step=self._optimization_steps,
            )

        batch = self._rng.sample(self._replay, int(self.config.batch_size))
        loss = self._apply_update(batch=batch)
        self._optimization_steps += 1
        if self._optimization_steps % int(self.config.target_sync_interval) == 0:
            self._target.model.load_state_dict(self._online.model.state_dict())
        return MetaUpdateResult(
            did_update=True,
            loss=loss,
            epsilon=self.epsilon,
            replay_size=len(self._replay),
            optimization_step=self._optimization_steps,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "seed": self._seed,
            "config": asdict(self.config),
            "phase_order": [phase.value for phase in self._phase_order],
            "online_state_dict": self._online.model.state_dict(),
            "target_state_dict": self._target.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "training_state": {
                "total_env_steps": self._total_env_steps,
                "optimization_steps": self._optimization_steps,
                "episodes_seen": self._episodes_seen,
                "replay_cursor": self._replay_cursor,
            },
            "replay": [
                {
                    "features": list(item.features),
                    "action_index": int(item.action_index),
                    "reward": float(item.reward),
                    "next_features": list(item.next_features),
                    "done": bool(item.done),
                    "next_valid_action_indices": list(item.next_valid_action_indices),
                }
                for item in self._replay
            ],
        }

    def load_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        version = int(payload.get("version", -1))
        if version != self.CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported meta checkpoint version {version}. Expected {self.CHECKPOINT_VERSION}."
            )
        self._online.model.load_state_dict(payload["online_state_dict"])
        self._target.model.load_state_dict(payload["target_state_dict"])
        self._optimizer.load_state_dict(payload["optimizer_state_dict"])

        training = payload.get("training_state", {})
        self._total_env_steps = int(training.get("total_env_steps", 0))
        self._optimization_steps = int(training.get("optimization_steps", 0))
        self._episodes_seen = int(training.get("episodes_seen", 0))
        self._replay_cursor = int(training.get("replay_cursor", 0))

        parsed_replay: list[MetaTransition] = []
        raw_replay = payload.get("replay", [])
        if not isinstance(raw_replay, list):
            raise ValueError("Meta checkpoint replay must be a list.")
        for item in raw_replay:
            if not isinstance(item, dict):
                continue
            parsed_replay.append(
                MetaTransition(
                    features=self._normalize_features(item.get("features", [])),
                    action_index=int(item.get("action_index", 0)),
                    reward=float(item.get("reward", 0.0)),
                    next_features=self._normalize_features(item.get("next_features", [])),
                    done=bool(item.get("done", False)),
                    next_valid_action_indices=tuple(
                        int(value) for value in item.get("next_valid_action_indices", [])
                    ),
                )
            )
        if len(parsed_replay) > int(self.config.replay_capacity):
            parsed_replay = parsed_replay[-int(self.config.replay_capacity):]
        self._replay = parsed_replay
        if self._replay:
            self._replay_cursor %= len(self._replay)
        else:
            self._replay_cursor = 0

    def copy_weights_from(self, other: MetaControllerDQN) -> None:
        self._online.model.load_state_dict(other._online.model.state_dict())
        self._target.model.load_state_dict(other._target.model.state_dict())

    def training_snapshot(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "total_env_steps": self._total_env_steps,
            "optimization_steps": self._optimization_steps,
            "episodes_seen": self._episodes_seen,
            "replay_size": len(self._replay),
        }

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self.checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config_override: MetaDQNConfig | None = None,
    ) -> MetaControllerDQN:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Meta checkpoint not found: {checkpoint_path}")
        torch_module, _nn, _optim = _require_torch()
        payload = torch_module.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Meta checkpoint root must be an object.")
        config_payload = payload.get("config", {})
        if not isinstance(config_payload, dict):
            raise ValueError("Meta checkpoint config must be an object.")
        config = config_override or MetaDQNConfig(**config_payload)
        controller = cls(config=config, seed=payload.get("seed"))
        controller.load_checkpoint_payload(payload)
        return controller

    def _normalize_features(self, values: Sequence[float] | Any) -> tuple[float, ...]:
        floats = [float(value) for value in values]
        target_size = int(self.config.feature_count)
        if len(floats) < target_size:
            floats.extend(0.0 for _ in range(target_size - len(floats)))
        elif len(floats) > target_size:
            floats = floats[:target_size]
        return tuple(floats)

    def _normalize_allowed_phases(
        self,
        phases: Sequence[ObjectivePhase] | None,
    ) -> tuple[ObjectivePhase, ...]:
        if phases is None:
            return self._phase_order
        normalized: list[ObjectivePhase] = []
        seen: set[ObjectivePhase] = set()
        for phase in phases:
            parsed = phase if isinstance(phase, ObjectivePhase) else ObjectivePhase(str(phase))
            if parsed in seen:
                continue
            seen.add(parsed)
            normalized.append(parsed)
        return tuple(normalized)

    def _append_transition(self, transition: MetaTransition) -> None:
        capacity = int(self.config.replay_capacity)
        if len(self._replay) < capacity:
            self._replay.append(transition)
            return
        self._replay[self._replay_cursor] = transition
        self._replay_cursor = (self._replay_cursor + 1) % capacity

    def _apply_update(self, *, batch: list[MetaTransition]) -> float:
        states = self._torch.tensor([item.features for item in batch], dtype=self._torch.float32)
        actions = self._torch.tensor([item.action_index for item in batch], dtype=self._torch.int64)
        rewards = self._torch.tensor([item.reward for item in batch], dtype=self._torch.float32)
        next_states = self._torch.tensor([item.next_features for item in batch], dtype=self._torch.float32)
        dones = self._torch.tensor([item.done for item in batch], dtype=self._torch.float32)

        q_values = self._online.model(states)
        selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with self._torch.no_grad():
            next_q_values = self._target.model(next_states)
            max_next_q: list[float] = []
            for row, transition in enumerate(batch):
                if transition.done or not transition.next_valid_action_indices:
                    max_next_q.append(0.0)
                    continue
                allowed_values = [
                    float(next_q_values[row, index].item())
                    for index in transition.next_valid_action_indices
                ]
                max_next_q.append(max(allowed_values) if allowed_values else 0.0)
            next_q_tensor = self._torch.tensor(max_next_q, dtype=self._torch.float32)
            targets = rewards + ((1.0 - dones) * float(self.config.gamma) * next_q_tensor)

        loss = self._loss_fn(selected_q, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._torch.nn.utils.clip_grad_norm_(
            self._online.model.parameters(),
            max_norm=max(1.0, math.sqrt(float(self.config.hidden_size))),
        )
        self._optimizer.step()
        return float(loss.item())

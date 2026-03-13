"""Threat DRQN for low-level tactical overrides in the hybrid stack."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.hybrid.meta_controller import _linear_epsilon, _require_torch
from src.hybrid.types import ThreatOverride


@dataclass(frozen=True)
class ThreatDRQNConfig:
    """Core hyperparameters for threat DRQN."""

    gamma: float = 0.98
    learning_rate: float = 0.001
    replay_capacity: int = 20_000
    min_replay_size: int = 512
    batch_size: int = 64
    target_sync_interval: int = 300
    epsilon_start: float = 0.80
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25_000
    feature_count: int = 35
    hidden_size: int = 96
    sequence_length: int = 8


@dataclass(frozen=True)
class ThreatTransition:
    """Replay transition for DRQN updates."""

    features: tuple[float, ...]
    action_index: int
    reward: float
    next_features: tuple[float, ...]
    done: bool
    next_valid_action_indices: tuple[int, ...]


@dataclass(frozen=True)
class ThreatUpdateResult:
    """Summary returned by one threat-controller observe call."""

    did_update: bool
    loss: float | None
    epsilon: float
    replay_size: int
    optimization_step: int


class _ThreatQNetwork:
    """Sequence DRQN (encoder + GRU + Q head)."""

    def __init__(
        self,
        *,
        nn_module: Any,
        feature_count: int,
        hidden_size: int,
        action_count: int,
    ) -> None:
        self.model = nn_module.Sequential(
            nn_module.Linear(feature_count, hidden_size),
            nn_module.ReLU(),
        )
        self.gru = nn_module.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn_module.Linear(hidden_size, action_count)

    def parameters(self) -> Any:
        return list(self.model.parameters()) + list(self.gru.parameters()) + list(self.head.parameters())

    def state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.model.state_dict(),
            "gru": self.gru.state_dict(),
            "head": self.head.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["encoder"])
        self.gru.load_state_dict(state_dict["gru"])
        self.head.load_state_dict(state_dict["head"])

    def forward(
        self,
        *,
        features_tensor: Any,
        hidden_state: Any | None = None,
    ) -> tuple[Any, Any]:
        encoded = self.model(features_tensor)
        gru_out, new_hidden = self.gru(encoded, hidden_state)
        q_values = self.head(gru_out)
        return (q_values, new_hidden)


class ThreatControllerDRQN:
    """Threat controller with recurrent state and sequence-fragment DQN updates."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        *,
        config: ThreatDRQNConfig = ThreatDRQNConfig(),
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
        if config.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        if not (0.0 <= config.epsilon_end <= 1.0 and 0.0 <= config.epsilon_start <= 1.0):
            raise ValueError("epsilon values must be in [0, 1].")

        torch_module, nn_module, optim_module = _require_torch()
        self._torch = torch_module
        self._nn = nn_module
        self._optim = optim_module

        self.config = config
        self._seed = seed
        self._rng = random.Random(seed)
        self._override_order = tuple(ThreatOverride)
        self._override_to_index = {override: index for index, override in enumerate(self._override_order)}

        self._online = _ThreatQNetwork(
            nn_module=self._nn,
            feature_count=self.config.feature_count,
            hidden_size=self.config.hidden_size,
            action_count=len(self._override_order),
        )
        self._target = _ThreatQNetwork(
            nn_module=self._nn,
            feature_count=self.config.feature_count,
            hidden_size=self.config.hidden_size,
            action_count=len(self._override_order),
        )
        self._target.load_state_dict(self._online.state_dict())
        self._optimizer = self._optim.Adam(
            self._online.parameters(),
            lr=float(self.config.learning_rate),
        )
        self._loss_fn = self._nn.MSELoss()

        self._replay_episodes: list[list[ThreatTransition]] = []
        self._replay_transition_count = 0
        self._current_episode_replay: list[ThreatTransition] | None = None
        self._total_env_steps = 0
        self._optimization_steps = 0
        self._episodes_seen = 0
        self._hidden_state: Any | None = None
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
        self._hidden_state = None
        self._current_episode_replay = []
        self._replay_episodes.append(self._current_episode_replay)

    def select_override(
        self,
        *,
        features: Sequence[float],
        threat_active: bool,
        allowed_overrides: Sequence[ThreatOverride] | None = None,
        explore: bool = True,
    ) -> tuple[ThreatOverride, str, float | None]:
        if not threat_active:
            self.last_decision_reason = "threat_inactive_route_default"
            return (ThreatOverride.ROUTE_DEFAULT, self.last_decision_reason, None)

        vector = self._normalize_features(features)
        allowed = self._normalize_allowed_overrides(allowed_overrides)
        allowed_indices = tuple(self._override_to_index[override] for override in allowed)
        if not allowed_indices:
            raise ValueError("No allowed threat overrides supplied.")

        with self._torch.no_grad():
            feature_tensor = self._torch.tensor(vector, dtype=self._torch.float32).view(1, 1, -1)
            q_tensor, new_hidden = self._online.forward(
                features_tensor=feature_tensor,
                hidden_state=self._hidden_state,
            )
            self._hidden_state = new_hidden.detach()
            q_values = q_tensor[:, -1, :].squeeze(0)

        if explore and self._rng.random() < self.epsilon:
            selected = self._override_order[self._rng.choice(allowed_indices)]
            self.last_decision_reason = "threat_epsilon_explore"
            return (selected, self.last_decision_reason, None)

        best_index = max(allowed_indices, key=lambda index: float(q_values[index].item()))
        selected = self._override_order[best_index]
        q_value = float(q_values[best_index].item())
        self.last_decision_reason = "threat_greedy_q"
        return (selected, self.last_decision_reason, q_value)

    def observe(
        self,
        *,
        features: Sequence[float],
        chosen_override: ThreatOverride,
        reward: float,
        next_features: Sequence[float],
        done: bool,
        next_allowed_overrides: Sequence[ThreatOverride] | None,
    ) -> ThreatUpdateResult:
        if chosen_override not in self._override_to_index:
            raise ValueError(f"Unknown threat override: {chosen_override}")
        current_vector = self._normalize_features(features)
        next_vector = self._normalize_features(next_features)
        next_allowed = self._normalize_allowed_overrides(next_allowed_overrides)
        next_indices = tuple(self._override_to_index[item] for item in next_allowed)
        transition = ThreatTransition(
            features=current_vector,
            action_index=self._override_to_index[chosen_override],
            reward=float(reward),
            next_features=next_vector,
            done=bool(done),
            next_valid_action_indices=next_indices,
        )
        self._append_transition(transition)
        self._total_env_steps += 1

        minimum_replay = max(int(self.config.min_replay_size), int(self.config.batch_size))
        if self._replay_transition_count < minimum_replay:
            return ThreatUpdateResult(
                did_update=False,
                loss=None,
                epsilon=self.epsilon,
                replay_size=self._replay_transition_count,
                optimization_step=self._optimization_steps,
            )

        batch = [self._sample_sequence_fragment() for _ in range(int(self.config.batch_size))]
        loss = self._apply_update(batch=batch)
        self._optimization_steps += 1
        if self._optimization_steps % int(self.config.target_sync_interval) == 0:
            self._target.load_state_dict(self._online.state_dict())
        return ThreatUpdateResult(
            did_update=True,
            loss=loss,
            epsilon=self.epsilon,
            replay_size=self._replay_transition_count,
            optimization_step=self._optimization_steps,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "seed": self._seed,
            "config": asdict(self.config),
            "override_order": [override.value for override in self._override_order],
            "online_state_dict": self._online.state_dict(),
            "target_state_dict": self._target.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "training_state": {
                "total_env_steps": self._total_env_steps,
                "optimization_steps": self._optimization_steps,
                "episodes_seen": self._episodes_seen,
                "replay_transition_count": self._replay_transition_count,
            },
            "replay_episodes": [
                [
                    {
                        "features": list(item.features),
                        "action_index": int(item.action_index),
                        "reward": float(item.reward),
                        "next_features": list(item.next_features),
                        "done": bool(item.done),
                        "next_valid_action_indices": list(item.next_valid_action_indices),
                    }
                    for item in episode
                ]
                for episode in self._replay_episodes
                if episode
            ],
        }

    def load_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        version = int(payload.get("version", -1))
        if version not in {1, self.CHECKPOINT_VERSION}:
            raise ValueError(
                f"Unsupported threat checkpoint version {version}. Expected 1 or {self.CHECKPOINT_VERSION}."
            )
        self._online.load_state_dict(payload["online_state_dict"])
        self._target.load_state_dict(payload["target_state_dict"])
        self._optimizer.load_state_dict(payload["optimizer_state_dict"])

        training = payload.get("training_state", {})
        self._total_env_steps = int(training.get("total_env_steps", 0))
        self._optimization_steps = int(training.get("optimization_steps", 0))
        self._episodes_seen = int(training.get("episodes_seen", 0))
        self._hidden_state = None

        if version == 1:
            raw_replay = payload.get("replay", [])
            if not isinstance(raw_replay, list):
                raise ValueError("Threat checkpoint replay must be a list.")
            parsed_episodes = []
            for item in raw_replay:
                if not isinstance(item, dict):
                    continue
                parsed_episodes.append([self._parse_transition_payload(item)])
            self._replay_episodes = parsed_episodes
        else:
            raw_replay = payload.get("replay_episodes", [])
            if not isinstance(raw_replay, list):
                raise ValueError("Threat checkpoint replay_episodes must be a list.")
            self._replay_episodes = []
            for episode in raw_replay:
                if not isinstance(episode, list):
                    continue
                parsed_episode = [
                    self._parse_transition_payload(item)
                    for item in episode
                    if isinstance(item, dict)
                ]
                if parsed_episode:
                    self._replay_episodes.append(parsed_episode)

        self._replay_transition_count = sum(len(episode) for episode in self._replay_episodes)
        self._trim_replay_capacity()
        self._current_episode_replay = None

    def copy_weights_from(self, other: ThreatControllerDRQN) -> None:
        self._online.load_state_dict(other._online.state_dict())
        self._target.load_state_dict(other._target.state_dict())

    def training_snapshot(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "total_env_steps": self._total_env_steps,
            "optimization_steps": self._optimization_steps,
            "episodes_seen": self._episodes_seen,
            "replay_transition_count": self._replay_transition_count,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self.checkpoint_payload(), target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config_override: ThreatDRQNConfig | None = None,
    ) -> ThreatControllerDRQN:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Threat checkpoint not found: {checkpoint_path}")
        torch_module, _nn, _optim = _require_torch()
        payload = torch_module.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Threat checkpoint root must be an object.")
        config_payload = payload.get("config", {})
        if not isinstance(config_payload, dict):
            raise ValueError("Threat checkpoint config must be an object.")
        config = config_override or ThreatDRQNConfig(**config_payload)
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

    def _normalize_allowed_overrides(
        self,
        values: Sequence[ThreatOverride] | None,
    ) -> tuple[ThreatOverride, ...]:
        if values is None:
            return self._override_order
        normalized: list[ThreatOverride] = []
        seen: set[ThreatOverride] = set()
        for value in values:
            parsed = value if isinstance(value, ThreatOverride) else ThreatOverride(str(value))
            if parsed in seen:
                continue
            seen.add(parsed)
            normalized.append(parsed)
        return tuple(normalized)

    def _parse_transition_payload(self, item: dict[str, Any]) -> ThreatTransition:
        return ThreatTransition(
            features=self._normalize_features(item.get("features", [])),
            action_index=int(item.get("action_index", 0)),
            reward=float(item.get("reward", 0.0)),
            next_features=self._normalize_features(item.get("next_features", [])),
            done=bool(item.get("done", False)),
            next_valid_action_indices=tuple(
                int(value) for value in item.get("next_valid_action_indices", [])
            ),
        )

    def _append_transition(self, transition: ThreatTransition) -> None:
        if self._current_episode_replay is None:
            self._current_episode_replay = []
            self._replay_episodes.append(self._current_episode_replay)
        self._current_episode_replay.append(transition)
        self._replay_transition_count += 1
        self._trim_replay_capacity()

    def _trim_replay_capacity(self) -> None:
        capacity = int(self.config.replay_capacity)
        while self._replay_episodes and self._replay_transition_count > capacity:
            first_episode = self._replay_episodes[0]
            excess = self._replay_transition_count - capacity
            if len(first_episode) <= excess:
                self._replay_transition_count -= len(first_episode)
                self._replay_episodes.pop(0)
                continue
            del first_episode[:excess]
            self._replay_transition_count -= excess
        self._replay_episodes = [episode for episode in self._replay_episodes if episode]
        if self._current_episode_replay is not None and not self._current_episode_replay:
            self._current_episode_replay = None

    def _sample_sequence_fragment(self) -> list[ThreatTransition]:
        valid_episodes = [episode for episode in self._replay_episodes if episode]
        if not valid_episodes:
            raise ValueError("Cannot sample from empty threat replay.")
        episode = self._rng.choice(valid_episodes)
        end_index = self._rng.randrange(len(episode))
        start_index = max(0, end_index - int(self.config.sequence_length) + 1)
        return episode[start_index : end_index + 1]

    def _apply_update(self, *, batch: list[list[ThreatTransition]]) -> float:
        sequence_length = int(self.config.sequence_length)
        zero_vector = [0.0 for _ in range(int(self.config.feature_count))]
        state_sequences: list[list[list[float]]] = []
        action_values: list[int] = []
        reward_values: list[float] = []
        next_state_values: list[list[float]] = []
        done_values: list[bool] = []
        next_index_values: list[tuple[int, ...]] = []

        for fragment in batch:
            padded = [zero_vector[:] for _ in range(sequence_length)]
            for offset, transition in enumerate(fragment[-sequence_length:]):
                padded[sequence_length - len(fragment[-sequence_length:]) + offset] = list(transition.features)
            last_transition = fragment[-1]
            state_sequences.append(padded)
            action_values.append(last_transition.action_index)
            reward_values.append(last_transition.reward)
            next_state_values.append(list(last_transition.next_features))
            done_values.append(last_transition.done)
            next_index_values.append(last_transition.next_valid_action_indices)

        state_tensor = self._torch.tensor(state_sequences, dtype=self._torch.float32)
        action_tensor = self._torch.tensor(action_values, dtype=self._torch.int64)
        reward_tensor = self._torch.tensor(reward_values, dtype=self._torch.float32)
        next_state_tensor = self._torch.tensor(next_state_values, dtype=self._torch.float32).view(len(batch), 1, -1)
        done_tensor = self._torch.tensor(done_values, dtype=self._torch.float32)

        q_seq, _hidden = self._online.forward(features_tensor=state_tensor, hidden_state=None)
        selected_q = q_seq[:, -1, :].gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with self._torch.no_grad():
            next_q_seq, _next_hidden = self._target.forward(
                features_tensor=next_state_tensor,
                hidden_state=None,
            )
            next_last_q = next_q_seq[:, -1, :]
            max_next_values: list[float] = []
            for index, next_indices in enumerate(next_index_values):
                if done_values[index] or not next_indices:
                    max_next_values.append(0.0)
                    continue
                allowed = [float(next_last_q[index, offset].item()) for offset in next_indices]
                max_next_values.append(max(allowed) if allowed else 0.0)
            next_q_tensor = self._torch.tensor(max_next_values, dtype=self._torch.float32)
            targets = reward_tensor + ((1.0 - done_tensor) * float(self.config.gamma) * next_q_tensor)

        loss = self._loss_fn(selected_q, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._torch.nn.utils.clip_grad_norm_(
            self._online.parameters(),
            max_norm=max(1.0, math.sqrt(float(self.config.hidden_size))),
        )
        self._optimizer.step()
        return float(loss.item())

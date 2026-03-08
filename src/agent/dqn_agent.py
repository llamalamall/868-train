"""DQN-style learning agent with replay/target updates and checkpointing."""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.state.schema import GameStateSnapshot, GridPosition

_MOVE_VECTORS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
)


class DQNCheckpointError(RuntimeError):
    """Raised when a DQN checkpoint cannot be read or validated."""


@dataclass(frozen=True)
class DQNConfig:
    """Core hyperparameters for the first DQN implementation."""

    gamma: float = 0.99
    learning_rate: float = 0.005
    replay_capacity: int = 20_000
    min_replay_size: int = 256
    batch_size: int = 64
    target_sync_interval: int = 500
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    feature_count: int = 22
    feature_clip_abs: float = 100.0
    max_gradient_norm: float = 10.0


@dataclass(frozen=True)
class ReplayTransition:
    """One replay item used by minibatch Q-learning updates."""

    state_features: tuple[float, ...]
    action_index: int
    reward: float
    next_state_features: tuple[float, ...]
    done: bool
    next_available_action_indices: tuple[int, ...]


@dataclass(frozen=True)
class DQNUpdateResult:
    """Summary from one `observe()` call."""

    did_update: bool
    loss: float | None
    epsilon: float
    replay_size: int
    optimization_step: int


@dataclass(frozen=True)
class DQNTrainingState:
    """Progress counters persisted in checkpoints."""

    total_env_steps: int
    optimization_steps: int
    episodes_seen: int
    last_loss: float | None


class ReplayBuffer:
    """Fixed-size ring buffer for replay transitions."""

    def __init__(self, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1.")
        self._capacity = capacity
        self._cursor = 0
        self._items: list[ReplayTransition] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, transition: ReplayTransition) -> None:
        if len(self._items) < self._capacity:
            self._items.append(transition)
            return
        self._items[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self._capacity

    def sample(self, *, batch_size: int, rng: random.Random) -> tuple[ReplayTransition, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if batch_size > len(self._items):
            raise ValueError("batch_size cannot exceed current replay size.")
        return tuple(rng.sample(self._items, batch_size))

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self._capacity,
            "cursor": self._cursor,
            "items": [
                {
                    "state_features": list(item.state_features),
                    "action_index": item.action_index,
                    "reward": item.reward,
                    "next_state_features": list(item.next_state_features),
                    "done": item.done,
                    "next_available_action_indices": list(item.next_available_action_indices),
                }
                for item in self._items
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        expected_feature_count: int,
        expected_action_count: int,
    ) -> ReplayBuffer:
        capacity = int(payload.get("capacity", 0))
        buffer = cls(capacity=capacity)
        cursor = int(payload.get("cursor", 0))
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise DQNCheckpointError("Replay buffer 'items' must be a list.")

        parsed: list[ReplayTransition] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise DQNCheckpointError("Replay buffer item must be an object.")
            state_features = tuple(float(v) for v in raw_item.get("state_features", []))
            next_state_features = tuple(float(v) for v in raw_item.get("next_state_features", []))
            action_index = int(raw_item.get("action_index"))
            reward = float(raw_item.get("reward"))
            done = bool(raw_item.get("done"))
            next_indices = tuple(int(v) for v in raw_item.get("next_available_action_indices", []))

            if len(state_features) != expected_feature_count:
                raise DQNCheckpointError(
                    "Replay transition state_features size mismatch "
                    f"(expected {expected_feature_count}, got {len(state_features)})."
                )
            if len(next_state_features) != expected_feature_count:
                raise DQNCheckpointError(
                    "Replay transition next_state_features size mismatch "
                    f"(expected {expected_feature_count}, got {len(next_state_features)})."
                )
            if action_index < 0 or action_index >= expected_action_count:
                raise DQNCheckpointError(f"Replay transition action_index out of range: {action_index}.")
            if any(index < 0 or index >= expected_action_count for index in next_indices):
                raise DQNCheckpointError("Replay transition has invalid next action index.")

            parsed.append(
                ReplayTransition(
                    state_features=state_features,
                    action_index=action_index,
                    reward=reward,
                    next_state_features=next_state_features,
                    done=done,
                    next_available_action_indices=next_indices,
                )
            )

        if len(parsed) > capacity:
            raise DQNCheckpointError(
                f"Replay buffer item count {len(parsed)} exceeds capacity {capacity}."
            )
        if parsed and (cursor < 0 or cursor >= max(len(parsed), 1)):
            raise DQNCheckpointError(
                f"Replay cursor {cursor} is out of range for replay size {len(parsed)}."
            )

        buffer._items = parsed
        buffer._cursor = cursor if parsed else 0
        return buffer


class _LinearQModel:
    """Lightweight linear approximator for per-action Q-values."""

    def __init__(self, *, action_count: int, feature_count: int, rng: random.Random) -> None:
        if action_count < 1:
            raise ValueError("action_count must be >= 1.")
        if feature_count < 1:
            raise ValueError("feature_count must be >= 1.")
        scale = 1.0 / math.sqrt(float(feature_count))
        self.weights: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(feature_count)] for _ in range(action_count)
        ]
        self.bias: list[float] = [0.0 for _ in range(action_count)]

    @property
    def action_count(self) -> int:
        return len(self.weights)

    @property
    def feature_count(self) -> int:
        return len(self.weights[0]) if self.weights else 0

    def q_values(self, features: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            _dot_product(action_weights, features) + self.bias[action_index]
            for action_index, action_weights in enumerate(self.weights)
        )

    def copy_from(self, other: _LinearQModel) -> None:
        if self.action_count != other.action_count or self.feature_count != other.feature_count:
            raise ValueError("Cannot copy model parameters with mismatched shapes.")
        self.weights = [list(row) for row in other.weights]
        self.bias = list(other.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _LinearQModel:
        raw_weights = payload.get("weights")
        raw_bias = payload.get("bias")
        if not isinstance(raw_weights, list) or not raw_weights:
            raise DQNCheckpointError("Model checkpoint 'weights' must be a non-empty list.")
        if not isinstance(raw_bias, list) or not raw_bias:
            raise DQNCheckpointError("Model checkpoint 'bias' must be a non-empty list.")
        if len(raw_weights) != len(raw_bias):
            raise DQNCheckpointError(
                "Model checkpoint shape mismatch between weights and bias."
            )

        weights = [[float(value) for value in row] for row in raw_weights]
        feature_count = len(weights[0])
        if feature_count < 1:
            raise DQNCheckpointError("Model checkpoint feature_count must be >= 1.")
        if any(len(row) != feature_count for row in weights):
            raise DQNCheckpointError("Model checkpoint weights rows must have equal lengths.")

        model = cls(
            action_count=len(weights),
            feature_count=feature_count,
            rng=random.Random(0),
        )
        model.weights = weights
        model.bias = [float(value) for value in raw_bias]
        return model


class DQNAgent:
    """Simple DQN agent for compact vector state spaces."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        *,
        action_space: Sequence[str],
        config: DQNConfig = DQNConfig(),
        seed: int | None = None,
    ) -> None:
        action_names = tuple(str(action) for action in action_space)
        if not action_names:
            raise ValueError("action_space must include at least one action.")
        if len(set(action_names)) != len(action_names):
            raise ValueError("action_space must contain unique action names.")

        _validate_config(config)
        self.action_space = action_names
        self.config = config
        self._action_to_index = {name: index for index, name in enumerate(self.action_space)}
        self._rng = random.Random(seed)
        self._seed = seed

        self._online_model = _LinearQModel(
            action_count=len(self.action_space),
            feature_count=config.feature_count,
            rng=self._rng,
        )
        self._target_model = _LinearQModel(
            action_count=len(self.action_space),
            feature_count=config.feature_count,
            rng=self._rng,
        )
        self._target_model.copy_from(self._online_model)
        self._replay_buffer = ReplayBuffer(capacity=config.replay_capacity)

        self._total_env_steps = 0
        self._optimization_steps = 0
        self._episodes_seen = 0
        self._last_loss: float | None = None
        self._checkpoint_metadata: dict[str, Any] = {}
        self.last_decision_reason: str | None = None

    @property
    def training_state(self) -> DQNTrainingState:
        return DQNTrainingState(
            total_env_steps=self._total_env_steps,
            optimization_steps=self._optimization_steps,
            episodes_seen=self._episodes_seen,
            last_loss=self._last_loss,
        )

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self._checkpoint_metadata)

    @property
    def replay_size(self) -> int:
        return len(self._replay_buffer)

    @property
    def epsilon(self) -> float:
        return _linear_epsilon(
            start=self.config.epsilon_start,
            end=self.config.epsilon_end,
            decay_steps=self.config.epsilon_decay_steps,
            steps=self._total_env_steps,
        )

    def start_episode(self) -> None:
        """Record a new training episode."""
        self._episodes_seen += 1

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        available_actions: Sequence[str] | None = None,
        explore: bool = True,
    ) -> str:
        """Select one action using epsilon-greedy policy over valid actions."""
        valid_actions = self._resolve_actions(available_actions)
        if not valid_actions:
            raise ValueError("No valid actions available for selection.")

        if explore and self._rng.random() < self.epsilon:
            self.last_decision_reason = "epsilon_explore"
            return str(self._rng.choice(valid_actions))

        state_features = self.featurize_state(state)
        q_values = self._online_model.q_values(state_features)
        best_action = max(valid_actions, key=lambda action: q_values[self._action_to_index[action]])
        self.last_decision_reason = "greedy_q"
        return str(best_action)

    def observe(
        self,
        *,
        state: GameStateSnapshot,
        action: str,
        reward: float,
        next_state: GameStateSnapshot,
        done: bool,
        next_available_actions: Sequence[str] | None = None,
    ) -> DQNUpdateResult:
        """Store transition and run one replay update when enough samples exist."""
        if action not in self._action_to_index:
            raise ValueError(f"Unknown action '{action}'.")

        state_features = self.featurize_state(state)
        next_features = self.featurize_state(next_state)
        next_indices = self._resolve_action_indices(next_available_actions)
        transition = ReplayTransition(
            state_features=state_features,
            action_index=self._action_to_index[action],
            reward=float(reward),
            next_state_features=next_features,
            done=bool(done),
            next_available_action_indices=next_indices,
        )
        self._replay_buffer.add(transition)
        self._total_env_steps += 1

        minimum_replay = max(self.config.min_replay_size, self.config.batch_size)
        if len(self._replay_buffer) < minimum_replay:
            return DQNUpdateResult(
                did_update=False,
                loss=None,
                epsilon=self.epsilon,
                replay_size=len(self._replay_buffer),
                optimization_step=self._optimization_steps,
            )

        batch = self._replay_buffer.sample(batch_size=self.config.batch_size, rng=self._rng)
        loss = self._apply_minibatch_update(batch=batch)
        self._last_loss = loss
        self._optimization_steps += 1

        if self._optimization_steps % self.config.target_sync_interval == 0:
            self._target_model.copy_from(self._online_model)

        return DQNUpdateResult(
            did_update=True,
            loss=loss,
            epsilon=self.epsilon,
            replay_size=len(self._replay_buffer),
            optimization_step=self._optimization_steps,
        )

    def featurize_state(self, state: GameStateSnapshot) -> tuple[float, ...]:
        """Convert a raw game snapshot into compact numeric features."""
        features = state_to_feature_vector(state, clip_abs=self.config.feature_clip_abs)
        if len(features) != self.config.feature_count:
            raise ValueError(
                "state featurizer produced wrong size "
                f"(expected {self.config.feature_count}, got {len(features)})."
            )
        return features

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Persist model parameters, replay buffer, and training counters."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_checkpoint_payload(metadata=metadata)
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> DQNAgent:
        """Load an agent from a previously saved checkpoint file."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise DQNCheckpointError(f"Checkpoint not found: {checkpoint_path}.")
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DQNCheckpointError(f"Invalid JSON checkpoint: {checkpoint_path}.") from error

        if not isinstance(payload, dict):
            raise DQNCheckpointError("Checkpoint root must be a JSON object.")

        version = int(payload.get("version", -1))
        if version != cls.CHECKPOINT_VERSION:
            raise DQNCheckpointError(
                f"Unsupported checkpoint version {version} "
                f"(expected {cls.CHECKPOINT_VERSION})."
            )

        raw_action_space = payload.get("action_space")
        if not isinstance(raw_action_space, list) or not raw_action_space:
            raise DQNCheckpointError("Checkpoint action_space must be a non-empty list.")
        action_space = tuple(str(action) for action in raw_action_space)

        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise DQNCheckpointError("Checkpoint config must be an object.")
        config = DQNConfig(**raw_config)

        agent = cls(
            action_space=action_space,
            config=config,
            seed=payload.get("seed"),
        )

        raw_online = payload.get("online_model")
        raw_target = payload.get("target_model")
        if not isinstance(raw_online, dict) or not isinstance(raw_target, dict):
            raise DQNCheckpointError("Checkpoint must include online_model and target_model objects.")
        online_model = _LinearQModel.from_dict(raw_online)
        target_model = _LinearQModel.from_dict(raw_target)

        expected_shape = (len(action_space), config.feature_count)
        _assert_model_shape(model=online_model, expected_shape=expected_shape, name="online_model")
        _assert_model_shape(model=target_model, expected_shape=expected_shape, name="target_model")
        agent._online_model = online_model
        agent._target_model = target_model

        replay_payload = payload.get("replay_buffer", {})
        if not isinstance(replay_payload, dict):
            raise DQNCheckpointError("Checkpoint replay_buffer must be an object.")
        agent._replay_buffer = ReplayBuffer.from_dict(
            replay_payload,
            expected_feature_count=config.feature_count,
            expected_action_count=len(action_space),
        )

        raw_training_state = payload.get("training_state")
        if not isinstance(raw_training_state, dict):
            raise DQNCheckpointError("Checkpoint training_state must be an object.")
        agent._total_env_steps = int(raw_training_state.get("total_env_steps", 0))
        agent._optimization_steps = int(raw_training_state.get("optimization_steps", 0))
        agent._episodes_seen = int(raw_training_state.get("episodes_seen", 0))
        raw_last_loss = raw_training_state.get("last_loss")
        agent._last_loss = None if raw_last_loss is None else float(raw_last_loss)
        if (
            agent._total_env_steps < 0
            or agent._optimization_steps < 0
            or agent._episodes_seen < 0
        ):
            raise DQNCheckpointError("Checkpoint training counters must be non-negative.")

        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise DQNCheckpointError("Checkpoint metadata must be an object.")
        agent._checkpoint_metadata = dict(raw_metadata)
        return agent

    def _build_checkpoint_payload(self, *, metadata: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "seed": self._seed,
            "action_space": list(self.action_space),
            "config": asdict(self.config),
            "online_model": self._online_model.to_dict(),
            "target_model": self._target_model.to_dict(),
            "replay_buffer": self._replay_buffer.to_dict(),
            "training_state": {
                "total_env_steps": self._total_env_steps,
                "optimization_steps": self._optimization_steps,
                "episodes_seen": self._episodes_seen,
                "last_loss": self._last_loss,
            },
            "metadata": dict(metadata or {}),
        }

    def _resolve_actions(self, available_actions: Sequence[str] | None) -> tuple[str, ...]:
        if available_actions is None:
            return self.action_space
        return tuple(action for action in available_actions if action in self._action_to_index)

    def _resolve_action_indices(self, available_actions: Sequence[str] | None) -> tuple[int, ...]:
        actions = self._resolve_actions(available_actions)
        if not actions:
            actions = self.action_space
        return tuple(self._action_to_index[action] for action in actions)

    def _apply_minibatch_update(self, *, batch: tuple[ReplayTransition, ...]) -> float:
        action_count = len(self.action_space)
        feature_count = self.config.feature_count

        grad_weights = [[0.0 for _ in range(feature_count)] for _ in range(action_count)]
        grad_bias = [0.0 for _ in range(action_count)]
        total_loss = 0.0

        for transition in batch:
            q_values = self._online_model.q_values(transition.state_features)
            predicted_q = q_values[transition.action_index]

            if transition.done or not transition.next_available_action_indices:
                target_q = transition.reward
            else:
                next_q_values = self._target_model.q_values(transition.next_state_features)
                max_next_q = max(next_q_values[index] for index in transition.next_available_action_indices)
                target_q = transition.reward + (self.config.gamma * max_next_q)

            td_error = predicted_q - target_q
            total_loss += 0.5 * (td_error * td_error)
            grad_bias[transition.action_index] += td_error
            row = grad_weights[transition.action_index]
            for feature_index, feature_value in enumerate(transition.state_features):
                row[feature_index] += td_error * feature_value

        scale = 1.0 / float(len(batch))
        _clip_gradients(grad_weights=grad_weights, grad_bias=grad_bias, max_norm=self.config.max_gradient_norm)

        for action_index in range(action_count):
            bias_update = self.config.learning_rate * (grad_bias[action_index] * scale)
            self._online_model.bias[action_index] -= bias_update
            for feature_index in range(feature_count):
                weight_update = self.config.learning_rate * (
                    grad_weights[action_index][feature_index] * scale
                )
                self._online_model.weights[action_index][feature_index] -= weight_update

        return total_loss * scale


def state_to_feature_vector(
    state: GameStateSnapshot,
    *,
    clip_abs: float = 100.0,
) -> tuple[float, ...]:
    """Project a rich snapshot to a fixed-size compact vector for DQN."""
    map_known = state.map.status == "ok"
    width = int(state.map.width) if map_known else 1
    height = int(state.map.height) if map_known else 1
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)
    max_distance = max(width + height - 2, 1)

    player = state.map.player_position if map_known else None
    exit_position = state.map.exit_position if map_known else None
    has_player = 1.0 if player is not None else 0.0
    has_exit = 1.0 if exit_position is not None else 0.0

    player_x = float(player.x) / float(max_x) if player is not None else 0.0
    player_y = float(player.y) / float(max_y) if player is not None else 0.0
    exit_x = float(exit_position.x) / float(max_x) if exit_position is not None else 0.0
    exit_y = float(exit_position.y) / float(max_y) if exit_position is not None else 0.0

    exit_steps = _shortest_path_distance(state, target=exit_position) if exit_position is not None else None
    exit_distance = (
        float(exit_steps) / float(max_distance)
        if player is not None and exit_steps is not None
        else 1.0
    )

    siphon_positions = tuple(state.map.siphons) if map_known else ()
    enemy_positions = (
        tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds and enemy.type_id > 0)
        if map_known
        else ()
    )
    on_siphon_tile = 1.0 if player is not None and player in siphon_positions else 0.0
    on_exit_tile = 1.0 if player is not None and exit_position is not None and player == exit_position else 0.0

    nearest_siphon_steps = _nearest_path_distance_to_targets(state=state, targets=siphon_positions)
    nearest_siphon = (
        float(nearest_siphon_steps) / float(max_distance)
        if player is not None and nearest_siphon_steps is not None
        else 1.0
    )
    nearest_enemy_steps = _nearest_path_distance_to_targets(state=state, targets=enemy_positions)
    nearest_enemy = (
        float(nearest_enemy_steps) / float(max_distance)
        if player is not None and nearest_enemy_steps is not None
        else 1.0
    )

    if siphon_positions:
        phase_siphon, phase_enemy, phase_exit = (1.0, 0.0, 0.0)
    elif enemy_positions:
        phase_siphon, phase_enemy, phase_exit = (0.0, 1.0, 0.0)
    else:
        phase_siphon, phase_enemy, phase_exit = (0.0, 0.0, 1.0 if map_known else 0.0)

    mask = state.prog_slots_available_mask
    if mask is not None:
        usable_prog_slots = int(mask & 0x3FF).bit_count()
    elif state.inventory.status == "ok":
        usable_prog_slots = len(state.inventory.raw_prog_ids[:10])
    else:
        usable_prog_slots = 0

    features = (
        _scaled_numeric(state.health.value if state.health.status == "ok" else None, scale=10.0),
        _scaled_numeric(state.energy.value if state.energy.status == "ok" else None, scale=10.0),
        _scaled_numeric(state.currency.value if state.currency.status == "ok" else None, scale=25.0),
        1.0 if (state.fail_state.status == "ok" and bool(state.fail_state.value)) else 0.0,
        1.0 if map_known else 0.0,
        has_player,
        player_x,
        player_y,
        has_exit,
        exit_x,
        exit_y,
        float(exit_distance),
        float(len(siphon_positions)) / 6.0,
        float(len(enemy_positions)) / 8.0,
        float(nearest_siphon),
        float(nearest_enemy),
        on_siphon_tile,
        on_exit_tile,
        phase_siphon,
        phase_enemy,
        phase_exit,
        float(min(max(usable_prog_slots, 0), 10)) / 10.0,
    )
    return tuple(_clip_feature(value, clip_abs=clip_abs) for value in features)


def _validate_config(config: DQNConfig) -> None:
    if not (0.0 <= config.gamma <= 1.0):
        raise ValueError("gamma must be between 0 and 1.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.replay_capacity < 1:
        raise ValueError("replay_capacity must be >= 1.")
    if config.min_replay_size < 1:
        raise ValueError("min_replay_size must be >= 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.batch_size > config.replay_capacity:
        raise ValueError("batch_size cannot exceed replay_capacity.")
    if config.target_sync_interval < 1:
        raise ValueError("target_sync_interval must be >= 1.")
    if config.epsilon_decay_steps < 1:
        raise ValueError("epsilon_decay_steps must be >= 1.")
    if not (0.0 <= config.epsilon_end <= 1.0 and 0.0 <= config.epsilon_start <= 1.0):
        raise ValueError("epsilon_start and epsilon_end must be between 0 and 1.")
    if config.feature_count < 1:
        raise ValueError("feature_count must be >= 1.")
    if config.feature_clip_abs <= 0:
        raise ValueError("feature_clip_abs must be > 0.")
    if config.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be > 0.")


def _assert_model_shape(
    *,
    model: _LinearQModel,
    expected_shape: tuple[int, int],
    name: str,
) -> None:
    expected_actions, expected_features = expected_shape
    if model.action_count != expected_actions or model.feature_count != expected_features:
        raise DQNCheckpointError(
            f"Checkpoint {name} shape mismatch: expected ({expected_actions}, {expected_features}), "
            f"got ({model.action_count}, {model.feature_count})."
        )


def _dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _linear_epsilon(*, start: float, end: float, decay_steps: int, steps: int) -> float:
    if steps <= 0:
        return start
    ratio = min(float(steps) / float(decay_steps), 1.0)
    return start + ((end - start) * ratio)


def _scaled_numeric(value: object | None, *, scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    try:
        if value is None or isinstance(value, bool):
            return 0.0
        return float(value) / scale
    except (TypeError, ValueError):
        return 0.0


def _clip_feature(value: float, *, clip_abs: float) -> float:
    if value > clip_abs:
        return clip_abs
    if value < -clip_abs:
        return -clip_abs
    return value


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    positions = {cell.position for cell in state.map.cells if cell.is_wall}
    if positions:
        return positions
    return {wall.position for wall in state.map.walls}


def _shortest_path_distance(state: GameStateSnapshot, *, target: GridPosition | None) -> int | None:
    if state.map.status != "ok" or state.map.player_position is None or target is None:
        return None
    width = int(state.map.width)
    height = int(state.map.height)
    if width <= 0 or height <= 0:
        return None
    if not _is_in_bounds(target, width=width, height=height):
        return None

    start = state.map.player_position
    if start == target:
        return 0

    walls = _wall_positions(state)
    if target in walls:
        return None

    queue: deque[tuple[GridPosition, int]] = deque([(start, 0)])
    visited: set[GridPosition] = {start}
    while queue:
        current, distance = queue.popleft()
        for dx, dy in _MOVE_VECTORS:
            candidate = GridPosition(x=current.x + dx, y=current.y + dy)
            if not _is_in_bounds(candidate, width=width, height=height):
                continue
            if candidate in walls or candidate in visited:
                continue
            if candidate == target:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    return None


def _nearest_path_distance_to_targets(
    *,
    state: GameStateSnapshot,
    targets: tuple[GridPosition, ...],
) -> int | None:
    if not targets:
        return None
    best: int | None = None
    for target in targets:
        distance = _shortest_path_distance(state, target=target)
        if distance is None:
            continue
        if best is None or distance < best:
            best = distance
    return best


def _clip_gradients(
    *,
    grad_weights: list[list[float]],
    grad_bias: list[float],
    max_norm: float,
) -> None:
    total_norm_sq = 0.0
    for row in grad_weights:
        for value in row:
            total_norm_sq += value * value
    for value in grad_bias:
        total_norm_sq += value * value
    if total_norm_sq <= 0:
        return
    norm = math.sqrt(total_norm_sq)
    if norm <= max_norm:
        return
    scale = max_norm / norm
    for row in grad_weights:
        for index in range(len(row)):
            row[index] *= scale
    for index in range(len(grad_bias)):
        grad_bias[index] *= scale

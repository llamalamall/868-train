"""Gym-like environment wrapper around the live game client."""

from __future__ import annotations

import concurrent.futures
import ctypes
import ctypes.wintypes
import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from src.config.offsets import load_offset_registry
from src.controller.action_api import ActionAPI, ActionConfig
from src.controller.input_driver import InputDriver
from src.controller.window_attach import WindowAttachError, attach_window, focus_window
from src.env.enemy_spawn_suppression import EnemySpawnSuppressor
from src.env.game_tick_speedup import (
    BackgroundMotionDisablePatcher,
    GameTickSpeedupPatcher,
    IdleFrameDelayBypassPatcher,
    TileAnimationFreezePatcher,
)
from src.env.reset_manager import NoopResetManager, ResetStrategy, SequenceResetManager
from src.memory.process_attach import attach_process, close_attached_process
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult
from src.state.extractor import extract_state
from src.state.fail_detector import MemoryFailDetector
from src.state.schema import GameStateSnapshot, GridPosition, MapState

LOGGER = logging.getLogger(__name__)
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_MOVE_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}
_PROG_SLOT_ACTION_BY_INDEX: tuple[str, ...] = tuple(f"prog_slot_{index}" for index in range(1, 11))
_PROG_SLOT_INDEX_BY_ACTION: dict[str, int] = {
    action_name: index
    for index, action_name in enumerate(_PROG_SLOT_ACTION_BY_INDEX)
}


def _prog_slot_index_for_action(action_name: str) -> int | None:
    return _PROG_SLOT_INDEX_BY_ACTION.get(action_name)


def _owned_prog_slot_actions(snapshot: GameStateSnapshot | None) -> set[str]:
    if snapshot is None or snapshot.inventory.status != "ok":
        return set()

    available: set[str] = set()
    for slot_index, _prog_id in enumerate(snapshot.inventory.raw_prog_ids[:10]):
        available.add(_PROG_SLOT_ACTION_BY_INDEX[slot_index])
    return available


def _prog_slot_allowed_by_ui_mask(
    *,
    snapshot: GameStateSnapshot | None,
    action_name: str,
) -> bool | None:
    if snapshot is None:
        return None
    mask = snapshot.prog_slots_available_mask
    if mask is None:
        return None
    slot_index = _prog_slot_index_for_action(action_name)
    if slot_index is None:
        return None
    return bool(mask & (1 << slot_index))


def _state_numeric(field: Any) -> float | None:
    status = getattr(field, "status", None)
    value = getattr(field, "value", None)
    if status != "ok" or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_siphons(snapshot: GameStateSnapshot) -> int | None:
    if snapshot.map.status != "ok":
        return None
    return len(snapshot.map.siphons)


def _count_live_enemies(snapshot: GameStateSnapshot) -> int | None:
    if snapshot.map.status != "ok":
        return None
    return sum(1 for enemy in snapshot.map.enemies if enemy.in_bounds)


def _player_on_exit(snapshot: GameStateSnapshot) -> bool:
    if snapshot.map.status != "ok":
        return False
    player = snapshot.map.player_position
    exit_position = snapshot.map.exit_position
    return player is not None and exit_position is not None and player == exit_position


def _field_effect_signature(field: Any) -> tuple[str, Any | None]:
    status = str(getattr(field, "status", "missing"))
    if status != "ok":
        return (status, None)
    return ("ok", getattr(field, "value", None))


def _clamp_action_press_duration_to_game_tick(
    *,
    action_config: ActionConfig,
    game_tick_ms: int,
) -> ActionConfig:
    """Cap key press hold duration so it never exceeds the configured game tick."""
    tick_seconds = max(float(game_tick_ms), 1.0) / 1000.0
    current_press_seconds = max(float(action_config.timings.press_duration_seconds), 0.0)
    clamped_press_seconds = min(current_press_seconds, tick_seconds)
    if abs(clamped_press_seconds - current_press_seconds) <= 1e-9:
        return action_config
    return replace(
        action_config,
        timings=replace(
            action_config.timings,
            press_duration_seconds=clamped_press_seconds,
        ),
    )


def _extra_fields_effect_signature(snapshot: GameStateSnapshot) -> tuple[tuple[str, tuple[str, Any | None]], ...]:
    if not snapshot.extra_fields:
        return ()
    return tuple(
        sorted(
            (str(key), _field_effect_signature(field))
            for key, field in snapshot.extra_fields.items()
        )
    )


def _inventory_effect_signature(snapshot: GameStateSnapshot) -> tuple[Any, ...]:
    inventory = snapshot.inventory
    if inventory.status != "ok":
        return (inventory.status,)
    return (
        "ok",
        tuple(int(prog_id) for prog_id in inventory.raw_prog_ids),
        tuple((int(item.prog_id), int(item.count)) for item in inventory.collected_progs),
    )


def _map_effect_signature(snapshot: GameStateSnapshot) -> tuple[Any, ...]:
    map_state = snapshot.map
    if map_state.status != "ok":
        return (map_state.status,)
    return (
        "ok",
        int(map_state.width),
        int(map_state.height),
        map_state.player_position,
        map_state.exit_position,
        tuple(map_state.cells),
        tuple(map_state.siphons),
        tuple(map_state.walls),
        tuple(map_state.resource_cells),
        tuple(map_state.enemies),
    )


def _state_effect_signature(snapshot: GameStateSnapshot) -> tuple[Any, ...]:
    return (
        _field_effect_signature(snapshot.health),
        _field_effect_signature(snapshot.energy),
        _field_effect_signature(snapshot.currency),
        _field_effect_signature(snapshot.fail_state),
        _inventory_effect_signature(snapshot),
        _map_effect_signature(snapshot),
        snapshot.can_siphon_now,
        snapshot.prog_slots_available_mask,
        _extra_fields_effect_signature(snapshot),
    )


def _state_has_effect_observability(snapshot: GameStateSnapshot) -> bool:
    scalar_visible = any(
        _field_effect_signature(field)[0] == "ok"
        for field in (snapshot.health, snapshot.energy, snapshot.currency, snapshot.fail_state)
    )
    inventory_visible = snapshot.inventory.status == "ok"
    map_visible = snapshot.map.status == "ok"
    ui_visible = (
        snapshot.can_siphon_now is not None
        or snapshot.prog_slots_available_mask is not None
    )
    extra_visible = any(
        _field_effect_signature(field)[0] == "ok"
        for field in snapshot.extra_fields.values()
    )
    return scalar_visible or inventory_visible or map_visible or ui_visible or extra_visible


class GameEnvError(RuntimeError):
    """Base environment error."""


class EnvironmentClosedError(GameEnvError):
    """Raised when operating on a closed environment."""


class ResetTimeoutError(GameEnvError):
    """Raised when reset does not complete before watchdog timeout."""


class StepTimeoutError(GameEnvError):
    """Raised when step path exceeds watchdog timeout."""


class StateDesyncError(GameEnvError):
    """Raised when state snapshots look desynced/invalid after bounded retries."""


class ActionPerformer(Protocol):
    """Subset of action API required by the environment."""

    def perform_action(self, action_name: str) -> None:
        """Execute one named control action."""


class StateProvider(Protocol):
    """Callable state-provider contract used by environment."""

    def __call__(self) -> GameStateSnapshot:
        """Read and return current normalized state snapshot."""


class FailDetector(Protocol):
    """Fail-detector contract for environment step loop."""

    def check(self) -> Any:
        """Evaluate terminal state and return detector result."""


RewardFunction = Callable[[GameStateSnapshot, GameStateSnapshot, bool, dict[str, Any]], float]
RecoveryHook = Callable[[str, Exception | None], None]
BeforeActionHook = Callable[[str], None]


class TelemetryLogger(Protocol):
    """Duck-typed telemetry contract used by GameEnv."""

    def start_episode(self, *, episode_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a new episode start."""

    def log_step(
        self,
        *,
        episode_id: str,
        step_index: int,
        action: str,
        pre_state: GameStateSnapshot,
        post_state: GameStateSnapshot,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record one environment step."""

    def log_terminal(
        self,
        *,
        episode_id: str,
        step_index: int,
        reason: str,
        terminal_state: GameStateSnapshot,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record terminal event details."""

    def close(self) -> None:
        """Release logger resources."""


@dataclass(frozen=True)
class GameEnvConfig:
    """Runtime controls for step/reset synchronization and watchdog behavior."""

    step_timeout_seconds: float = 3.0
    reset_timeout_seconds: float = 15.0
    state_timeout_seconds: float = 2.0
    state_poll_interval_seconds: float = 0.10
    post_action_poll_delay_seconds: float = 0.2
    wait_for_action_processing: bool = False
    action_ack_timeout_seconds: float = 0.35
    action_ack_poll_interval_seconds: float = 0.05
    post_reset_grace_seconds: float = 0.10
    require_non_terminal_on_reset: bool = True
    prog_slot_backoff_steps: int = 3
    prog_slot_fallback_min_energy: int = 1
    action_retry_attempts: int = 2
    reset_retry_attempts: int = 2
    state_read_retry_attempts: int = 2
    runtime_recovery_attempts: int = 2
    runtime_recovery_delay_seconds: float = 0.5
    attach_retry_attempts: int = 5
    attach_retry_delay_seconds: float = 0.5
    window_attach_retry_attempts: int = 3
    window_attach_retry_delay_seconds: float = 0.3
    window_focus_retry_attempts: int = 3
    window_focus_retry_delay_seconds: float = 0.2


@dataclass(frozen=True)
class RandomPolicyEpisodeResult:
    """Episode result emitted by random-policy validation runs."""

    episode_id: str
    steps: int
    done: bool
    total_reward: float
    terminal_reason: str | None


class MODULEENTRY32W(ctypes.Structure):
    """ctypes mapping for MODULEENTRY32W."""

    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("GlblcntUsage", ctypes.wintypes.DWORD),
        ("ProccntUsage", ctypes.wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("modBaseSize", ctypes.wintypes.DWORD),
        ("hModule", ctypes.wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256),
        ("szExePath", ctypes.c_wchar * 260),
    ]


def _default_reward_fn(
    _previous_state: GameStateSnapshot,
    _current_state: GameStateSnapshot,
    _done: bool,
    _info: dict[str, Any],
) -> float:
    return 0.0


def _state_is_terminal(snapshot: GameStateSnapshot) -> bool:
    return snapshot.fail_state.status == "ok" and bool(snapshot.fail_state.value)


def _is_terminal_health_value(value: Any) -> bool:
    try:
        return int(float(value)) == -1
    except (TypeError, ValueError):
        return False


def _is_null_pointer_error(error_code: Any) -> bool:
    return isinstance(error_code, str) and error_code.strip().lower() == "null_pointer"


def _state_indicates_start_screen(snapshot: GameStateSnapshot) -> bool:
    health = snapshot.health
    return health.status != "ok" and _is_null_pointer_error(health.error_code)


def _resolve_default_action_space(action_api: ActionPerformer) -> tuple[str, ...]:
    config = getattr(action_api, "config", None)
    bindings = getattr(config, "action_key_bindings", None) if config is not None else None
    if isinstance(bindings, dict) and bindings:
        actions = tuple(bindings.keys())
    else:
        actions = ("move_up", "move_down", "move_left", "move_right", "confirm", "space")
    if "wait" not in actions:
        actions = (*actions, "wait")
    return actions


class GameEnv:
    """Gym-like environment wrapper with reset/step contract."""

    def __init__(
        self,
        *,
        action_api: ActionPerformer,
        state_provider: StateProvider,
        fail_detector: FailDetector | None = None,
        reset_strategy: ResetStrategy | None = None,
        reward_fn: RewardFunction = _default_reward_fn,
        config: GameEnvConfig = GameEnvConfig(),
        action_space: tuple[str, ...] | None = None,
        telemetry_logger: TelemetryLogger | None = None,
        close_telemetry_on_close: bool = False,
        recovery_hook: RecoveryHook | None = None,
        before_action_hook: BeforeActionHook | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._action_api = action_api
        self._state_provider = state_provider
        self._fail_detector = fail_detector
        self._reset_strategy = reset_strategy or NoopResetManager()
        self._reward_fn = reward_fn
        self._config = config
        self._action_space = action_space or _resolve_default_action_space(action_api)
        self._telemetry_logger = telemetry_logger
        self._close_telemetry_on_close = close_telemetry_on_close
        self._recovery_hook = recovery_hook
        self._before_action_hook = before_action_hook
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._logger = logger or LOGGER

        self._closed = False
        self._current_state: GameStateSnapshot | None = None
        self._current_episode_id: str | None = None
        self._episode_counter = 0
        self._step_index = 0
        self._prog_action_backoff: dict[str, int] = {}
        self._cleanup_callbacks: list[Callable[[], None]] = []

    @property
    def action_space(self) -> tuple[str, ...]:
        """All available discrete action names."""
        return self._action_space

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        """Return current action subset filtered for map-edge and wall collisions."""
        base_actions = tuple(action for action in self._action_space if action not in {"wait", "cancel"})
        snapshot = state if state is not None else self._current_state
        owned_prog_actions = _owned_prog_slot_actions(snapshot)
        fail_active = (
            snapshot is not None
            and snapshot.fail_state.status == "ok"
            and bool(snapshot.fail_state.value)
        )

        filtered_base: list[str] = []
        for action in base_actions:
            prog_slot_index = _prog_slot_index_for_action(action)
            if prog_slot_index is not None:
                if action not in owned_prog_actions:
                    continue
                if self._prog_action_backoff.get(action, 0) > 0:
                    continue
                allowed_by_ui_mask = _prog_slot_allowed_by_ui_mask(
                    snapshot=snapshot,
                    action_name=action,
                )
                if allowed_by_ui_mask is False:
                    continue
                if allowed_by_ui_mask is None and not self._fallback_prog_slot_allowed(snapshot):
                    continue

            if action == "confirm" and not fail_active:
                continue
            if action == "space" and not self._space_action_allowed(snapshot):
                continue
            filtered_base.append(action)

        base_actions = tuple(filtered_base)

        if snapshot is None or snapshot.map.status != "ok":
            return base_actions
        player_position = snapshot.map.player_position
        if player_position is None:
            return base_actions

        wall_positions = {cell.position for cell in snapshot.map.cells if cell.is_wall}
        if not wall_positions and snapshot.map.walls:
            wall_positions = {wall.position for wall in snapshot.map.walls}

        filtered: list[str] = []
        for action in base_actions:
            delta = _MOVE_ACTION_DELTAS.get(action)
            if delta is None:
                filtered.append(action)
                continue

            candidate = GridPosition(
                x=player_position.x + delta[0],
                y=player_position.y + delta[1],
            )
            if (
                candidate.x < 0
                or candidate.y < 0
                or candidate.x >= snapshot.map.width
                or candidate.y >= snapshot.map.height
            ):
                continue
            if candidate in wall_positions:
                continue
            filtered.append(action)

        return tuple(filtered)

    @property
    def current_episode_id(self) -> str | None:
        """Episode identifier for current active episode."""
        return self._current_episode_id

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register callback executed when environment closes."""
        self._cleanup_callbacks.append(callback)

    def close(self) -> None:
        """Close environment resources and registered handles."""
        if self._closed:
            return
        self._closed = True

        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive cleanup path
                self._logger.exception("GameEnv cleanup callback failed.")

        if self._close_telemetry_on_close and self._telemetry_logger is not None:
            self._telemetry_logger.close()

    def reset(self) -> GameStateSnapshot:
        """Reset environment and return initial episode state."""
        self._ensure_open()

        self._run_reset_strategy_with_retries()
        if self._config.post_reset_grace_seconds > 0:
            self._sleep_fn(self._config.post_reset_grace_seconds)

        state = self._poll_state_until_ready(
            timeout_seconds=self._config.reset_timeout_seconds,
            require_non_terminal=self._config.require_non_terminal_on_reset,
        )

        self._episode_counter += 1
        self._current_episode_id = f"episode-{self._episode_counter:05d}"
        self._step_index = 0
        self._current_state = state
        self._prog_action_backoff.clear()

        if self._telemetry_logger is not None:
            self._telemetry_logger.start_episode(
                episode_id=self._current_episode_id,
                metadata={"source": "game_env"},
            )

        return state

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, Any]]:
        """Apply one action and return `(state, reward, done, info)`."""
        self._ensure_open()
        if action not in self._action_space:
            raise GameEnvError(
                f"Unknown action '{action}'. Allowed actions: {', '.join(self._action_space)}."
            )

        reset_performed = False
        if self._current_state is None or self._current_episode_id is None:
            self.reset()
            reset_performed = True
        assert self._current_state is not None
        assert self._current_episode_id is not None
        previous_state = self._current_state
        step_index = self._step_index
        self._advance_prog_action_backoff()

        self._perform_action_with_retries(action=action, error_cls=StepTimeoutError)
        if self._config.post_action_poll_delay_seconds > 0:
            self._sleep_fn(self._config.post_action_poll_delay_seconds)

        (
            current_state,
            action_acknowledged,
            action_ack_reason,
            action_ack_checks,
        ) = self._read_state_after_action(
            action=action,
            previous_state=previous_state,
        )
        done = _state_is_terminal(current_state)
        terminal_reason = "state:fail_state" if done else None

        detector_info: dict[str, Any] = {}
        if self._fail_detector is not None:
            detector_result = self._run_with_timeout(
                self._fail_detector.check,
                timeout_seconds=self._config.state_timeout_seconds,
                error_cls=StepTimeoutError,
                operation_name="fail_detector.check",
            )
            detector_info = {
                "fail_detector_reason": getattr(detector_result, "reason", None),
                "fail_detector_source": getattr(detector_result, "source", None),
                "fail_detector_error": getattr(detector_result, "error", None),
            }
            if bool(getattr(detector_result, "is_terminal", False)):
                done = True
                terminal_reason = str(getattr(detector_result, "reason", "terminal"))
        if not done and (
            _state_indicates_start_screen(current_state)
            or _is_null_pointer_error(detector_info.get("fail_detector_error"))
        ):
            done = True
            terminal_reason = "state:start_screen"
            detector_info["start_screen_detected"] = True

        action_effective, invalid_action_reason = self._analyze_action_effectiveness(
            action=action,
            previous_state=previous_state,
            current_state=current_state,
        )
        if not action_acknowledged and action != "wait" and invalid_action_reason is None:
            action_effective = False
            invalid_action_reason = "action_not_acknowledged"
        siphons_remaining = _count_siphons(current_state)
        enemies_remaining = _count_live_enemies(current_state)
        premature_exit_attempt = (
            _player_on_exit(current_state)
            and (
                (siphons_remaining is not None and siphons_remaining > 0)
                or (enemies_remaining is not None and enemies_remaining > 0)
            )
        )
        if (
            invalid_action_reason is not None
            and action.startswith("prog_slot_")
            and self._config.prog_slot_backoff_steps > 0
        ):
            self._prog_action_backoff[action] = max(
                self._prog_action_backoff.get(action, 0),
                int(self._config.prog_slot_backoff_steps),
            )

        info: dict[str, Any] = {
            "episode_id": self._current_episode_id,
            "step_index": step_index,
            "action": action,
            "reset_performed": reset_performed,
            "terminal_reason": terminal_reason,
            "action_effective": action_effective,
            "invalid_action_reason": invalid_action_reason,
            "action_acknowledged": action_acknowledged,
            "action_ack_reason": action_ack_reason,
            "action_ack_checks": action_ack_checks,
            "premature_exit_attempt": premature_exit_attempt,
        }
        info.update(detector_info)

        reward_value = float(self._reward_fn(previous_state, current_state, done, info))

        if self._telemetry_logger is not None:
            self._telemetry_logger.log_step(
                episode_id=self._current_episode_id,
                step_index=step_index,
                action=action,
                pre_state=previous_state,
                post_state=current_state,
                reward=reward_value,
                done=done,
                info=info,
            )
            if done and terminal_reason is not None:
                self._telemetry_logger.log_terminal(
                    episode_id=self._current_episode_id,
                    step_index=step_index,
                    reason=terminal_reason,
                    terminal_state=current_state,
                    info=detector_info,
                )

        self._current_state = current_state
        self._step_index += 1
        return (current_state, reward_value, done, info)

    def _read_state_after_action(
        self,
        *,
        action: str,
        previous_state: GameStateSnapshot,
    ) -> tuple[GameStateSnapshot, bool, str, int]:
        current_state = self._read_state_once_for_step()
        polls = 1
        acknowledged, reason = self._is_action_acknowledged(
            action=action,
            previous_state=previous_state,
            current_state=current_state,
        )
        if acknowledged:
            return (current_state, True, reason, polls)

        if action == "wait" or not bool(self._config.wait_for_action_processing):
            return (current_state, True, "action_ack_disabled", polls)

        timeout_seconds = max(float(self._config.action_ack_timeout_seconds), 0.0)
        if timeout_seconds <= 0:
            return (current_state, False, "action_ack_timeout", polls)

        poll_interval = max(float(self._config.action_ack_poll_interval_seconds), 0.0)
        deadline = self._monotonic_fn() + timeout_seconds
        while self._monotonic_fn() <= deadline:
            if poll_interval > 0:
                self._sleep_fn(poll_interval)
            current_state = self._read_state_once_for_step()
            polls += 1
            acknowledged, reason = self._is_action_acknowledged(
                action=action,
                previous_state=previous_state,
                current_state=current_state,
            )
            if acknowledged:
                return (current_state, True, reason, polls)

        return (current_state, False, "action_ack_timeout", polls)

    def _is_action_acknowledged(
        self,
        *,
        action: str,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
    ) -> tuple[bool, str]:
        if action == "wait":
            return (True, "wait_action")
        if _state_is_terminal(current_state):
            return (True, "terminal_state")
        if self._transition_has_effect(previous_state=previous_state, current_state=current_state):
            return (True, "state_changed")
        return (False, "no_observed_effect")

    def _space_action_allowed(self, snapshot: GameStateSnapshot | None) -> bool:
        if snapshot is None:
            return False
        if snapshot.can_siphon_now is True:
            return True
        if snapshot.can_siphon_now is False:
            return False
        if snapshot.map.status != "ok" or snapshot.map.player_position is None:
            return False
        player = snapshot.map.player_position
        for siphon in snapshot.map.siphons:
            if abs(player.x - siphon.x) + abs(player.y - siphon.y) <= 1:
                return True
        return False

    def _fallback_prog_slot_allowed(self, snapshot: GameStateSnapshot | None) -> bool:
        if snapshot is None:
            return True
        energy_value = _state_numeric(snapshot.energy)
        if energy_value is None:
            return True
        return energy_value >= float(self._config.prog_slot_fallback_min_energy)

    def _advance_prog_action_backoff(self) -> None:
        for action_name in tuple(self._prog_action_backoff):
            remaining = self._prog_action_backoff[action_name] - 1
            if remaining <= 0:
                self._prog_action_backoff.pop(action_name, None)
            else:
                self._prog_action_backoff[action_name] = remaining

    def _analyze_action_effectiveness(
        self,
        *,
        action: str,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
    ) -> tuple[bool, str | None]:
        state_changed = self._transition_has_effect(
            previous_state=previous_state,
            current_state=current_state,
        )

        if action == "space":
            if state_changed:
                return (True, None)
            return (False, "space_no_effect")

        if action.startswith("prog_slot_"):
            if not state_changed:
                return (False, "prog_no_effect")
        return (True, None)

    def _transition_has_effect(
        self,
        *,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
    ) -> bool:
        if (
            not _state_has_effect_observability(previous_state)
            and not _state_has_effect_observability(current_state)
        ):
            return True
        return _state_effect_signature(previous_state) != _state_effect_signature(current_state)

    @staticmethod
    def _bounded_attempts(value: int) -> int:
        return max(int(value), 1)

    @staticmethod
    def _is_read_desync_code(code: Any) -> bool:
        if not isinstance(code, str):
            return False
        normalized = code.strip().lower()
        if not normalized or normalized == "null_pointer":
            return False
        if normalized.startswith(
            (
                "read_",
                "short_read",
                "invalid_address",
                "invalid_size",
                "vector_",
                "module_base_resolve_failed",
                "resolve_failed",
            )
        ):
            return True
        return normalized in {
            "pointer_chain_failed",
            "null_pointer_chain",
        }

    def _state_desync_markers(self, state: GameStateSnapshot) -> tuple[str, ...]:
        if _state_indicates_start_screen(state):
            return ()

        markers: list[str] = []
        for field_name in ("health", "energy", "currency", "fail_state"):
            field = getattr(state, field_name)
            if getattr(field, "status", None) != "invalid":
                continue
            error_code = getattr(field, "error_code", None)
            if self._is_read_desync_code(error_code):
                markers.append(f"{field_name}:{error_code}")

        for field_name in ("inventory", "map"):
            field = getattr(state, field_name)
            if getattr(field, "status", None) != "invalid":
                continue
            error_code = getattr(field, "error_code", None)
            if self._is_read_desync_code(error_code):
                markers.append(f"{field_name}:{error_code}")

        health_marker = any(marker.startswith("health:") for marker in markers)
        if health_marker or len(markers) >= 2:
            return tuple(markers)
        return ()

    def _raise_actionable_reliability_error(
        self,
        *,
        category: str,
        operation: str,
        attempts: int,
        last_error: Exception,
    ) -> None:
        raise GameEnvError(
            f"{category} operation={operation} attempts={attempts} "
            f"last_error={type(last_error).__name__}: {last_error}"
        ) from last_error

    def _attempt_runtime_recovery(
        self,
        *,
        reason: str,
        operation: str,
        attempt: int,
        prior_error: Exception | None,
    ) -> None:
        if self._recovery_hook is None:
            return
        self._logger.warning(
            "Runtime recovery requested reason=%s operation=%s attempt=%s prior_error=%s",
            reason,
            operation,
            attempt,
            repr(prior_error),
        )
        try:
            self._recovery_hook(reason, prior_error)
        except Exception as recovery_error:
            self._raise_actionable_reliability_error(
                category="runtime_recovery_failed",
                operation=operation,
                attempts=attempt,
                last_error=recovery_error,
            )

    def _perform_action_with_retries(
        self,
        *,
        action: str,
        error_cls: type[GameEnvError],
    ) -> None:
        max_attempts = self._bounded_attempts(self._config.action_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if self._before_action_hook is not None:
                    self._run_with_timeout(
                        lambda: self._before_action_hook(action),
                        timeout_seconds=self._config.step_timeout_seconds,
                        error_cls=error_cls,
                        operation_name=f"before_action:{action}",
                    )
                self._run_with_timeout(
                    lambda: self._action_api.perform_action(action),
                    timeout_seconds=self._config.step_timeout_seconds,
                    error_cls=error_cls,
                    operation_name=f"action:{action}",
                )
                return
            except Exception as error:
                last_error = error
                if attempt >= max_attempts:
                    break
                self._attempt_runtime_recovery(
                    reason=f"action_dispatch_failed:{action}",
                    operation=f"action:{action}",
                    attempt=attempt,
                    prior_error=error,
                )

        assert last_error is not None
        if isinstance(last_error, (ResetTimeoutError, StepTimeoutError)):
            raise last_error
        self._raise_actionable_reliability_error(
            category="action_dispatch_failed",
            operation=f"action:{action}",
            attempts=max_attempts,
            last_error=last_error,
        )

    def _run_reset_strategy_with_retries(self) -> None:
        max_attempts = self._bounded_attempts(self._config.reset_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                self._run_with_timeout(
                    self._reset_strategy.reset,
                    timeout_seconds=self._config.reset_timeout_seconds,
                    error_cls=ResetTimeoutError,
                    operation_name="reset_strategy.reset",
                )
                return
            except Exception as error:
                last_error = error
                if attempt >= max_attempts:
                    break
                self._attempt_runtime_recovery(
                    reason="reset_strategy_failed",
                    operation="reset_strategy.reset",
                    attempt=attempt,
                    prior_error=error,
                )

        assert last_error is not None
        if isinstance(last_error, ResetTimeoutError):
            raise last_error
        self._raise_actionable_reliability_error(
            category="reset_strategy_failed",
            operation="reset_strategy.reset",
            attempts=max_attempts,
            last_error=last_error,
        )

    def _read_state_with_retries(
        self,
        *,
        timeout_seconds: float,
        error_cls: type[GameEnvError],
        operation_name: str,
    ) -> GameStateSnapshot:
        max_attempts = self._bounded_attempts(self._config.state_read_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                state = self._run_with_timeout(
                    self._state_provider,
                    timeout_seconds=timeout_seconds,
                    error_cls=error_cls,
                    operation_name=operation_name,
                )
            except Exception as error:
                last_error = error
                if attempt >= max_attempts:
                    break
                self._attempt_runtime_recovery(
                    reason="state_provider_failed",
                    operation=operation_name,
                    attempt=attempt,
                    prior_error=error,
                )
                continue

            desync_markers = self._state_desync_markers(state)
            if not desync_markers:
                return state

            last_error = StateDesyncError(
                "Detected stale/invalid state snapshot markers: " + ", ".join(desync_markers)
            )
            if attempt >= max_attempts:
                break
            self._attempt_runtime_recovery(
                reason=f"state_desync:{'|'.join(desync_markers)}",
                operation=operation_name,
                attempt=attempt,
                prior_error=last_error,
            )

        assert last_error is not None
        if isinstance(last_error, (ResetTimeoutError, StepTimeoutError, StateDesyncError)):
            raise last_error
        self._raise_actionable_reliability_error(
            category="state_provider_failed",
            operation=operation_name,
            attempts=max_attempts,
            last_error=last_error,
        )

    def _read_state_once_for_step(self) -> GameStateSnapshot:
        return self._read_state_with_retries(
            timeout_seconds=self._config.state_timeout_seconds,
            operation_name="state_provider",
            error_cls=StepTimeoutError,
        )

    def _poll_state_until_ready(
        self,
        *,
        timeout_seconds: float,
        require_non_terminal: bool,
    ) -> GameStateSnapshot:
        deadline = self._monotonic_fn() + timeout_seconds
        while self._monotonic_fn() <= deadline:
            state = self._read_state_with_retries(
                timeout_seconds=self._config.state_timeout_seconds,
                operation_name="state_provider",
                error_cls=ResetTimeoutError,
            )
            if _state_indicates_start_screen(state):
                self._dispatch_reset_recovery_actions(reason="start_screen_null_pointer")
                self._sleep_fn(self._config.state_poll_interval_seconds)
                continue
            if _state_is_terminal(state):
                if not require_non_terminal:
                    return state
                self._dispatch_reset_recovery_actions(reason="terminal_fail_state")
                self._sleep_fn(self._config.state_poll_interval_seconds)
                continue
            return state

        raise ResetTimeoutError(
            "Reset timed out waiting for a non-terminal state."
            if require_non_terminal
            else "Reset timed out waiting for state."
        ) from None

    def _run_with_timeout(
        self,
        func: Callable[[], Any],
        *,
        timeout_seconds: float,
        error_cls: type[GameEnvError],
        operation_name: str,
    ) -> Any:
        if timeout_seconds <= 0:
            return func()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            raise error_cls(
                f"Watchdog timeout during {operation_name} after {timeout_seconds:.2f}s."
            ) from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _dispatch_reset_recovery_actions(self, *, reason: str) -> None:
        recovery_actions = tuple(
            action for action in ("confirm", "space") if action in self._action_space
        )
        if not recovery_actions:
            self._logger.warning(
                "Detected reset recovery condition (%s), but no confirm/space actions are configured.",
                reason,
            )
            return

        self._logger.info(
            "Detected reset recovery condition (%s); dispatching recovery actions: %s.",
            reason,
            ", ".join(recovery_actions),
        )
        for action_name in recovery_actions:
            self._perform_action_with_retries(action=action_name, error_cls=ResetTimeoutError)
            if self._config.post_action_poll_delay_seconds > 0:
                self._sleep_fn(self._config.post_action_poll_delay_seconds)

    def _ensure_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("GameEnv is closed.")

    @classmethod
    def from_live_process(
        cls,
        *,
        executable_name: str = "868-HACK.exe",
        offsets_config_path: Path | None = None,
        config: GameEnvConfig = GameEnvConfig(),
        telemetry_logger: TelemetryLogger | None = None,
        reset_sequence: tuple[str, ...] | None = ("confirm",),
        launch_process_if_missing: bool = True,
        focus_window_on_attach: bool = True,
        window_targeted_input: bool = False,
        action_config: ActionConfig | None = None,
        pre_reset_hook: Callable[[], None] | None = None,
        reward_fn: RewardFunction = _default_reward_fn,
        game_tick_ms: int = 16,
        no_enemies_mode: bool = False,
        disable_idle_frame_delay: bool = False,
        disable_background_motion: bool = False,
        disable_wall_animations: bool = False,
    ) -> GameEnv:
        """Create a live environment bound to running game process/window."""
        if int(game_tick_ms) < 1 or int(game_tick_ms) > 16:
            raise ValueError("game_tick_ms must be between 1 and 16.")
        registry = load_offset_registry(config_path=offsets_config_path)
        runtime_lock = threading.RLock()
        attach_retries = max(1, int(config.attach_retry_attempts))
        attach_retry_delay = float(config.attach_retry_delay_seconds)
        window_attach_retries = max(1, int(config.window_attach_retry_attempts))
        window_attach_retry_delay = float(config.window_attach_retry_delay_seconds)
        window_focus_retries = max(1, int(config.window_focus_retry_attempts))
        window_focus_retry_delay = float(config.window_focus_retry_delay_seconds)

        attached_process = attach_process(
            executable_name=executable_name,
            retries=attach_retries,
            retry_delay_seconds=attach_retry_delay,
            launch_if_missing=launch_process_if_missing,
        )
        try:
            attached_window = attach_window(
                pid=attached_process.pid,
                retries=window_attach_retries,
                retry_delay_seconds=window_attach_retry_delay,
            )
        except WindowAttachError:
            LOGGER.exception("Failed attaching window for pid=%s.", attached_process.pid)
            close_attached_process(attached_process)
            raise
        if focus_window_on_attach:
            try:
                focus_window(
                    attached_window,
                    retries=window_focus_retries,
                    retry_delay_seconds=window_focus_retry_delay,
                )
            except WindowAttachError:
                LOGGER.warning(
                    "Unable to focus window hwnd=%s pid=%s; continuing with window-targeted input.",
                    attached_window.hwnd,
                    attached_window.pid,
                )

        reader = ProcessMemoryReader(process_handle=attached_process.handle)
        runtime: dict[str, Any] = {
            "attached_process": attached_process,
            "attached_window": attached_window,
        }
        enemy_spawn_suppressor = EnemySpawnSuppressor(
            enabled=bool(no_enemies_mode),
            logger=LOGGER,
        )
        if enemy_spawn_suppressor.enabled:
            LOGGER.info("no_enemies_mode_enabled: active enemy slots will be suppressed.")
        no_enemy_map_root_address: int | None = None
        previous_map_state: MapState | None = None
        kernel32 = _get_kernel32()
        tick_speedup = GameTickSpeedupPatcher(
            game_tick_ms=int(game_tick_ms),
            logger=LOGGER,
        )
        idle_frame_delay_bypass = IdleFrameDelayBypassPatcher(
            enabled=bool(disable_idle_frame_delay),
            logger=LOGGER,
        )
        background_motion_disable = BackgroundMotionDisablePatcher(
            enabled=bool(disable_background_motion),
            logger=LOGGER,
        )
        tile_animation_freeze = TileAnimationFreezePatcher(
            enabled=bool(disable_wall_animations),
            logger=LOGGER,
        )
        runtime_patchers: tuple[Any, ...] = (
            tick_speedup,
            idle_frame_delay_bypass,
            background_motion_disable,
            tile_animation_freeze,
        )

        def _resolve_module_base_for_process(process: Any) -> int | None:
            module_name = str(getattr(process, "executable_name", executable_name)).strip()
            if not module_name:
                module_name = executable_name
            try:
                return _find_module_base(
                    pid=int(process.pid),
                    module_name=module_name,
                    kernel32=kernel32,
                )
            except Exception as error:  # pragma: no cover - runtime integration path
                LOGGER.warning(
                    "Unable to resolve module base for runtime patches pid=%s module=%s error=%s",
                    getattr(process, "pid", None),
                    module_name,
                    error,
                )
                return None

        def _apply_runtime_patches(process: Any) -> None:
            enabled_patchers = tuple(patcher for patcher in runtime_patchers if bool(patcher.enabled))
            if not enabled_patchers:
                return
            module_base = _resolve_module_base_for_process(process)
            if module_base is None:
                return
            for patcher in enabled_patchers:
                patcher.apply(
                    process_handle=int(process.handle),
                    module_base=int(module_base),
                )

        def _restore_runtime_patches(process: Any) -> None:
            enabled_patchers = tuple(patcher for patcher in runtime_patchers if bool(patcher.enabled))
            if not enabled_patchers:
                return
            module_base = _resolve_module_base_for_process(process)
            if module_base is None:
                return
            for patcher in enabled_patchers:
                patcher.restore(
                    process_handle=int(process.handle),
                    module_base=int(module_base),
                )

        _apply_runtime_patches(attached_process)

        def module_base_resolver(module_name: str) -> ReadResult[int]:
            try:
                with runtime_lock:
                    pid = int(runtime["attached_process"].pid)
                base = _find_module_base(
                    pid=pid,
                    module_name=module_name,
                    kernel32=kernel32,
                )
            except Exception as error:  # pragma: no cover - runtime integration path
                return ReadResult.fail(
                    ReadFailure(
                        code="module_base_resolve_failed",
                        message=f"Unable to resolve module base for '{module_name}'.",
                        detail=str(error),
                    )
                )
            return ReadResult.ok(base)

        def _extract_live_state() -> GameStateSnapshot:
            nonlocal previous_map_state
            snapshot = extract_state(
                reader=reader,
                registry=registry,
                module_base_resolver=module_base_resolver,
                previous_map_state=previous_map_state,
            )
            if snapshot.map.status == "ok":
                previous_map_state = snapshot.map
            else:
                previous_map_state = None
            return snapshot

        def _suppress_enemies_for_root(
            *,
            map_root_address: int,
            slots: tuple[int, ...] | None = None,
        ) -> None:
            if not enemy_spawn_suppressor.enabled:
                return
            with runtime_lock:
                active_process = runtime.get("attached_process")
            if active_process is None:
                return
            enemy_spawn_suppressor.suppress(
                process_handle=int(active_process.handle),
                map_root_address=int(map_root_address),
                slots=slots,
            )

        def state_provider() -> GameStateSnapshot:
            nonlocal no_enemy_map_root_address
            snapshot = _extract_live_state()
            if not enemy_spawn_suppressor.enabled:
                return snapshot

            map_state = snapshot.map
            if map_state.status != "ok" or map_state.address is None:
                return snapshot

            no_enemy_map_root_address = int(map_state.address)
            enemy_slots = tuple(int(enemy.slot) for enemy in map_state.enemies if int(enemy.slot) > 0)
            if not enemy_slots:
                return snapshot

            _suppress_enemies_for_root(
                map_root_address=no_enemy_map_root_address,
                slots=enemy_slots,
            )
            refreshed_snapshot = _extract_live_state()
            refreshed_map = refreshed_snapshot.map
            if refreshed_map.status == "ok" and refreshed_map.address is not None:
                no_enemy_map_root_address = int(refreshed_map.address)
            return refreshed_snapshot

        fail_entry = next(
            (entry for entry in registry.entries if entry.name in {"player_health", "health"}),
            None,
        )
        fail_detector = (
            MemoryFailDetector(
                reader=reader,
                fail_entry=fail_entry,
                module_base_resolver=module_base_resolver,
                is_terminal_value=_is_terminal_health_value,
            )
            if fail_entry is not None
            else None
        )

        action_api_holder: dict[str, ActionAPI | None] = {"value": None}

        def _close_process_handle_safe(process: Any) -> None:
            try:
                close_attached_process(process)
            except Exception:  # pragma: no cover - defensive cleanup path
                LOGGER.exception(
                    "Failed closing attached process handle pid=%s handle=%s.",
                    getattr(process, "pid", None),
                    getattr(process, "handle", None),
                )

        def _swap_runtime_handles(*, new_process: Any, new_window: Any) -> None:
            nonlocal no_enemy_map_root_address, previous_map_state
            with runtime_lock:
                previous_process = runtime["attached_process"]
                runtime["attached_process"] = new_process
                runtime["attached_window"] = new_window
                reader._process_handle = int(new_process.handle)  # type: ignore[attr-defined]
                if window_targeted_input and action_api_holder["value"] is not None:
                    action_api_holder["value"].target_hwnd = int(new_window.hwnd)
                no_enemy_map_root_address = None
                previous_map_state = None

            _apply_runtime_patches(new_process)
            if int(previous_process.handle) != int(new_process.handle):
                _close_process_handle_safe(previous_process)

        def _recover_live_bindings(reason: str, prior_error: Exception | None = None) -> None:
            max_attempts = max(1, int(config.runtime_recovery_attempts))
            retry_delay = float(config.runtime_recovery_delay_seconds)
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                new_process: Any | None = None
                try:
                    new_process = attach_process(
                        executable_name=executable_name,
                        retries=attach_retries,
                        retry_delay_seconds=attach_retry_delay,
                        launch_if_missing=launch_process_if_missing,
                    )
                    new_window = attach_window(
                        pid=int(new_process.pid),
                        retries=window_attach_retries,
                        retry_delay_seconds=window_attach_retry_delay,
                    )
                    if focus_window_on_attach:
                        try:
                            focus_window(
                                new_window,
                                retries=window_focus_retries,
                                retry_delay_seconds=window_focus_retry_delay,
                            )
                        except WindowAttachError:
                            LOGGER.warning(
                                "Unable to focus recovered window hwnd=%s pid=%s; proceeding.",
                                new_window.hwnd,
                                new_window.pid,
                            )
                    _swap_runtime_handles(new_process=new_process, new_window=new_window)
                    LOGGER.warning(
                        "Recovered live bindings reason=%s attempt=%s pid=%s hwnd=%s prior_error=%s",
                        reason,
                        attempt,
                        new_window.pid,
                        new_window.hwnd,
                        repr(prior_error),
                    )
                    return
                except Exception as recovery_error:
                    last_error = recovery_error
                    if new_process is not None:
                        _close_process_handle_safe(new_process)
                    LOGGER.warning(
                        "Recovery attempt %s/%s failed reason=%s error=%s",
                        attempt,
                        max_attempts,
                        reason,
                        recovery_error,
                    )
                    if attempt < max_attempts and retry_delay > 0:
                        time.sleep(retry_delay)

            if last_error is None:
                raise GameEnvError(
                    f"runtime_recovery_failed reason={reason} prior_error={prior_error!r}"
                )
            raise GameEnvError(
                f"runtime_recovery_failed reason={reason} attempts={max_attempts} "
                f"last_error={type(last_error).__name__}: {last_error}"
            ) from last_error

        def reacquire_window_handle() -> int:
            with runtime_lock:
                pid = int(runtime["attached_process"].pid)
            refreshed_window = attach_window(
                pid=pid,
                retries=window_attach_retries,
                retry_delay_seconds=window_attach_retry_delay,
            )
            if focus_window_on_attach:
                try:
                    focus_window(
                        refreshed_window,
                        retries=window_focus_retries,
                        retry_delay_seconds=window_focus_retry_delay,
                    )
                except WindowAttachError:
                    LOGGER.warning(
                        "Unable to focus refreshed window hwnd=%s pid=%s; proceeding.",
                        refreshed_window.hwnd,
                        refreshed_window.pid,
                    )
            with runtime_lock:
                runtime["attached_window"] = refreshed_window
            return refreshed_window.hwnd

        def before_action(action_name: str) -> None:
            if enemy_spawn_suppressor.enabled and no_enemy_map_root_address is not None:
                _suppress_enemies_for_root(map_root_address=no_enemy_map_root_address)
            if window_targeted_input or not focus_window_on_attach:
                return
            with runtime_lock:
                current_window = runtime["attached_window"]
            try:
                focus_window(
                    current_window,
                    retries=window_focus_retries,
                    retry_delay_seconds=window_focus_retry_delay,
                )
            except WindowAttachError as error:
                _recover_live_bindings(f"focus_lost_before_action:{action_name}", error)

        base_action_config = action_config or ActionConfig()
        resolved_action_config = _clamp_action_press_duration_to_game_tick(
            action_config=base_action_config,
            game_tick_ms=int(game_tick_ms),
        )
        if resolved_action_config.timings.press_duration_seconds < base_action_config.timings.press_duration_seconds:
            LOGGER.info(
                "action_press_duration_clamped from=%s to=%s tick_ms=%s",
                base_action_config.timings.press_duration_seconds,
                resolved_action_config.timings.press_duration_seconds,
                int(game_tick_ms),
            )

        action_api = ActionAPI(
            input_driver=InputDriver(),
            config=resolved_action_config,
            target_hwnd=attached_window.hwnd if window_targeted_input else None,
            window_reacquire_hook=reacquire_window_handle if window_targeted_input else None,
        )
        action_api_holder["value"] = action_api
        if reset_sequence:
            reset_strategy: ResetStrategy = SequenceResetManager(
                action_api=action_api,
                sequence=reset_sequence,
                before_sequence_hook=pre_reset_hook,
            )
        else:
            reset_strategy = NoopResetManager()
        env = cls(
            action_api=action_api,
            state_provider=state_provider,
            fail_detector=fail_detector,
            reset_strategy=reset_strategy,
            reward_fn=reward_fn,
            config=config,
            telemetry_logger=telemetry_logger,
            recovery_hook=_recover_live_bindings,
            before_action_hook=(
                before_action
                if (focus_window_on_attach or enemy_spawn_suppressor.enabled)
                else None
            ),
        )

        def _cleanup_live_bindings() -> None:
            with runtime_lock:
                active_process = runtime.get("attached_process")
                runtime["attached_process"] = None
            if active_process is not None:
                _close_process_handle_safe(active_process)

        env.add_cleanup_callback(_cleanup_live_bindings)

        def _cleanup_runtime_patches() -> None:
            with runtime_lock:
                active_process = runtime.get("attached_process")
            if active_process is None:
                return
            _restore_runtime_patches(active_process)

        env.add_cleanup_callback(_cleanup_runtime_patches)
        return env


def run_random_policy(
    *,
    env: GameEnv,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
    actions: tuple[str, ...] | None = None,
    before_step_callback: Callable[[dict[str, Any]], None] | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RandomPolicyEpisodeResult, ...]:
    """Run random actions for N episodes to validate env contract."""
    import random

    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be >= 1.")
    if actions is None:
        policy_actions = env.action_space
    else:
        if not actions:
            raise ValueError("actions must include at least one action.")
        unknown_actions = tuple(action for action in actions if action not in env.action_space)
        if unknown_actions:
            raise ValueError(
                "actions contains unknown action names: "
                + ", ".join(sorted(set(unknown_actions)))
            )
        policy_actions = actions

    rng = random.Random(seed)
    results: list[RandomPolicyEpisodeResult] = []

    for _ in range(episodes):
        env.reset()
        assert env.current_episode_id is not None
        total_reward = 0.0
        steps = 0
        done = False
        terminal_reason: str | None = None

        while steps < max_steps_per_episode and not done:
            step_index = steps
            if actions is None:
                step_actions = env.available_actions()
            else:
                step_actions = tuple(action for action in env.available_actions() if action in policy_actions)
            if not step_actions:
                raise GameEnvError("No available actions after map-edge/wall filtering.")

            action = rng.choice(step_actions)
            if before_step_callback is not None:
                before_step_callback(
                    {
                        "episode_id": env.current_episode_id,
                        "step_index": step_index,
                        "action": action,
                        "action_reason": "random_policy_sample",
                        "total_reward": total_reward,
                    }
                )
            _, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            terminal_reason_value = info.get("terminal_reason")
            if isinstance(terminal_reason_value, str):
                terminal_reason = terminal_reason_value

            if step_callback is not None:
                step_callback(
                    {
                        "episode_id": env.current_episode_id,
                        "step_index": step_index,
                        "action": action,
                        "action_reason": "random_policy_sample",
                        "reward": float(reward),
                        "total_reward": total_reward,
                        "done": done,
                        "terminal_reason": terminal_reason,
                        "reward_breakdown": info.get("reward_breakdown"),
                    }
                )

        results.append(
            RandomPolicyEpisodeResult(
                episode_id=env.current_episode_id,
                steps=steps,
                done=done,
                total_reward=total_reward,
                terminal_reason=terminal_reason,
            )
        )

    return tuple(results)


def _get_kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise GameEnvError("Live game environment is only supported on Windows.")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _find_module_base(pid: int, module_name: str, kernel32: ctypes.WinDLL) -> int:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snapshot == INVALID_HANDLE_VALUE:
        error_code = ctypes.get_last_error()
        raise GameEnvError(
            f"CreateToolhelp32Snapshot(module) failed for pid={pid} (error={error_code})."
        )

    module_entry = MODULEENTRY32W()
    module_entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
    wanted = module_name.lower()

    try:
        has_module = bool(kernel32.Module32FirstW(snapshot, ctypes.byref(module_entry)))
        while has_module:
            current_name = str(module_entry.szModule).lower()
            if current_name == wanted:
                base = ctypes.cast(module_entry.modBaseAddr, ctypes.c_void_p).value
                if base is None:
                    raise GameEnvError(f"Module '{module_name}' found, but base address is null.")
                return int(base)
            has_module = bool(kernel32.Module32NextW(snapshot, ctypes.byref(module_entry)))
    finally:
        kernel32.CloseHandle(snapshot)

    raise GameEnvError(f"Module '{module_name}' not found in pid={pid}.")

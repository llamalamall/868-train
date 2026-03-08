"""Gym-like environment wrapper around the live game client."""

from __future__ import annotations

import concurrent.futures
import ctypes
import ctypes.wintypes
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from src.config.offsets import load_offset_registry
from src.controller.action_api import ActionAPI, ActionConfig
from src.controller.input_driver import InputDriver
from src.controller.window_attach import WindowAttachError, attach_window, focus_window
from src.env.reset_manager import NoopResetManager, ResetStrategy, SequenceResetManager
from src.memory.process_attach import attach_process, close_attached_process
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult
from src.state.extractor import extract_state
from src.state.fail_detector import MemoryFailDetector
from src.state.schema import GameStateSnapshot, GridPosition

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
    return sum(1 for enemy in snapshot.map.enemies if enemy.in_bounds and enemy.type_id > 0)


def _player_on_exit(snapshot: GameStateSnapshot) -> bool:
    if snapshot.map.status != "ok":
        return False
    player = snapshot.map.player_position
    exit_position = snapshot.map.exit_position
    return player is not None and exit_position is not None and player == exit_position


class GameEnvError(RuntimeError):
    """Base environment error."""


class EnvironmentClosedError(GameEnvError):
    """Raised when operating on a closed environment."""


class ResetTimeoutError(GameEnvError):
    """Raised when reset does not complete before watchdog timeout."""


class StepTimeoutError(GameEnvError):
    """Raised when step path exceeds watchdog timeout."""


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
    post_action_poll_delay_seconds: float = 0.05
    post_reset_grace_seconds: float = 0.10
    require_non_terminal_on_reset: bool = True
    prog_slot_backoff_steps: int = 3
    prog_slot_fallback_min_energy: int = 1


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

        self._run_with_timeout(
            self._reset_strategy.reset,
            timeout_seconds=self._config.reset_timeout_seconds,
            error_cls=ResetTimeoutError,
            operation_name="reset_strategy.reset",
        )
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

        self._run_with_timeout(
            lambda: self._action_api.perform_action(action),
            timeout_seconds=self._config.step_timeout_seconds,
            error_cls=StepTimeoutError,
            operation_name=f"action:{action}",
        )
        if self._config.post_action_poll_delay_seconds > 0:
            self._sleep_fn(self._config.post_action_poll_delay_seconds)

        current_state = self._read_state_once_for_step()
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

    def _space_action_allowed(self, snapshot: GameStateSnapshot | None) -> bool:
        if snapshot is None:
            return False
        if snapshot.can_siphon_now is True:
            return True
        if snapshot.can_siphon_now is False:
            return False
        if snapshot.map.status != "ok" or snapshot.map.player_position is None:
            return False
        return snapshot.map.player_position in set(snapshot.map.siphons)

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
        if action == "space":
            if self._transition_has_effect(previous_state=previous_state, current_state=current_state):
                return (True, None)
            return (False, "space_no_effect")

        if action.startswith("prog_slot_"):
            energy_before = _state_numeric(previous_state.energy)
            energy_after = _state_numeric(current_state.energy)
            energy_spent: bool | None = None
            if energy_before is not None and energy_after is not None:
                energy_spent = energy_after < energy_before

            state_changed = self._transition_has_effect(
                previous_state=previous_state,
                current_state=current_state,
            )
            if energy_spent is None and not state_changed:
                return (False, "prog_no_effect")
            if energy_spent is False and not state_changed:
                return (False, "prog_no_effect")
        return (True, None)

    def _transition_has_effect(
        self,
        *,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
    ) -> bool:
        numeric_changes: list[bool | None] = []
        for before_field, after_field in (
            (previous_state.energy, current_state.energy),
            (previous_state.currency, current_state.currency),
            (previous_state.health, current_state.health),
        ):
            before_value = _state_numeric(before_field)
            after_value = _state_numeric(after_field)
            if before_value is None or after_value is None:
                numeric_changes.append(None)
                continue
            numeric_changes.append(after_value != before_value)

        prev_siphons = _count_siphons(previous_state)
        curr_siphons = _count_siphons(current_state)
        siphon_change = None if prev_siphons is None or curr_siphons is None else (prev_siphons != curr_siphons)

        prev_enemies = _count_live_enemies(previous_state)
        curr_enemies = _count_live_enemies(current_state)
        enemy_change = None if prev_enemies is None or curr_enemies is None else (prev_enemies != curr_enemies)

        player_change: bool | None = None
        if previous_state.map.status == "ok" and current_state.map.status == "ok":
            player_change = previous_state.map.player_position != current_state.map.player_position

        inventory_change: bool | None = None
        if previous_state.inventory.status == "ok" and current_state.inventory.status == "ok":
            inventory_change = previous_state.inventory.raw_prog_ids != current_state.inventory.raw_prog_ids

        signals = (
            *numeric_changes,
            siphon_change,
            enemy_change,
            player_change,
            inventory_change,
        )
        if any(signal is True for signal in signals):
            return True
        return not any(signal is not None for signal in signals)

    def _read_state_once_for_step(self) -> GameStateSnapshot:
        return self._run_with_timeout(
            self._state_provider,
            timeout_seconds=self._config.state_timeout_seconds,
            error_cls=StepTimeoutError,
            operation_name="state_provider",
        )

    def _poll_state_until_ready(
        self,
        *,
        timeout_seconds: float,
        require_non_terminal: bool,
    ) -> GameStateSnapshot:
        deadline = self._monotonic_fn() + timeout_seconds
        while self._monotonic_fn() <= deadline:
            state = self._run_with_timeout(
                self._state_provider,
                timeout_seconds=self._config.state_timeout_seconds,
                error_cls=ResetTimeoutError,
                operation_name="state_provider",
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as error:
                raise error_cls(
                    f"Watchdog timeout during {operation_name} after {timeout_seconds:.2f}s."
                ) from error

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
            self._run_with_timeout(
                lambda action=action_name: self._action_api.perform_action(action),
                timeout_seconds=self._config.step_timeout_seconds,
                error_cls=ResetTimeoutError,
                operation_name=f"start_screen_recovery:{action_name}",
            )
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
        reward_fn: RewardFunction = _default_reward_fn,
    ) -> GameEnv:
        """Create a live environment bound to running game process/window."""
        registry = load_offset_registry(config_path=offsets_config_path)
        attached_process = attach_process(
            executable_name=executable_name,
            retries=5,
            retry_delay_seconds=0.5,
            launch_if_missing=launch_process_if_missing,
        )
        try:
            attached_window = attach_window(
                pid=attached_process.pid,
                retries=3,
                retry_delay_seconds=0.3,
            )
        except WindowAttachError:
            LOGGER.exception("Failed attaching window for pid=%s.", attached_process.pid)
            close_attached_process(attached_process)
            raise
        if focus_window_on_attach:
            try:
                focus_window(
                    attached_window,
                    retries=3,
                    retry_delay_seconds=0.2,
                )
            except WindowAttachError:
                LOGGER.warning(
                    "Unable to focus window hwnd=%s pid=%s; continuing with window-targeted input.",
                    attached_window.hwnd,
                    attached_window.pid,
                )

        reader = ProcessMemoryReader(process_handle=attached_process.handle)
        kernel32 = _get_kernel32()

        def module_base_resolver(module_name: str) -> ReadResult[int]:
            try:
                base = _find_module_base(
                    pid=attached_process.pid,
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

        def state_provider() -> GameStateSnapshot:
            return extract_state(
                reader=reader,
                registry=registry,
                module_base_resolver=module_base_resolver,
            )

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

        def reacquire_window_handle() -> int:
            refreshed_window = attach_window(
                pid=attached_process.pid,
                retries=3,
                retry_delay_seconds=0.3,
            )
            if focus_window_on_attach:
                try:
                    focus_window(
                        refreshed_window,
                        retries=3,
                        retry_delay_seconds=0.2,
                    )
                except WindowAttachError:
                    LOGGER.warning(
                        "Unable to focus refreshed window hwnd=%s pid=%s; proceeding.",
                        refreshed_window.hwnd,
                        refreshed_window.pid,
                    )
            return refreshed_window.hwnd

        action_api = ActionAPI(
            input_driver=InputDriver(),
            config=action_config or ActionConfig(),
            target_hwnd=attached_window.hwnd if window_targeted_input else None,
            window_reacquire_hook=reacquire_window_handle if window_targeted_input else None,
        )
        if reset_sequence:
            reset_strategy: ResetStrategy = SequenceResetManager(
                action_api=action_api,
                sequence=reset_sequence,
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
        )
        env.add_cleanup_callback(lambda: close_attached_process(attached_process))
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

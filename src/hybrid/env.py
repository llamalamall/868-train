"""Hybrid live environment wrapper around low-level runtime integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.controller.action_api import ActionConfig
from src.env.game_env import GameEnv, GameEnvConfig
from src.state.schema import GameStateSnapshot, GridPosition

_WASD_KEY_CODES = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
}

_NUMPAD_KEY_CODES = {
    "NUMPAD2": 0x62,
    "NUMPAD4": 0x64,
    "NUMPAD6": 0x66,
    "NUMPAD8": 0x68,
}

_PROG_SLOT_ACTION_BINDINGS = {
    "prog_slot_1": "1",
    "prog_slot_2": "2",
    "prog_slot_3": "3",
    "prog_slot_4": "4",
    "prog_slot_5": "5",
    "prog_slot_6": "6",
    "prog_slot_7": "7",
    "prog_slot_8": "8",
    "prog_slot_9": "9",
    "prog_slot_10": "0",
}
_SIPHON_REACH_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
)


def _zero_reward(
    _previous_state: Any,
    _current_state: Any,
    _done: bool,
    _info: dict[str, Any],
) -> float:
    return 0.0


def _siphon_candidates_near_player(player_position: GridPosition) -> set[GridPosition]:
    return {
        GridPosition(x=player_position.x + dx, y=player_position.y + dy)
        for dx, dy in _SIPHON_REACH_DELTAS
    }


def _hybrid_resource_siphon_available(snapshot: GameStateSnapshot | None) -> bool:
    if snapshot is None or snapshot.map.status != "ok" or snapshot.map.player_position is None:
        return False
    player_x = int(snapshot.map.player_position.x)
    player_y = int(snapshot.map.player_position.y)
    if player_x < 0 or player_y < 0 or player_x >= int(snapshot.map.width) or player_y >= int(snapshot.map.height):
        return False

    layers = snapshot.map.layers
    expected_height = int(snapshot.map.height)
    expected_width = int(snapshot.map.width)
    layer_ready = (
        len(layers.energy_map) == expected_height
        and len(layers.credits_map) == expected_height
        and len(layers.points_map) == expected_height
        and len(layers.siphon_penalty_map) == expected_height
        and len(layers.progs_map) == expected_height
        and all(len(row) == expected_width for row in layers.energy_map)
        and all(len(row) == expected_width for row in layers.credits_map)
        and all(len(row) == expected_width for row in layers.points_map)
        and all(len(row) == expected_width for row in layers.siphon_penalty_map)
        and all(len(row) == expected_width for row in layers.progs_map)
    )
    if layer_ready:
        energy_gain = int(layers.energy_map[player_y][player_x])
        credit_gain = int(layers.credits_map[player_y][player_x])
        points_gain = int(layers.points_map[player_y][player_x])
        penalty_gain = int(layers.siphon_penalty_map[player_y][player_x])
        prog_ids = layers.progs_map[player_y][player_x]
        if energy_gain > 0 or credit_gain > 0 or points_gain > 0 or penalty_gain > 0 or bool(prog_ids):
            return True

    # Fallback for partially initialized map-layer snapshots.
    nearby = _siphon_candidates_near_player(snapshot.map.player_position)
    for cell in snapshot.map.cells:
        if cell.position not in nearby:
            continue
        if cell.is_wall:
            if int(cell.points) > 0 or int(cell.threat) > 0:
                return True
            if cell.prog_id is not None and int(cell.prog_id) > 0:
                return True
            continue
        if int(cell.credits) > 0 or int(cell.energy) > 0:
            return True
    return False


def _build_action_config(
    movement_keys: str,
    *,
    include_prog_actions: bool,
    siphon_key: str,
) -> ActionConfig:
    default_config = ActionConfig()
    bindings = dict(default_config.action_key_bindings)
    key_codes = dict(default_config.key_codes)

    normalized_siphon_key = str(siphon_key).strip().lower()
    if normalized_siphon_key not in {"space", "z"}:
        raise ValueError("siphon_key must be one of: space, z.")
    bindings["space"] = "SPACE" if normalized_siphon_key == "space" else "Z"

    if movement_keys == "wasd":
        bindings.update(
            {
                "move_up": "W",
                "move_down": "S",
                "move_left": "A",
                "move_right": "D",
            }
        )
        key_codes.update(_WASD_KEY_CODES)
    elif movement_keys == "numpad":
        bindings.update(
            {
                "move_up": "NUMPAD8",
                "move_down": "NUMPAD2",
                "move_left": "NUMPAD4",
                "move_right": "NUMPAD6",
            }
        )
        key_codes.update(_NUMPAD_KEY_CODES)
    elif movement_keys != "arrows":
        raise ValueError("movement_keys must be one of: arrows, wasd, numpad.")

    if include_prog_actions:
        bindings.update(_PROG_SLOT_ACTION_BINDINGS)
    else:
        bindings = {
            action_name: key_name
            for action_name, key_name in bindings.items()
            if not action_name.startswith("prog_slot_")
        }

    return ActionConfig(
        action_key_bindings=bindings,
        key_codes=key_codes,
        timings=default_config.timings,
    )


@dataclass(frozen=True)
class HybridLiveEnvConfig:
    """Config options for building the hybrid live environment."""

    step_timeout_seconds: float = 3.0
    reset_timeout_seconds: float = 15.0
    post_action_delay_seconds: float = 0.01
    wait_for_action_processing: bool = True
    action_ack_timeout_seconds: float = 0.35
    action_ack_poll_interval_seconds: float = 0.05
    prog_slot_backoff_steps: int = 3
    require_non_terminal_on_reset: bool = True
    game_tick_ms: int = 1
    disable_idle_frame_delay: bool = True
    disable_background_motion: bool = True
    disable_wall_animations: bool = True

    def to_game_env_config(self) -> GameEnvConfig:
        return GameEnvConfig(
            step_timeout_seconds=float(self.step_timeout_seconds),
            reset_timeout_seconds=float(self.reset_timeout_seconds),
            post_action_poll_delay_seconds=max(float(self.post_action_delay_seconds), 0.0),
            wait_for_action_processing=bool(self.wait_for_action_processing),
            action_ack_timeout_seconds=max(float(self.action_ack_timeout_seconds), 0.0),
            action_ack_poll_interval_seconds=max(float(self.action_ack_poll_interval_seconds), 0.0),
            prog_slot_backoff_steps=max(int(self.prog_slot_backoff_steps), 0),
            require_non_terminal_on_reset=bool(self.require_non_terminal_on_reset),
        )


class HybridLiveEnv:
    """Hybrid environment wrapper implementing episode reset/step contract."""

    def __init__(self, *, game_env: GameEnv, no_enemies_mode: bool) -> None:
        self._game_env = game_env
        self.no_enemies_mode = bool(no_enemies_mode)

    @property
    def action_space(self) -> tuple[str, ...]:
        return self._game_env.action_space

    @property
    def current_episode_id(self) -> str | None:
        return self._game_env.current_episode_id

    def reset(self) -> Any:
        return self._game_env.reset()

    def step(self, action: str) -> tuple[Any, float, bool, dict[str, Any]]:
        return self._game_env.step(action)

    def available_actions(self, state: Any | None = None) -> tuple[str, ...]:
        actions = self._game_env.available_actions(state)
        if "space" not in self._game_env.action_space or "space" in actions:
            return actions

        snapshot: GameStateSnapshot | None
        if isinstance(state, GameStateSnapshot):
            snapshot = state
        else:
            current_state = getattr(self._game_env, "_current_state", None)
            snapshot = current_state if isinstance(current_state, GameStateSnapshot) else None

        if _hybrid_resource_siphon_available(snapshot):
            return (*actions, "space")
        return actions

    def close(self) -> None:
        self._game_env.close()

    @classmethod
    def from_live_process(
        cls,
        *,
        executable_name: str,
        config: HybridLiveEnvConfig,
        movement_keys: str,
        include_prog_actions: bool,
        siphon_key: str,
        reset_sequence: tuple[str, ...] | None,
        launch_process_if_missing: bool,
        focus_window_on_attach: bool,
        window_targeted_input: bool,
        no_enemies_mode: bool,
        pre_reset_hook: Callable[[], None] | None = None,
    ) -> HybridLiveEnv:
        action_config = _build_action_config(
            movement_keys,
            include_prog_actions=include_prog_actions,
            siphon_key=siphon_key,
        )
        game_env = GameEnv.from_live_process(
            executable_name=executable_name,
            config=config.to_game_env_config(),
            reset_sequence=reset_sequence if reset_sequence else None,
            launch_process_if_missing=bool(launch_process_if_missing),
            focus_window_on_attach=bool(focus_window_on_attach),
            window_targeted_input=bool(window_targeted_input),
            action_config=action_config,
            pre_reset_hook=pre_reset_hook,
            reward_fn=_zero_reward,
            game_tick_ms=int(config.game_tick_ms),
            no_enemies_mode=bool(no_enemies_mode),
            disable_idle_frame_delay=bool(config.disable_idle_frame_delay),
            disable_background_motion=bool(config.disable_background_motion),
            disable_wall_animations=bool(config.disable_wall_animations),
        )
        return cls(game_env=game_env, no_enemies_mode=bool(no_enemies_mode))

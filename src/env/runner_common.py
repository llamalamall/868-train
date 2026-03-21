"""Shared helpers for live runner CLI modules."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src.controller.action_api import ActionConfig

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

_APP_SAVE_FOLDER_NAME = "868-HACK"
_APP_SAVE_FILE_NAME = "savegame_868"


def default_game_save_target_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / _APP_SAVE_FOLDER_NAME / _APP_SAVE_FILE_NAME
    return Path.home() / "AppData" / "Roaming" / _APP_SAVE_FOLDER_NAME / _APP_SAVE_FILE_NAME


def resolve_restore_save_source_path(args: argparse.Namespace) -> Path | None:
    if not args.restore_save_file:
        return None
    return Path(str(args.restore_save_file)).expanduser().resolve()


def restore_selected_save_file(*, source_path: Path, target_path: Path) -> None:
    source = source_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Selected restore save file does not exist: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Selected restore save file must be a file: {source}")

    target = target_path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        try:
            if source.samefile(target):
                return
        except OSError:
            pass

    shutil.copy2(source, target)


def game_tick_ms_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:  # pragma: no cover - argparse emits user-facing error.
        raise argparse.ArgumentTypeError("game tick must be an integer.") from error
    if parsed < 1 or parsed > 16:
        raise argparse.ArgumentTypeError("game tick ms must be between 1 and 16.")
    return parsed


def build_action_config(
    movement_keys: str,
    *,
    include_prog_actions: bool = True,
    siphon_key: str = "space",
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


def format_monitor_actions(actions: object, *, limit: int = 8) -> str:
    del limit
    if not isinstance(actions, (tuple, list)):
        return "-"
    normalized = tuple(str(item).strip() for item in actions if str(item).strip())
    if not normalized:
        return "-"
    return ",".join(normalized)

"""Simple Tkinter launcher for Hybrid run and evaluation workflows."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from src.gui.hybrid_presets import (
    AUTO_LATEST_BETA_META_CHECKPOINT as _PRESET_AUTO_LATEST_BETA_META_CHECKPOINT,
    HYBRID_CHECKPOINT_DIR as _PRESET_HYBRID_CHECKPOINT_DIR,
    HYBRID_META_CHECKPOINT_DIR as _PRESET_HYBRID_META_CHECKPOINT_DIR,
    APPDATA_GAME_SAVE_DIR as _PRESET_APPDATA_GAME_SAVE_DIR,
    REPO_ROOT as _PRESET_REPO_ROOT,
    initial_browse_dir as _initial_browse_dir_impl,
    latest_completed_meta_checkpoint as _latest_completed_meta_checkpoint_impl,
    resolve_preset_overrides as _resolve_preset_overrides_impl,
    run_hybrid_preset_overrides as _run_hybrid_preset_overrides_impl,
)
from src.gui.monitor_binding import (
    estimate_epsilon_eta_seconds as _estimate_epsilon_eta_seconds,
    format_duration_seconds as _format_duration_seconds,
    format_epsilon_progress_text as _format_epsilon_progress_text,
    format_phase_breakdown_tooltip as _format_phase_breakdown_tooltip,
    format_reward_breakdown_tooltip as _format_reward_breakdown_tooltip,
    load_live_monitor_binding_from_status_file as _load_live_monitor_binding_from_status_file,
    monitor_action_card_values as _monitor_action_card_values,
    monitor_key_label_for_action as _monitor_key_label_for_action,
    parse_bool_value as _parse_bool_value,
    parse_episode_progress as _parse_episode_progress,
    parse_next_available_actions as _parse_next_available_actions,
    parse_status_values as _parse_status_values,
    parse_step_value as _parse_step_value,
    resolve_reward_metric_value as _resolve_reward_metric_value,
)
from src.hybrid import runner as hybrid_runner
from src.memory.state_monitor_tui import (
    CONTROL_MODE_AUTO,
    CONTROL_MODE_PAUSED,
    MemoryStateMonitor,
    PollSnapshot,
    load_external_control_snapshot,
    set_external_control_mode,
    step_external_control,
)

_REPO_ROOT = _PRESET_REPO_ROOT
_PATH_LIKE_DESTS = {
    "exe",
    "checkpoint",
    "checkpoint_root",
    "resume_checkpoint",
    "warmstart_checkpoint",
    "restore_save_file",
}
_HIDDEN_GUI_DESTS = {"external_status_file", "external_control_file"}
_MAX_FORM_COLUMNS = 5
_HYBRID_CHECKPOINT_DIR = _PRESET_HYBRID_CHECKPOINT_DIR
_HYBRID_META_CHECKPOINT_DIR = _PRESET_HYBRID_META_CHECKPOINT_DIR
_APPDATA_GAME_SAVE_DIR = _PRESET_APPDATA_GAME_SAVE_DIR
_TEXTUAL_MARKUP_SEGMENT_PATTERN = re.compile(r"\[([a-zA-Z0-9_]+)\](.*?)\[/\]", re.DOTALL)
_REWARD_HISTORY_LIMIT = 500
_AUTO_LATEST_BETA_META_CHECKPOINT = _PRESET_AUTO_LATEST_BETA_META_CHECKPOINT
_MONITOR_KEY_LABELS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "UP",
    "LEFT",
    "DOWN",
    "RIGHT",
    "SPACE",
)
_PALETTE = {
    "bg": "#0b0f14",
    "surface": "#121922",
    "surface_alt": "#1a2533",
    "text": "#e6edf6",
    "muted": "#94a7bb",
    "accent": "#31c6b2",
    "accent_alt": "#ffb703",
    "accent_soft": "#162433",
    "danger": "#ff6b7a",
    "terminal_bg": "#070b11",
    "terminal_fg": "#d7e0ea",
}
_FONTS = {
    "title": ("Bahnschrift SemiBold", 19),
    "subtitle": ("Segoe UI", 9),
    "body": ("Segoe UI", 9),
    "mono": ("Consolas", 9),
    "small": ("Segoe UI", 8),
}
_TEXTUAL_STYLE_COLORS = {
    "bright_black": "#7f8d99",
    "yellow": "#f7b955",
    "cyan": "#5bd1c2",
    "green": "#5acc6f",
    "magenta": "#d48ef7",
    "red": "#ff7078",
    "bright_white": "#f5f9ff",
    "white": "#d7e0ea",
}


def _iter_parser_actions(parser: argparse.ArgumentParser) -> tuple[argparse.Action, ...]:
    actions: list[argparse.Action] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            continue
        if not action.option_strings:
            continue
        if action.dest in _HIDDEN_GUI_DESTS:
            continue
        actions.append(action)
    return tuple(actions)


def _sort_form_actions(actions: tuple[argparse.Action, ...]) -> tuple[argparse.Action, ...]:
    priority = {
        "exe": 0,
        "checkpoint": 1,
        "restore_save_file": 2,
    }
    indexed_actions = list(enumerate(actions))
    indexed_actions.sort(key=lambda item: (priority.get(item[1].dest, 999), item[0]))
    return tuple(action for _, action in indexed_actions)


def _get_subparser(parser: argparse.ArgumentParser, *, command_name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            subparser = action.choices.get(command_name)
            if isinstance(subparser, argparse.ArgumentParser):
                return subparser
    raise ValueError(f"Subparser '{command_name}' not found.")


def _is_boolean_optional(action: argparse.Action) -> bool:
    return isinstance(action, argparse.BooleanOptionalAction)


def _is_boolean_flag(action: argparse.Action) -> bool:
    if _is_boolean_optional(action):
        return True
    if action.nargs != 0:
        return False
    if not isinstance(action.default, bool):
        return False
    return isinstance(getattr(action, "const", None), bool)


def _primary_option(action: argparse.Action) -> str:
    for option in action.option_strings:
        if option.startswith("--no-"):
            continue
        return option
    return action.option_strings[0]


def _boolean_option_for_value(action: argparse.Action, *, value: bool) -> str:
    positive = [option for option in action.option_strings if not option.startswith("--no-")]
    negative = [option for option in action.option_strings if option.startswith("--no-")]
    if value:
        if positive:
            return positive[0]
        return action.option_strings[0]
    if negative:
        return negative[0]
    return action.option_strings[-1]


def _default_text(action: argparse.Action) -> str:
    if action.default is None:
        return ""
    return str(action.default)


def _display_label(action: argparse.Action) -> str:
    return _primary_option(action).removeprefix("--")


def _is_numeric_action(action: argparse.Action) -> bool:
    if action.type in {int, float}:
        return True
    default = action.default
    return isinstance(default, (int, float)) and not isinstance(default, bool)


def _numeric_step(action: argparse.Action) -> int | float:
    default = action.default
    is_int_like = action.type is int or (
        isinstance(default, int) and not isinstance(default, bool)
    )
    if is_int_like:
        magnitude = abs(int(default)) if isinstance(default, int) else 0
        if magnitude >= 10_000:
            return 1_000
        if magnitude >= 1_000:
            return 100
        if magnitude >= 100:
            return 10
        if magnitude >= 20:
            return 5
        return 1

    magnitude = abs(float(default)) if isinstance(default, (int, float)) else 0.0
    if magnitude <= 0:
        return 0.1
    exponent = math.floor(math.log10(magnitude))
    return float(10 ** (exponent - 1))


def _max_form_columns(action_count: int) -> int:
    if action_count >= 30:
        return _MAX_FORM_COLUMNS
    if action_count >= 18:
        return 4
    return 3


def _field_column_span(action: argparse.Action, *, max_columns: int) -> int:
    if max_columns < 4:
        return 1
    if action.dest in _PATH_LIKE_DESTS:
        return 2
    if _is_numeric_action(action) or _is_boolean_flag(action) or action.choices is not None:
        return 1
    return 2


def _widget_width_for_action(action: argparse.Action) -> int:
    if action.dest in _PATH_LIKE_DESTS:
        return 24
    if _is_numeric_action(action):
        return 7
    if action.choices is not None:
        longest_choice = max(len(str(choice)) for choice in action.choices)
        return max(9, min(longest_choice + 2, 14))
    return 13


def _initial_browse_dir(*, dest: str, current_value: str) -> Path:
    return _initial_browse_dir_impl(dest=dest, current_value=current_value)


def _run_hybrid_preset_overrides(*, command_name: str) -> dict[str, dict[str, object]]:
    return _run_hybrid_preset_overrides_impl(command_name=command_name)


def _latest_completed_meta_checkpoint(*, checkpoint_root: Path | None = None) -> Path:
    if checkpoint_root is None:
        return _latest_completed_meta_checkpoint_impl()
    return _latest_completed_meta_checkpoint_impl(checkpoint_root=checkpoint_root)


def _resolve_preset_overrides(*, overrides: dict[str, object]) -> dict[str, object]:
    return _resolve_preset_overrides_impl(
        overrides=overrides,
        latest_checkpoint_resolver=_latest_completed_meta_checkpoint,
    )


def _validate_text_input(action: argparse.Action, *, value: str, field_name: str) -> None:
    if action.type is not None:
        try:
            action.type(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} has invalid value '{value}': {error}") from error
    if action.choices is not None:
        choices = {str(choice) for choice in action.choices}
        if value not in choices:
            raise ValueError(
                f"{field_name} must be one of: {', '.join(str(choice) for choice in action.choices)}."
            )


def _format_command(command: list[str]) -> str:
    if not command:
        return ""
    return subprocess.list2cmdline(command)


@dataclass
class _FormField:
    action: argparse.Action
    variable: tk.Variable


class _ArgForm(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        profile_id: str,
        parser: argparse.ArgumentParser,
        module_args: tuple[str, ...],
        presets: dict[str, dict[str, object]] | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, padding=(6, 6, 6, 4), style="Surface.TFrame")
        self._profile_id = profile_id
        self._module_args = module_args
        self._fields: list[_FormField] = []
        self._field_by_dest: dict[str, _FormField] = {}
        self._presets = presets or {}
        self._on_change = on_change
        self._default_help = "Hover or focus a field to show option help."
        self._help_text = tk.StringVar(value=self._default_help)
        self._selected_preset = tk.StringVar()

        self.columnconfigure(0, weight=1)

        body_row = 0
        if self._presets:
            preset_row = ttk.Frame(self, padding=(8, 4, 8, 6), style="Surface.TFrame")
            preset_row.grid(row=0, column=0, sticky="ew")
            preset_row.columnconfigure(2, weight=1)
            ttk.Label(preset_row, text="Preset", style="FormLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
            )
            preset_names = tuple(self._presets.keys())
            self._selected_preset.set(preset_names[0])
            preset_combo = ttk.Combobox(
                preset_row,
                textvariable=self._selected_preset,
                values=preset_names,
                state="readonly",
                width=24,
            )
            preset_combo.grid(row=0, column=1, sticky="w", padx=(6, 6))
            preset_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_selected_preset())
            apply_preset = ttk.Button(preset_row, text="Apply", command=self._apply_selected_preset)
            apply_preset.grid(row=0, column=2, sticky="w")
            self._bind_help(
                preset_combo,
                "Apply a pre-populated settings profile for common reward configurations.",
            )
            self._bind_help(
                apply_preset,
                "Load selected preset values into the form fields.",
            )
            body_row = 1
        self.rowconfigure(body_row, weight=1)

        body_shell = ttk.Frame(self, style="Surface.TFrame")
        body_shell.grid(row=body_row, column=0, sticky="nsew")
        body_shell.columnconfigure(0, weight=1)
        body_shell.rowconfigure(0, weight=1)

        body_canvas = tk.Canvas(
            body_shell,
            background=_PALETTE["surface"],
            highlightthickness=0,
            bd=0,
        )
        body_canvas.grid(row=0, column=0, sticky="nsew")
        body_vscroll = ttk.Scrollbar(body_shell, orient="vertical", command=body_canvas.yview)
        body_vscroll.grid(row=0, column=1, sticky="ns")
        body_hscroll = ttk.Scrollbar(body_shell, orient="horizontal", command=body_canvas.xview)
        body_hscroll.grid(row=1, column=0, sticky="ew")
        body_canvas.configure(yscrollcommand=body_vscroll.set, xscrollcommand=body_hscroll.set)

        body = ttk.Frame(body_canvas, padding=(4, 2, 4, 2), style="Surface.TFrame")
        body_window = body_canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_resize(_: tk.Event[tk.Misc]) -> None:
            body_canvas.configure(scrollregion=body_canvas.bbox("all"))

        def _on_canvas_resize(event: tk.Event[tk.Misc]) -> None:
            body_canvas.itemconfigure(body_window, width=event.width)

        body.bind("<Configure>", _on_body_resize)
        body_canvas.bind("<Configure>", _on_canvas_resize)

        actions = _sort_form_actions(_iter_parser_actions(parser))
        max_columns = _max_form_columns(len(actions))
        action_dests = {action.dest for action in actions}
        if "exe" in action_dests and "checkpoint" in action_dests:
            max_columns = max(4, max_columns)
        for column_index in range(max_columns):
            body.columnconfigure(column_index, weight=1, uniform="arg-columns")

        row_index = 0
        column_index = 0
        for action in actions:
            span = min(_field_column_span(action, max_columns=max_columns), max_columns)
            if column_index + span > max_columns:
                row_index += 1
                column_index = 0

            card = ttk.Frame(body, padding=(6, 4, 6, 4), style="ArgCard.TFrame")
            card.grid(
                row=row_index,
                column=column_index,
                columnspan=span,
                sticky="ew",
                padx=3,
                pady=3,
            )
            card.columnconfigure(0, weight=1)

            help_text = action.help or ""
            label_text = _display_label(action)
            if _is_boolean_flag(action):
                variable = tk.BooleanVar(value=bool(action.default))
                widget = ttk.Checkbutton(
                    card,
                    variable=variable,
                    text=label_text,
                    style="FormCheck.TCheckbutton",
                )
                widget.grid(row=0, column=0, sticky="w")
                self._bind_help(widget, help_text)
                field = _FormField(action=action, variable=variable)
                self._fields.append(field)
                self._field_by_dest[action.dest] = field
            else:
                label = ttk.Label(card, text=label_text, style="FormLabel.TLabel")
                label.grid(row=0, column=0, sticky="w")
                self._bind_help(label, help_text)
                variable = tk.StringVar(value=_default_text(action))
                widget = self._build_value_widget(card, action=action, variable=variable)
                self._bind_help(widget, help_text)
                field = _FormField(action=action, variable=variable)
                self._fields.append(field)
                self._field_by_dest[action.dest] = field

            column_index += span
            if column_index >= max_columns:
                row_index += 1
                column_index = 0

        help_label = ttk.Label(
            self,
            textvariable=self._help_text,
            style="FormHelp.TLabel",
            padding=(8, 4, 8, 2),
            anchor="w",
        )
        help_label.grid(row=body_row + 1, column=0, sticky="ew")

    @property
    def profile_id(self) -> str:
        return self._profile_id

    def value_for_dest(self, dest: str) -> str | bool | None:
        field = self._field_by_dest.get(dest)
        if field is None:
            return None
        if isinstance(field.variable, tk.BooleanVar):
            return bool(field.variable.get())
        return str(field.variable.get()).strip()

    def _apply_selected_preset(self) -> None:
        preset_name = self._selected_preset.get()
        overrides = self._presets.get(preset_name)
        if overrides is None:
            return
        try:
            resolved_overrides = _resolve_preset_overrides(overrides=overrides)
        except ValueError as error:
            messagebox.showerror("Preset Error", str(error), parent=self)
            return

        for field in self._fields:
            action = field.action
            default = action.default
            if isinstance(field.variable, tk.BooleanVar):
                field.variable.set(bool(default))
                continue
            if default is None:
                field.variable.set("")
            else:
                field.variable.set(str(default))

        for dest, value in resolved_overrides.items():
            field = self._field_by_dest.get(dest)
            if field is None:
                continue
            if isinstance(field.variable, tk.BooleanVar):
                field.variable.set(bool(value))
            else:
                if value is None:
                    field.variable.set("")
                else:
                    field.variable.set(str(value))

        if callable(self._on_change):
            self._on_change()

    def _bind_help(self, widget: tk.Misc, help_text: str) -> None:
        if not help_text:
            return

        def _show_help(_: tk.Event[tk.Misc]) -> None:
            self._help_text.set(help_text)

        def _restore_help(_: tk.Event[tk.Misc]) -> None:
            self._help_text.set(self._default_help)

        widget.bind("<FocusIn>", _show_help)
        widget.bind("<FocusOut>", _restore_help)
        widget.bind("<Enter>", _show_help)
        widget.bind("<Leave>", _restore_help)

    def _build_value_widget(
        self,
        card: ttk.Frame,
        *,
        action: argparse.Action,
        variable: tk.StringVar,
    ) -> tk.Misc:
        if action.choices is not None:
            default_value = _default_text(action)
            initial = default_value or str(next(iter(action.choices)))
            variable.set(initial)
            widget = ttk.Combobox(
                card,
                textvariable=variable,
                values=[str(choice) for choice in action.choices],
                state="readonly",
                width=_widget_width_for_action(action),
            )
            widget.grid(row=1, column=0, sticky="w")
            return widget

        if _is_numeric_action(action):
            widget = ttk.Spinbox(
                card,
                textvariable=variable,
                from_=-1_000_000_000,
                to=1_000_000_000,
                increment=_numeric_step(action),
                width=_widget_width_for_action(action),
                justify="right",
            )
            widget.grid(row=1, column=0, sticky="w")
            return widget

        if action.dest in _PATH_LIKE_DESTS:
            row = ttk.Frame(card)
            row.grid(row=1, column=0, sticky="ew")
            row.columnconfigure(0, weight=1)
            widget = ttk.Entry(
                row,
                textvariable=variable,
                width=max(_widget_width_for_action(action), 20),
            )
            widget.grid(row=0, column=0, sticky="ew")
            browse_button = ttk.Button(
                row,
                text="Browse",
                style="Secondary.TButton",
                width=7,
                command=lambda var=variable, dest=action.dest: self._browse_path(var, dest),
            )
            browse_button.grid(row=0, column=1, sticky="e", padx=(4, 0))
            self._bind_help(browse_button, action.help or "")
            return widget

        widget = ttk.Entry(card, textvariable=variable, width=_widget_width_for_action(action))
        widget.grid(row=1, column=0, sticky="ew")
        return widget

    def _browse_path(self, variable: tk.StringVar, dest: str) -> None:
        current_value = variable.get().strip()
        initial_dir = str(_initial_browse_dir(dest=dest, current_value=current_value))
        if dest == "checkpoint" and self._profile_id.startswith("hybrid-") and not current_value:
            initial_dir = str(_HYBRID_CHECKPOINT_DIR)
        if dest == "exe":
            path = filedialog.askopenfilename(
                parent=self,
                title="Select executable",
                initialdir=initial_dir,
                filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            )
        elif dest == "checkpoint":
            path = filedialog.askdirectory(
                parent=self,
                title="Select hybrid checkpoint directory",
                initialdir=initial_dir,
            )
        elif dest in {"resume_checkpoint", "warmstart_checkpoint"}:
            path = filedialog.askdirectory(
                parent=self,
                title="Select hybrid checkpoint directory",
                initialdir=initial_dir,
            )
        elif dest == "checkpoint_root":
            path = filedialog.askdirectory(
                parent=self,
                title="Select checkpoint root directory",
                initialdir=initial_dir,
            )
        elif dest == "restore_save_file":
            path = filedialog.askopenfilename(
                parent=self,
                title="Select source save file",
                initialdir=initial_dir,
                filetypes=[("All files", "*.*")],
            )
        else:
            path = filedialog.asksaveasfilename(
                parent=self,
                title="Select output path",
                initialdir=initial_dir,
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
        if path:
            variable.set(path)

    def build_command(self) -> list[str]:
        cli_args: list[str] = []
        for field in self._fields:
            action = field.action
            field_name = _primary_option(action)
            if _is_boolean_flag(action):
                value = bool(field.variable.get())
                default = bool(action.default)
                if value != default:
                    cli_args.append(_boolean_option_for_value(action, value=value))
                continue

            text_value = str(field.variable.get()).strip()
            default_text = _default_text(action)
            primary = _primary_option(action)

            if action.required and text_value == "":
                raise ValueError(f"{field_name} is required.")

            if action.nargs == "?":
                if not action.required and text_value == default_text:
                    continue
                cli_args.append(primary)
                if text_value != "":
                    _validate_text_input(action, value=text_value, field_name=field_name)
                    cli_args.append(text_value)
                continue

            if text_value == "":
                if action.required:
                    raise ValueError(f"{field_name} is required.")
                continue

            _validate_text_input(action, value=text_value, field_name=field_name)
            if not action.required and text_value == default_text:
                continue
            cli_args.extend((primary, text_value))

        return [sys.executable, *self._module_args, *cli_args]


class HybridRunnerGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("868 Hybrid Launcher")
        self.geometry("1210x830")
        self.minsize(980, 680)
        self.configure(background=_PALETTE["bg"])
        self._setup_styles()

        self._event_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._forms: dict[str, _ArgForm] = {}
        self._status_text = tk.StringVar(value="READY")
        self._status_phase = 0
        self._last_form: _ArgForm | None = None
        self._external_status_file: Path | None = None
        self._external_control_file: Path | None = None
        self._owns_external_status_file = False
        self._owns_external_control_file = False
        self._runner_monitor_pid: int | None = None
        self._last_runner_monitor_bind_attempt_monotonic = 0.0
        self._runner_monitor_status = tk.StringVar(value="runner_session=idle")
        self._status_snapshot: tuple[str, str, str, str] = ("", "", "", "")
        self._monitor_training_line = tk.StringVar(value="training=idle")
        self._monitor_action_line = tk.StringVar(value="action=idle")
        self._monitor_reward_line = tk.StringVar(value="reward=idle")
        self._monitor_next_available_actions_line = tk.StringVar(
            value="next_available_actions=unavailable"
        )
        self._monitor_control_state = tk.StringVar(value="session=idle")
        self._monitor_pause_button: ttk.Button | None = None
        self._monitor_step_button: ttk.Button | None = None
        self._monitor_resume_button: ttk.Button | None = None
        self._monitor_metric_vars: dict[str, tk.StringVar] = {}
        self._reward_history: list[float] = []
        self._monitor_keycap_labels: dict[str, tk.Label] = {}
        self._monitor_available_key_labels: set[str] = set()
        self._monitor_selected_key_label: str | None = None
        self._reward_tooltip: tk.Toplevel | None = None
        self._reward_tooltip_label: tk.Label | None = None
        self._reward_hover_widgets: tuple[tk.Misc, ...] = ()
        self._phase_tooltip: tk.Toplevel | None = None
        self._phase_tooltip_label: tk.Label | None = None
        self._phase_hover_widgets: tuple[tk.Misc, ...] = ()
        self._epsilon_progress_value = tk.DoubleVar(value=0.0)
        self._epsilon_progress_text = tk.StringVar(value="0.0%")
        self._monitor_total_episodes: int | None = None
        self._monitor_epsilon_start: float | None = None
        self._monitor_epsilon_end: float | None = None
        self._monitor_epsilon_decay_steps: int | None = None
        self._monitor_track_epsilon_eta = False
        self._monitor_first_step_time: float | None = None
        self._monitor_step_events = 0
        self._monitor_last_step_key: tuple[int, int] | None = None
        self._monitor_episode_start_times: dict[int, float] = {}
        self._monitor_completed_episode_ids: set[int] = set()
        self._monitor_episode_duration_sum = 0.0
        self._monitor_episode_duration_count = 0
        self._monitor_last_reward_step_key: tuple[int, int] | None = None
        self._state_monitor: MemoryStateMonitor | None = None
        self._state_monitor_error = tk.StringVar(value="")
        self._state_monitor_status = tk.StringVar(value="state_monitor=idle")
        self._state_monitor_timestamp = tk.StringVar(value="last_snapshot=-")
        self._state_monitor_pid = tk.StringVar(value="pid=-")
        self._live_monitor_tab_widget: ttk.Frame | None = None
        self._state_tab_widget: ttk.Frame | None = None
        self._ascii_maps_tab_widget: ttk.Frame | None = None
        self._state_fields_tree: ttk.Treeview | None = None
        self._ascii_maps_text: tk.Text | None = None
        self._state_last_poll_monotonic = 0.0
        self._state_poll_interval_seconds = 0.5

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=4)
        self.rowconfigure(3, weight=1)

        hero = ttk.Frame(self, padding=(14, 12, 14, 10), style="Hero.TFrame")
        hero.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        hero.columnconfigure(0, weight=1)
        hero.columnconfigure(1, weight=0)

        title = ttk.Label(
            hero,
            text="868 Runner Control Deck",
            style="HeroTitle.TLabel",
        )
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            hero,
            text="Train and evaluate the Hybrid stack from one surface with every CLI flag exposed.",
            style="HeroSubtitle.TLabel",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._status_label = ttk.Label(
            hero,
            textvariable=self._status_text,
            style="StatusReady.TLabel",
            anchor="center",
            padding=(14, 6),
        )
        self._status_label.grid(row=0, column=1, rowspan=2, sticky="e")

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self._notebook.bind("<<NotebookTabChanged>>", lambda _: self._on_tab_changed())

        self._build_profiles()

        controls = ttk.Frame(self, padding=(10, 8, 10, 8), style="Surface.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        self._command_preview = tk.StringVar(value="")
        ttk.Label(controls, text="Command Preview", style="SectionLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        preview_entry = ttk.Entry(
            controls,
            textvariable=self._command_preview,
            font=_FONTS["mono"],
        )
        preview_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        action_strip = ttk.Frame(controls, style="Surface.TFrame")
        action_strip.grid(row=2, column=0, sticky="ew")
        action_strip.columnconfigure(0, weight=1)
        action_strip.columnconfigure(1, weight=0)
        action_strip.columnconfigure(2, weight=1)

        self._preview_button = ttk.Button(
            action_strip,
            text="Preview Command",
            command=self._refresh_preview,
            style="Secondary.TButton",
        )
        self._preview_button.grid(row=0, column=0, sticky="w")

        run_group = ttk.Frame(action_strip, style="Surface.TFrame")
        run_group.grid(row=0, column=1)

        self._run_button = ttk.Button(
            run_group,
            text="Launch",
            command=self._run_selected,
            style="Primary.TButton",
        )
        self._run_button.grid(row=0, column=0)
        self._stop_button = ttk.Button(
            run_group,
            text="Stop",
            command=self._stop_process,
            state="disabled",
            style="Danger.TButton",
        )
        self._stop_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._monitor_pause_button = ttk.Button(
            run_group,
            text="||",
            command=self._pause_monitor_session,
            style="MonitorPause.TButton",
            state="disabled",
            width=3,
        )
        self._monitor_pause_button.grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._monitor_step_button = ttk.Button(
            run_group,
            text=">|",
            command=self._step_monitor_session,
            style="MonitorStep.TButton",
            state="disabled",
            width=3,
        )
        self._monitor_step_button.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self._monitor_resume_button = ttk.Button(
            run_group,
            text=">",
            command=self._resume_monitor_session,
            style="MonitorResume.TButton",
            state="disabled",
            width=3,
        )
        self._monitor_resume_button.grid(row=0, column=4, sticky="w", padx=(8, 0))
        ttk.Label(
            action_strip,
            textvariable=self._monitor_control_state,
            style="FormHelp.TLabel",
        ).grid(row=0, column=2, sticky="e")

        output_frame = ttk.Frame(self, padding=(10, 0, 10, 10), style="Surface.TFrame")
        output_frame.grid(row=3, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)

        ttk.Label(output_frame, text="Live Output", style="SectionLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self._output = tk.Text(
            output_frame,
            wrap="none",
            height=10,
            bg=_PALETTE["terminal_bg"],
            fg=_PALETTE["terminal_fg"],
            insertbackground=_PALETTE["terminal_fg"],
            relief="flat",
            padx=8,
            pady=8,
            font=_FONTS["mono"],
        )
        self._output.grid(row=1, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self._output.yview)
        output_scroll.grid(row=1, column=1, sticky="ns")
        output_scroll_x = ttk.Scrollbar(output_frame, orient="horizontal", command=self._output.xview)
        output_scroll_x.grid(row=2, column=0, sticky="ew")
        self._output.configure(yscrollcommand=output_scroll.set, xscrollcommand=output_scroll_x.set)
        self._output.tag_configure("command", foreground="#7fdbff")
        self._output.tag_configure("system", foreground="#f8c537")
        self._output.tag_configure("error", foreground="#ff7a8a")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._animate_status)
        self.after(100, self._drain_event_queue)
        self.after(200, self._poll_external_status)
        self.after(300, self._poll_state_monitor)
        self._refresh_tab_visuals()
        self._refresh_preview()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=_PALETTE["bg"], foreground=_PALETTE["text"], font=_FONTS["body"])
        style.configure("Surface.TFrame", background=_PALETTE["surface"])
        style.configure("Hero.TFrame", background=_PALETTE["accent_soft"])

        style.configure(
            "HeroTitle.TLabel",
            background=_PALETTE["accent_soft"],
            foreground=_PALETTE["text"],
            font=_FONTS["title"],
        )
        style.configure(
            "HeroSubtitle.TLabel",
            background=_PALETTE["accent_soft"],
            foreground=_PALETTE["muted"],
            font=_FONTS["subtitle"],
        )
        style.configure("SectionLabel.TLabel", background=_PALETTE["surface"], font=("Segoe UI Semibold", 10))
        style.configure("FormLabel.TLabel", background=_PALETTE["surface"], foreground=_PALETTE["muted"])
        style.configure("FormHelp.TLabel", background=_PALETTE["surface"], foreground=_PALETTE["muted"], font=_FONTS["small"])
        style.configure("ArgCard.TFrame", background=_PALETTE["surface_alt"])
        style.configure("FormCheck.TCheckbutton", background=_PALETTE["surface_alt"])
        style.configure("MetricCard.TFrame", background="#15212e")
        style.configure(
            "MetricName.TLabel",
            background="#15212e",
            foreground=_PALETTE["muted"],
            font=("Segoe UI", 8),
        )
        style.configure(
            "MetricValue.TLabel",
            background="#15212e",
            foreground=_PALETTE["text"],
            font=("Consolas", 11, "bold"),
        )
        style.configure("StatusReady.TLabel", background=_PALETTE["surface"], foreground=_PALETTE["accent"])
        style.configure("StatusRun.TLabel", background="#2a3522", foreground=_PALETTE["accent_alt"])
        style.configure("StatusStop.TLabel", background="#3d2026", foreground=_PALETTE["danger"])

        style.configure("TNotebook", background=_PALETTE["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=_PALETTE["surface_alt"],
            foreground=_PALETTE["muted"],
            padding=(10, 5),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", _PALETTE["surface"])],
            foreground=[("selected", _PALETTE["accent"])],
        )

        style.configure("TEntry", fieldbackground="#0f1620", foreground=_PALETTE["text"], borderwidth=1)
        style.configure("TCombobox", fieldbackground="#0f1620", foreground=_PALETTE["text"])
        style.configure("TSpinbox", fieldbackground="#0f1620", foreground=_PALETTE["text"])
        style.configure(
            "State.Treeview",
            background="#0f1620",
            fieldbackground="#0f1620",
            foreground=_PALETTE["text"],
            bordercolor="#243244",
            rowheight=22,
        )
        style.map(
            "State.Treeview",
            background=[("selected", "#1b2a3b")],
            foreground=[("selected", _PALETTE["text"])],
        )
        style.configure(
            "State.Treeview.Heading",
            background="#1a2533",
            foreground=_PALETTE["accent"],
            bordercolor="#243244",
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "State.Treeview.Heading",
            background=[("active", "#223246")],
            foreground=[("active", _PALETTE["accent"])],
        )
        style.configure(
            "MonitorEpsilon.Horizontal.TProgressbar",
            troughcolor="#0f1620",
            background=_PALETTE["accent"],
            bordercolor="#243244",
            lightcolor=_PALETTE["accent"],
            darkcolor=_PALETTE["accent"],
            thickness=14,
        )

        style.configure("Primary.TButton", background="#1f8f81", foreground="#ffffff", borderwidth=0, padding=(10, 5))
        style.map("Primary.TButton", background=[("active", "#28a594"), ("disabled", "#345955")])
        style.configure("Secondary.TButton", background="#2a3a4d", foreground=_PALETTE["text"], borderwidth=0, padding=(8, 5))
        style.map("Secondary.TButton", background=[("active", "#35485f")])
        style.configure("Danger.TButton", background="#5a2630", foreground="#ffdce1", borderwidth=0, padding=(8, 5))
        style.map("Danger.TButton", background=[("active", "#70313d"), ("disabled", "#3f2730")])
        style.configure(
            "MonitorPause.TButton",
            background="#3c3426",
            foreground="#ffd38a",
            borderwidth=0,
            padding=(7, 4),
            font=("Consolas", 10, "bold"),
        )
        style.map(
            "MonitorPause.TButton",
            background=[("active", "#534834"), ("disabled", "#2b2a28")],
            foreground=[("disabled", "#786b58")],
        )
        style.configure(
            "MonitorStep.TButton",
            background="#2a3a4d",
            foreground=_PALETTE["text"],
            borderwidth=0,
            padding=(7, 4),
            font=("Consolas", 10, "bold"),
        )
        style.map(
            "MonitorStep.TButton",
            background=[("active", "#35485f"), ("disabled", "#222b34")],
            foreground=[("disabled", "#738396")],
        )
        style.configure(
            "MonitorResume.TButton",
            background="#1f8f81",
            foreground="#ffffff",
            borderwidth=0,
            padding=(7, 4),
            font=("Consolas", 10, "bold"),
        )
        style.map(
            "MonitorResume.TButton",
            background=[("active", "#28a594"), ("disabled", "#345955")],
            foreground=[("disabled", "#93b7b1")],
        )

    def _build_profiles(self) -> None:
        hybrid_parser = hybrid_runner._build_parser()
        hybrid_movement_parser = _get_subparser(hybrid_parser, command_name="movement-test")
        hybrid_train_meta_parser = _get_subparser(hybrid_parser, command_name="train-meta-no-enemies")
        hybrid_train_full_parser = _get_subparser(hybrid_parser, command_name="train-full-hierarchical")
        hybrid_eval_parser = _get_subparser(hybrid_parser, command_name="eval-hybrid")

        profiles = (
            (
                "hybrid-movement",
                "Hybrid Movement Test",
                hybrid_movement_parser,
                ("-m", "src.hybrid.runner", "movement-test"),
                _run_hybrid_preset_overrides(command_name="movement-test"),
            ),
            (
                "hybrid-train-meta",
                "Hybrid Meta Train (No Enemies)",
                hybrid_train_meta_parser,
                ("-m", "src.hybrid.runner", "train-meta-no-enemies"),
                _run_hybrid_preset_overrides(command_name="train-meta-no-enemies"),
            ),
            (
                "hybrid-train-full",
                "Hybrid Full Train",
                hybrid_train_full_parser,
                ("-m", "src.hybrid.runner", "train-full-hierarchical"),
                _run_hybrid_preset_overrides(command_name="train-full-hierarchical"),
            ),
            (
                "hybrid-eval",
                "Hybrid Evaluate",
                hybrid_eval_parser,
                ("-m", "src.hybrid.runner", "eval-hybrid"),
                _run_hybrid_preset_overrides(command_name="eval-hybrid"),
            ),
        )

        for profile_id, title, parser, module_args, presets in profiles:
            form = _ArgForm(
                self._notebook,
                profile_id=profile_id,
                parser=parser,
                module_args=module_args,
                presets=presets,
                on_change=self._refresh_preview,
            )
            self._forms[profile_id] = form
            self._notebook.add(form, text=title)
            if self._last_form is None:
                self._last_form = form
        self._build_monitor_tab()
        self._build_state_tab()
        self._build_ascii_maps_tab()
        self._refresh_tab_visuals()

    def _build_monitor_tab(self) -> None:
        monitor = ttk.Frame(self._notebook, padding=(10, 10, 10, 10), style="Surface.TFrame")
        monitor.columnconfigure(0, weight=1)
        monitor.rowconfigure(3, weight=1)

        ttk.Label(
            monitor,
            text="Run Monitor",
            style="SectionLabel.TLabel",
        ).grid(row=0, column=0, sticky="w")

        metric_grid = ttk.Frame(monitor, style="Surface.TFrame")
        metric_grid.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        for col in range(4):
            metric_grid.columnconfigure(col, weight=1, uniform="metric-cols")

        metric_keys = (
            ("Episode", "episode"),
            ("Step", "step"),
            ("Reward", "reward"),
            ("Phase", "phase"),
            ("Total", "total"),
            ("Epsilon", "epsilon"),
            ("Threat Epsilon", "threat_epsilon"),
            ("Updates", "updates"),
            ("Done", "done"),
            ("Terminal", "terminal"),
            ("ETA Epsilon", "eta_epsilon"),
            ("ETA Session", "eta_session"),
        )
        for idx, (label, key) in enumerate(metric_keys):
            row = idx // 4
            col = idx % 4
            card = ttk.Frame(metric_grid, style="MetricCard.TFrame", padding=(8, 6, 8, 6))
            card.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
            self._monitor_metric_vars[key] = tk.StringVar(value="-")
            name_label = ttk.Label(card, text=label, style="MetricName.TLabel")
            name_label.grid(row=0, column=0, sticky="w")
            value_label = ttk.Label(card, textvariable=self._monitor_metric_vars[key], style="MetricValue.TLabel")
            value_label.grid(
                row=1,
                column=0,
                sticky="w",
            )
            if key == "reward":
                self._reward_hover_widgets = (card, name_label, value_label)
                for widget in self._reward_hover_widgets:
                    widget.bind("<Enter>", self._show_reward_tooltip, add="+")
                    widget.bind("<Motion>", self._move_reward_tooltip, add="+")
                    widget.bind("<Leave>", self._hide_reward_tooltip, add="+")
            if key == "phase":
                self._phase_hover_widgets = (card, name_label, value_label)
                for widget in self._phase_hover_widgets:
                    widget.bind("<Enter>", self._show_phase_tooltip, add="+")
                    widget.bind("<Motion>", self._move_phase_tooltip, add="+")
                    widget.bind("<Leave>", self._hide_phase_tooltip, add="+")

        monitor_meta = ttk.Frame(monitor, style="Surface.TFrame")
        monitor_meta.grid(row=2, column=0, sticky="ew", pady=(2, 8))
        monitor_meta.columnconfigure(0, weight=0)
        monitor_meta.columnconfigure(1, weight=0)
        monitor_meta.columnconfigure(2, weight=1)
        monitor_meta.columnconfigure(3, weight=1)
        monitor_meta.columnconfigure(4, weight=1)
        ttk.Button(
            monitor_meta,
            text="Attach",
            command=self._ensure_state_monitor_started,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            monitor_meta,
            text="Detach",
            command=self._stop_state_monitor,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Label(monitor_meta, textvariable=self._state_monitor_status, style="FormHelp.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
        )
        ttk.Label(monitor_meta, textvariable=self._state_monitor_pid, style="FormHelp.TLabel").grid(
            row=0,
            column=3,
            sticky="w",
        )
        ttk.Label(monitor_meta, textvariable=self._state_monitor_timestamp, style="FormHelp.TLabel").grid(
            row=0,
            column=4,
            sticky="e",
        )
        ttk.Label(monitor_meta, textvariable=self._runner_monitor_status, style="FormHelp.TLabel").grid(
            row=1,
            column=2,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )

        graph_shell = ttk.Frame(monitor, style="Surface.TFrame")
        graph_shell.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        graph_shell.columnconfigure(0, weight=1)
        graph_shell.rowconfigure(3, weight=1)

        ttk.Label(graph_shell, text="Input Deck", style="SectionLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        input_shell = tk.Frame(
            graph_shell,
            bg="#0d141d",
            highlightthickness=1,
            highlightbackground="#243244",
            padx=12,
            pady=10,
        )
        input_shell.grid(row=1, column=0, sticky="ew", pady=(6, 10))
        for col in range(10):
            input_shell.grid_columnconfigure(col, weight=1, uniform="monitor-input-cols")
        for key_label in _MONITOR_KEY_LABELS[:10]:
            key_widget = tk.Label(
                input_shell,
                text=key_label,
                font=("Consolas", 10, "bold"),
                bd=1,
                relief="solid",
                width=4,
                padx=2,
                pady=6,
            )
            key_widget.grid(row=0, column=("1234567890".index(key_label)), padx=3, pady=3, sticky="ew")
            self._monitor_keycap_labels[key_label] = key_widget
        for key_label, row, col in (
            ("UP", 1, 4),
            ("LEFT", 2, 3),
            ("DOWN", 2, 4),
            ("RIGHT", 2, 5),
        ):
            key_widget = tk.Label(
                input_shell,
                text=key_label,
                font=("Consolas", 10, "bold"),
                bd=1,
                relief="solid",
                width=6 if key_label == "RIGHT" else 5,
                padx=2,
                pady=6,
            )
            key_widget.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            self._monitor_keycap_labels[key_label] = key_widget
        space_widget = tk.Label(
            input_shell,
            text="SPACE",
            font=("Consolas", 10, "bold"),
            bd=1,
            relief="solid",
            padx=2,
            pady=6,
        )
        space_widget.grid(row=3, column=2, columnspan=6, padx=3, pady=(5, 2), sticky="ew")
        self._monitor_keycap_labels["SPACE"] = space_widget

        epsilon_row = ttk.Frame(graph_shell, style="Surface.TFrame")
        epsilon_row.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        epsilon_row.columnconfigure(0, weight=1)
        ttk.Label(epsilon_row, text="epsilon_progress", style="FormLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            epsilon_row,
            textvariable=self._epsilon_progress_text,
            style="FormHelp.TLabel",
        ).grid(row=0, column=1, sticky="e")
        self._epsilon_progress = ttk.Progressbar(
            epsilon_row,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self._epsilon_progress_value,
            style="MonitorEpsilon.Horizontal.TProgressbar",
        )
        self._epsilon_progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self._reward_canvas = tk.Canvas(
            graph_shell,
            height=180,
            bg="#0d141d",
            highlightthickness=1,
            highlightbackground="#243244",
        )
        self._reward_canvas.grid(row=3, column=0, sticky="nsew")
        self._reward_canvas.bind("<Configure>", lambda _: self._draw_reward_graph())
        self._update_monitor_keycaps(action_line="", next_available_actions_line="")
        self._set_monitor_controls_enabled(False)

        self._live_monitor_tab_widget = monitor
        self._notebook.add(monitor, text="Live Monitor")

    def _refresh_tab_visuals(self) -> None:
        selected = self._notebook.select()
        for tab_id in self._notebook.tabs():
            self._notebook.tab(tab_id, padding=(18, 10) if tab_id == selected else (10, 5))

    def _on_tab_changed(self) -> None:
        self._refresh_tab_visuals()
        self._refresh_preview()
        selected_tab = self._notebook.select()
        current_widget = self.nametowidget(selected_tab)
        if (
            (self._live_monitor_tab_widget is not None and current_widget == self._live_monitor_tab_widget)
            or
            (self._state_tab_widget is not None and current_widget == self._state_tab_widget)
            or (self._ascii_maps_tab_widget is not None and current_widget == self._ascii_maps_tab_widget)
        ):
            self._ensure_state_monitor_started()

    def _current_form(self) -> _ArgForm:
        selected_tab = self._notebook.select()
        current_widget = self.nametowidget(selected_tab)
        if isinstance(current_widget, _ArgForm):
            self._last_form = current_widget
            return current_widget
        if self._last_form is not None:
            return self._last_form
        if self._forms:
            self._last_form = next(iter(self._forms.values()))
            return self._last_form
        raise RuntimeError("No runnable profile form is available.")

    def _refresh_preview(self) -> None:
        try:
            command = self._current_form().build_command()
            self._command_preview.set(_format_command(command))
        except ValueError as error:
            self._command_preview.set(f"Invalid settings: {error}")

    @staticmethod
    def _parse_int_or_none(value: str | bool | None) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_float_or_none(value: str | bool | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _reset_monitor_estimates(self) -> None:
        self._monitor_total_episodes = None
        self._monitor_epsilon_start = None
        self._monitor_epsilon_end = None
        self._monitor_epsilon_decay_steps = None
        self._monitor_track_epsilon_eta = False
        self._monitor_first_step_time = None
        self._monitor_step_events = 0
        self._monitor_last_step_key = None
        self._monitor_episode_start_times.clear()
        self._monitor_completed_episode_ids.clear()
        self._monitor_episode_duration_sum = 0.0
        self._monitor_episode_duration_count = 0
        self._monitor_last_reward_step_key = None

    def _configure_monitor_estimates(self, *, form: _ArgForm) -> None:
        self._reset_monitor_estimates()

        total_episodes = self._parse_int_or_none(form.value_for_dest("episodes"))
        if total_episodes is not None and total_episodes >= 1:
            self._monitor_total_episodes = total_episodes

        epsilon_start: float | None = None
        epsilon_end: float | None = None
        epsilon_decay_steps: int | None = None

        if form.profile_id in {"hybrid-train-meta", "hybrid-train-full"}:
            epsilon_start = self._parse_float_or_none(form.value_for_dest("meta_epsilon_start"))
            epsilon_end = self._parse_float_or_none(form.value_for_dest("meta_epsilon_end"))
            epsilon_decay_steps = self._parse_int_or_none(form.value_for_dest("meta_epsilon_decay_steps"))
        else:
            return

        if epsilon_start is None or epsilon_end is None or epsilon_decay_steps is None:
            return
        if epsilon_decay_steps <= 0:
            return

        self._monitor_epsilon_start = epsilon_start
        self._monitor_epsilon_end = epsilon_end
        self._monitor_epsilon_decay_steps = epsilon_decay_steps
        self._monitor_track_epsilon_eta = True

    def _run_selected(self) -> None:
        if self._process is not None:
            messagebox.showerror("Process Running", "A command is already running.")
            return
        try:
            form = self._current_form()
            command = form.build_command()
        except ValueError as error:
            messagebox.showerror("Invalid Settings", str(error))
            return
        self._configure_monitor_estimates(form=form)
        command = self._prepare_command_for_monitor(command)

        self._append_output(f">>> {_format_command(command)}", tag="command")
        self._run_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._preview_button.configure(state="disabled")
        self._status_text.set("RUNNING")
        self._status_label.configure(style="StatusRun.TLabel")

        self._reader_thread = threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
        )
        self._reader_thread.start()

    def _prepare_command_for_monitor(self, command: list[str]) -> list[str]:
        if len(command) < 3:
            self._clear_monitor_files()
            self._reset_monitor_estimates()
            return command
        is_hybrid_runner = command[1:3] == ["-m", "src.hybrid.runner"]
        if not is_hybrid_runner:
            self._clear_monitor_files()
            self._reset_monitor_estimates()
            return command

        self._clear_monitor_files()
        status_file = self._create_status_file()
        control_file = self._create_control_file()
        self._set_monitor_files(
            status_file=status_file,
            control_file=control_file,
            owns_status_file=True,
            owns_control_file=control_file is not None,
        )
        self._reset_live_monitor_display(training_line="training=starting")
        self._runner_monitor_pid = None
        self._runner_monitor_status.set("runner_session=prepared")

        filtered: list[str] = []
        skip_next = False
        for index, token in enumerate(command):
            if skip_next:
                skip_next = False
                continue
            if token == "--external-status-file":
                skip_next = index + 1 < len(command)
                continue
            if token == "--external-control-file":
                skip_next = index + 1 < len(command)
                continue
            filtered.append(token)

        filtered.extend(["--external-status-file", str(status_file)])
        if control_file is not None:
            filtered.extend(["--external-control-file", str(control_file)])
        return filtered

    def _create_status_file(self) -> Path:
        handle, file_path = tempfile.mkstemp(prefix="868-gui-status-", suffix=".json")
        os.close(handle)
        return Path(file_path)

    def _create_control_file(self) -> Path:
        handle, file_path = tempfile.mkstemp(prefix="868-gui-control-", suffix=".json")
        os.close(handle)
        return Path(file_path)

    def _set_monitor_files(
        self,
        *,
        status_file: Path | None,
        control_file: Path | None,
        owns_status_file: bool,
        owns_control_file: bool,
    ) -> None:
        self._external_status_file = status_file
        self._external_control_file = control_file
        self._owns_external_status_file = status_file is not None and owns_status_file
        self._owns_external_control_file = control_file is not None and owns_control_file
        self._last_runner_monitor_bind_attempt_monotonic = 0.0

        if control_file is not None:
            self._monitor_control_state.set("session=connecting")
            self._set_monitor_controls_enabled(True)
            if owns_control_file:
                try:
                    snapshot = set_external_control_mode(control_file, mode=CONTROL_MODE_AUTO)
                except Exception:
                    pass
                else:
                    self._monitor_control_state.set(
                        f"session={snapshot.mode} advance={snapshot.advance_counter}"
                    )
            else:
                self._refresh_monitor_control_state()
        else:
            self._monitor_control_state.set("session=unavailable")
            self._set_monitor_controls_enabled(False)

    def _reset_live_monitor_display(self, *, training_line: str) -> None:
        self._status_snapshot = ("", "", "", "")
        self._monitor_training_line.set(training_line)
        self._monitor_action_line.set("action=idle")
        self._monitor_reward_line.set("reward=idle")
        self._monitor_next_available_actions_line.set("next_available_actions=unavailable")
        self._update_monitor_keycaps(action_line="", next_available_actions_line="")
        self._hide_reward_tooltip(force=True)
        self._hide_phase_tooltip(force=True)
        self._reward_history.clear()
        self._epsilon_progress_value.set(0.0)
        self._epsilon_progress_text.set("0.0%")
        self._draw_reward_graph()
        for variable in self._monitor_metric_vars.values():
            variable.set("-")

    def _clear_monitor_files(self) -> None:
        if self._external_status_file is not None and self._owns_external_status_file:
            self._external_status_file.unlink(missing_ok=True)
        if self._external_control_file is not None and self._owns_external_control_file:
            self._external_control_file.unlink(missing_ok=True)
        self._external_status_file = None
        self._external_control_file = None
        self._owns_external_status_file = False
        self._owns_external_control_file = False
        self._runner_monitor_pid = None
        self._last_runner_monitor_bind_attempt_monotonic = 0.0
        self._runner_monitor_status.set("runner_session=idle")
        self._monitor_control_state.set("session=idle")
        self._set_monitor_controls_enabled(False)
        self._hide_reward_tooltip(force=True)
        self._hide_phase_tooltip(force=True)

    def _ensure_runner_monitor_session_bound(self) -> None:
        status_file_binding = _load_live_monitor_binding_from_status_file(
            status_file=self._external_status_file,
            fallback_control_file=self._external_control_file,
        )
        if status_file_binding is not None:
            self._runner_monitor_pid = status_file_binding.runner_pid
            self._runner_monitor_status.set(
                "runner_session=attached pid={pid} module={module} via=status".format(
                    pid=status_file_binding.runner_pid,
                    module=status_file_binding.runner_module.removeprefix("src."),
                )
            )
            if (
                self._external_status_file == status_file_binding.status_file
                and self._external_control_file == status_file_binding.control_file
            ):
                return
            self._set_monitor_files(
                status_file=status_file_binding.status_file,
                control_file=status_file_binding.control_file,
                owns_status_file=False,
                owns_control_file=False,
            )
            self._reset_live_monitor_display(training_line="training=attaching")
            return

        current_process = self._process
        if (
            current_process is not None
            and current_process.poll() is None
            and self._external_status_file is not None
            and self._owns_external_status_file
        ):
            self._runner_monitor_pid = int(current_process.pid)
            self._runner_monitor_status.set(f"runner_session=attached pid={current_process.pid} via=gui")
            return

        self._runner_monitor_pid = None
        self._runner_monitor_status.set("runner_session=not_bound")

    def _poll_external_status(self) -> None:
        now = time.monotonic()
        if (
            self._runner_monitor_pid is None
            and now - self._last_runner_monitor_bind_attempt_monotonic >= 1.0
        ):
            self._last_runner_monitor_bind_attempt_monotonic = now
            self._ensure_runner_monitor_session_bound()

        status_file = self._external_status_file
        if status_file is not None and status_file.exists():
            try:
                payload = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                binding = _load_live_monitor_binding_from_status_file(
                    status_file=status_file,
                    fallback_control_file=self._external_control_file,
                )
                if binding is not None:
                    self._runner_monitor_pid = binding.runner_pid
                    self._runner_monitor_status.set(
                        "runner_session=attached pid={pid} module={module} via=status".format(
                            pid=binding.runner_pid,
                            module=binding.runner_module.removeprefix("src."),
                        )
                    )
                training_line = str(payload.get("training_line", ""))
                action_line = str(payload.get("action_line", ""))
                reward_line = str(payload.get("reward_line", ""))
                next_available_actions_line = str(payload.get("next_available_actions_line", ""))
                snapshot = (training_line, action_line, reward_line, next_available_actions_line)
                if snapshot != self._status_snapshot:
                    self._status_snapshot = snapshot
                    self._monitor_training_line.set(training_line or "training=idle")
                    self._monitor_action_line.set(action_line or "action=idle")
                    self._monitor_reward_line.set(reward_line or "reward=unavailable")
                    self._monitor_next_available_actions_line.set(
                        next_available_actions_line or "next_available_actions=unavailable"
                    )
                    self._update_monitor_keycaps(
                        action_line=action_line,
                        next_available_actions_line=next_available_actions_line,
                    )
                    self._refresh_reward_tooltip_contents()
                    self._refresh_phase_tooltip_contents()
                    self._update_monitor_metrics(
                        training_line,
                        action_line=action_line,
                        reward_line=reward_line,
                    )
        self._refresh_monitor_control_state()
        self.after(200, self._poll_external_status)

    def _poll_state_monitor(self) -> None:
        try:
            now = time.monotonic()
            if now - self._state_last_poll_monotonic < self._state_poll_interval_seconds:
                return
            self._state_last_poll_monotonic = now
            if self._state_monitor is None:
                return
            selected_tab = self._notebook.select()
            current_widget = self.nametowidget(selected_tab)
            state_selected = self._state_tab_widget is not None and current_widget == self._state_tab_widget
            ascii_selected = (
                self._ascii_maps_tab_widget is not None
                and current_widget == self._ascii_maps_tab_widget
            )
            live_selected = (
                self._live_monitor_tab_widget is not None
                and current_widget == self._live_monitor_tab_widget
            )
            if not state_selected and not ascii_selected and not live_selected:
                return
            self._refresh_state_snapshot()
        finally:
            self.after(300, self._poll_state_monitor)

    def _resolve_monitor_executable_name(self) -> str:
        try:
            form = self._current_form()
            value = str(form.value_for_dest("exe") or "").strip()
            if value:
                return value
        except Exception:
            pass
        return "868-HACK.exe"

    def _ensure_state_monitor_started(self) -> None:
        executable_name = self._resolve_monitor_executable_name()
        self._ensure_runner_monitor_session_bound()
        if self._state_monitor is not None:
            return
        try:
            monitor = MemoryStateMonitor(
                executable_name=executable_name,
                config_path=None,
                fields_filter="",
                resolve_each_poll=False,
            )
            monitor.start()
        except Exception as error:
            self._state_monitor_error.set(str(error))
            self._state_monitor_status.set("state_monitor=error")
            self._state_monitor_pid.set("pid=-")
            return
        self._state_monitor = monitor
        self._state_monitor_error.set("")
        self._state_monitor_status.set(f"state_monitor=attached exe={executable_name}")
        self._state_monitor_pid.set(f"pid={monitor.attached.pid}")
        self._refresh_state_snapshot()

    def _stop_state_monitor(self) -> None:
        if self._state_monitor is not None:
            try:
                self._state_monitor.stop()
            except Exception:
                pass
        self._state_monitor = None
        self._state_monitor_status.set("state_monitor=stopped")
        self._state_monitor_pid.set("pid=-")
        self._state_monitor_timestamp.set("last_snapshot=-")
        self._state_monitor_error.set("")
        if self._state_fields_tree is not None:
            for item_id in self._state_fields_tree.get_children():
                self._state_fields_tree.delete(item_id)
        self._set_ascii_maps_text("board snapshot unavailable")

    def _insert_textual_markup(self, widget: tk.Text, text: str) -> None:
        cursor = 0
        for match in _TEXTUAL_MARKUP_SEGMENT_PATTERN.finditer(text):
            if match.start() > cursor:
                widget.insert("end", text[cursor:match.start()])
            style_name = str(match.group(1)).strip().lower()
            segment = str(match.group(2))
            tag_name = f"textual_{style_name}"
            if tag_name in widget.tag_names():
                widget.insert("end", segment, tag_name)
            else:
                widget.insert("end", segment)
            cursor = match.end()
        if cursor < len(text):
            widget.insert("end", text[cursor:])

    def _set_ascii_maps_text(self, text: str) -> None:
        widget = self._ascii_maps_text
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        self._insert_textual_markup(widget, text)
        widget.configure(state="disabled")

    def _refresh_state_snapshot(self) -> None:
        monitor = self._state_monitor
        if monitor is None:
            return
        try:
            snapshot = monitor.poll()
        except Exception as error:
            self._state_monitor_error.set(str(error))
            self._state_monitor_status.set("state_monitor=poll_error")
            return
        self._state_monitor_error.set("")
        self._state_monitor_status.set("state_monitor=ok")
        self._state_monitor_timestamp.set(f"last_snapshot={snapshot.timestamp}")
        self._render_state_snapshot(snapshot=snapshot)

    def _render_state_snapshot(self, *, snapshot: PollSnapshot) -> None:
        if self._state_fields_tree is not None:
            for item_id in self._state_fields_tree.get_children():
                self._state_fields_tree.delete(item_id)
            for row in snapshot.fields:
                self._state_fields_tree.insert(
                    "",
                    "end",
                    values=(
                        row.name,
                        row.data_type,
                        row.confidence,
                        row.address,
                        row.value,
                        row.status,
                        row.error,
                    ),
                )
        if self._ascii_maps_text is not None:
            self._set_ascii_maps_text(snapshot.board_stats or "board snapshot unavailable")

    def _build_state_tab(self) -> None:
        state_tab = ttk.Frame(self._notebook, padding=(10, 10, 10, 10), style="Surface.TFrame")
        state_tab.columnconfigure(0, weight=1)
        state_tab.rowconfigure(3, weight=1)

        ttk.Label(state_tab, text="Game State Inspector", style="SectionLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )

        controls = ttk.Frame(state_tab, style="Surface.TFrame")
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        controls.columnconfigure(0, weight=0)
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(2, weight=1)
        ttk.Button(
            controls,
            text="Attach",
            command=self._ensure_state_monitor_started,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Detach",
            command=self._stop_state_monitor,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            controls,
            textvariable=self._state_monitor_status,
            style="FormHelp.TLabel",
        ).grid(row=0, column=2, sticky="e")

        meta = ttk.Frame(state_tab, style="Surface.TFrame")
        meta.grid(row=2, column=0, sticky="ew")
        meta.columnconfigure(0, weight=1)
        meta.columnconfigure(1, weight=1)
        meta.columnconfigure(2, weight=1)
        ttk.Label(meta, textvariable=self._state_monitor_pid, style="FormHelp.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(meta, textvariable=self._state_monitor_timestamp, style="FormHelp.TLabel").grid(
            row=0,
            column=1,
            sticky="w",
        )
        ttk.Label(meta, textvariable=self._state_monitor_error, style="FormHelp.TLabel").grid(
            row=0,
            column=2,
            sticky="e",
        )

        table_frame = ttk.Frame(state_tab, style="Surface.TFrame")
        table_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("field", "type", "confidence", "address", "value", "status", "error")
        tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="State.Treeview",
        )
        tree.heading("field", text="Field")
        tree.heading("type", text="Type")
        tree.heading("confidence", text="Confidence")
        tree.heading("address", text="Address")
        tree.heading("value", text="Value")
        tree.heading("status", text="Status")
        tree.heading("error", text="Error")
        tree.column("field", width=180, anchor="w")
        tree.column("type", width=90, anchor="w")
        tree.column("confidence", width=90, anchor="w")
        tree.column("address", width=130, anchor="w")
        tree.column("value", width=280, anchor="w")
        tree.column("status", width=100, anchor="w")
        tree.column("error", width=240, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self._state_fields_tree = tree

        self._state_tab_widget = state_tab

        self._notebook.add(state_tab, text="Game State")

    def _build_ascii_maps_tab(self) -> None:
        maps_tab = ttk.Frame(self._notebook, padding=(10, 10, 10, 10), style="Surface.TFrame")
        maps_tab.columnconfigure(0, weight=1)
        maps_tab.rowconfigure(2, weight=1)

        ttk.Label(maps_tab, text="ASCII Maps", style="SectionLabel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )

        meta = ttk.Frame(maps_tab, style="Surface.TFrame")
        meta.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        meta.columnconfigure(0, weight=0)
        meta.columnconfigure(1, weight=0)
        meta.columnconfigure(2, weight=1)
        meta.columnconfigure(3, weight=1)
        meta.columnconfigure(4, weight=1)
        ttk.Button(
            meta,
            text="Attach",
            command=self._ensure_state_monitor_started,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            meta,
            text="Detach",
            command=self._stop_state_monitor,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Label(meta, textvariable=self._state_monitor_status, style="FormHelp.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
        )
        ttk.Label(meta, textvariable=self._state_monitor_pid, style="FormHelp.TLabel").grid(
            row=0,
            column=3,
            sticky="w",
        )
        ttk.Label(meta, textvariable=self._state_monitor_timestamp, style="FormHelp.TLabel").grid(
            row=0,
            column=4,
            sticky="e",
        )

        maps_text = tk.Text(
            maps_tab,
            wrap="none",
            bg=_PALETTE["terminal_bg"],
            fg=_PALETTE["terminal_fg"],
            insertbackground=_PALETTE["terminal_fg"],
            relief="flat",
            padx=8,
            pady=8,
            font=_FONTS["mono"],
        )
        maps_text.grid(row=2, column=0, sticky="nsew")
        maps_scroll_y = ttk.Scrollbar(maps_tab, orient="vertical", command=maps_text.yview)
        maps_scroll_y.grid(row=2, column=1, sticky="ns")
        maps_scroll_x = ttk.Scrollbar(maps_tab, orient="horizontal", command=maps_text.xview)
        maps_scroll_x.grid(row=3, column=0, sticky="ew")
        maps_text.configure(yscrollcommand=maps_scroll_y.set, xscrollcommand=maps_scroll_x.set)
        for style_name, color in _TEXTUAL_STYLE_COLORS.items():
            maps_text.tag_configure(f"textual_{style_name}", foreground=color)
        maps_text.insert("1.0", "board snapshot unavailable")
        maps_text.configure(state="disabled")

        self._ascii_maps_text = maps_text
        self._ascii_maps_tab_widget = maps_tab
        self._notebook.add(maps_tab, text="ASCII Maps")

    def _set_monitor_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (
            self._monitor_pause_button,
            self._monitor_step_button,
            self._monitor_resume_button,
        ):
            if button is not None:
                button.configure(state=state)

    def _update_monitor_keycaps(self, *, action_line: str, next_available_actions_line: str) -> None:
        available_actions = _parse_next_available_actions(next_available_actions_line)
        self._monitor_available_key_labels = {
            key_label
            for action in available_actions
            for key_label in (_monitor_key_label_for_action(action),)
            if key_label is not None
        }
        current_action = _parse_status_values(action_line).get("action", "")
        self._monitor_selected_key_label = _monitor_key_label_for_action(current_action or "")

        for key_label, widget in self._monitor_keycap_labels.items():
            is_selected = key_label == self._monitor_selected_key_label
            is_available = key_label in self._monitor_available_key_labels
            if is_selected:
                background = _PALETTE["accent_alt"]
                foreground = _PALETTE["bg"]
                border = "#ffe0a3"
            elif is_available:
                background = "#18342f"
                foreground = "#a9f5e7"
                border = "#2d7b70"
            else:
                background = "#121a23"
                foreground = "#5f6e80"
                border = "#23303d"
            widget.configure(
                bg=background,
                fg=foreground,
                highlightbackground=border,
                highlightcolor=border,
                highlightthickness=1,
            )

    def _refresh_reward_tooltip_contents(self) -> None:
        if self._reward_tooltip_label is None:
            return
        self._reward_tooltip_label.configure(
            text=_format_reward_breakdown_tooltip(self._monitor_reward_line.get())
        )

    def _refresh_phase_tooltip_contents(self) -> None:
        if self._phase_tooltip_label is None:
            return
        self._phase_tooltip_label.configure(
            text=_format_phase_breakdown_tooltip(self._monitor_action_line.get())
        )

    def _create_monitor_tooltip(self, text: str) -> tuple[tk.Toplevel, tk.Label]:
        tooltip = tk.Toplevel(self)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        try:
            tooltip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(
            tooltip,
            text=text,
            justify="left",
            anchor="w",
            bg="#101821",
            fg=_PALETTE["text"],
            relief="solid",
            bd=1,
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        label.pack()
        return (tooltip, label)

    def _show_reward_tooltip(self, event: tk.Event[tk.Misc] | None = None) -> None:
        tooltip_text = _format_reward_breakdown_tooltip(self._monitor_reward_line.get())
        if self._reward_tooltip is None or not self._reward_tooltip.winfo_exists():
            self._reward_tooltip, self._reward_tooltip_label = self._create_monitor_tooltip(tooltip_text)
        else:
            self._refresh_reward_tooltip_contents()
        self._hide_phase_tooltip(force=True)
        self._move_reward_tooltip(event)
        if self._reward_tooltip is not None:
            self._reward_tooltip.deiconify()

    def _move_reward_tooltip(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if self._reward_tooltip is None or not self._reward_tooltip.winfo_exists():
            return
        pointer_x = self.winfo_pointerx() + 16
        pointer_y = self.winfo_pointery() + 18
        self._reward_tooltip.geometry(f"+{pointer_x}+{pointer_y}")

    def _show_phase_tooltip(self, event: tk.Event[tk.Misc] | None = None) -> None:
        tooltip_text = _format_phase_breakdown_tooltip(self._monitor_action_line.get())
        if self._phase_tooltip is None or not self._phase_tooltip.winfo_exists():
            self._phase_tooltip, self._phase_tooltip_label = self._create_monitor_tooltip(tooltip_text)
        else:
            self._refresh_phase_tooltip_contents()
        self._hide_reward_tooltip(force=True)
        self._move_phase_tooltip(event)
        if self._phase_tooltip is not None:
            self._phase_tooltip.deiconify()

    def _move_phase_tooltip(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if self._phase_tooltip is None or not self._phase_tooltip.winfo_exists():
            return
        pointer_x = self.winfo_pointerx() + 16
        pointer_y = self.winfo_pointery() + 18
        self._phase_tooltip.geometry(f"+{pointer_x}+{pointer_y}")

    def _hide_reward_tooltip(
        self,
        _event: tk.Event[tk.Misc] | None = None,
        *,
        force: bool = False,
    ) -> None:
        hovered_widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if not force and hovered_widget in self._reward_hover_widgets:
            return
        if self._reward_tooltip is not None and self._reward_tooltip.winfo_exists():
            self._reward_tooltip.destroy()
        self._reward_tooltip = None
        self._reward_tooltip_label = None

    def _hide_phase_tooltip(
        self,
        _event: tk.Event[tk.Misc] | None = None,
        *,
        force: bool = False,
    ) -> None:
        hovered_widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if not force and hovered_widget in self._phase_hover_widgets:
            return
        if self._phase_tooltip is not None and self._phase_tooltip.winfo_exists():
            self._phase_tooltip.destroy()
        self._phase_tooltip = None
        self._phase_tooltip_label = None

    def _refresh_monitor_control_state(self) -> None:
        control_file = self._external_control_file
        if control_file is None or not control_file.exists():
            return
        snapshot = load_external_control_snapshot(control_file)
        self._monitor_control_state.set(
            f"session={snapshot.mode} advance={snapshot.advance_counter}"
        )

    def _pause_monitor_session(self) -> None:
        control_file = self._external_control_file
        if control_file is None:
            return
        snapshot = set_external_control_mode(control_file, mode=CONTROL_MODE_PAUSED)
        self._monitor_control_state.set(
            f"session={snapshot.mode} advance={snapshot.advance_counter}"
        )

    def _step_monitor_session(self) -> None:
        control_file = self._external_control_file
        if control_file is None:
            return
        snapshot = step_external_control(control_file)
        self._monitor_control_state.set(
            f"session={snapshot.mode} advance={snapshot.advance_counter}"
        )

    def _resume_monitor_session(self) -> None:
        control_file = self._external_control_file
        if control_file is None:
            return
        snapshot = set_external_control_mode(control_file, mode=CONTROL_MODE_AUTO)
        self._monitor_control_state.set(
            f"session={snapshot.mode} advance={snapshot.advance_counter}"
        )

    def _update_monitor_metrics(
        self,
        training_line: str,
        *,
        action_line: str = "",
        reward_line: str = "",
    ) -> None:
        status_values = _parse_status_values(training_line)
        action_values = _monitor_action_card_values(action_line)
        now = time.monotonic()
        new_step_event = False

        current_episode, reported_total_episodes = _parse_episode_progress(
            status_values.get("episode", ""),
            fallback_total=self._monitor_total_episodes,
        )
        if reported_total_episodes is not None and reported_total_episodes >= 1:
            self._monitor_total_episodes = reported_total_episodes
        current_step = _parse_step_value(status_values.get("step", ""))
        is_done = _parse_bool_value(status_values.get("done", ""))

        if current_episode is not None and current_step is not None:
            step_key = (current_episode, current_step)
            if step_key != self._monitor_last_step_key:
                if self._monitor_first_step_time is None:
                    self._monitor_first_step_time = now
                self._monitor_step_events += 1
                self._monitor_last_step_key = step_key
                new_step_event = True
            self._monitor_episode_start_times.setdefault(current_episode, now)

        if (
            is_done is True
            and current_episode is not None
            and current_episode not in self._monitor_completed_episode_ids
        ):
            start_time = self._monitor_episode_start_times.pop(current_episode, None)
            if start_time is not None:
                duration_seconds = max(0.0, now - start_time)
                self._monitor_episode_duration_sum += duration_seconds
                self._monitor_episode_duration_count += 1
            self._monitor_completed_episode_ids.add(current_episode)

        seconds_per_step: float | None = None
        if self._monitor_first_step_time is not None and self._monitor_step_events > 0:
            elapsed = max(0.0, now - self._monitor_first_step_time)
            if elapsed > 0.0:
                seconds_per_step = elapsed / float(self._monitor_step_events)

        epsilon_value: float | None = None
        epsilon_text = status_values.get("epsilon")
        if epsilon_text is not None:
            try:
                epsilon_value = float(epsilon_text)
            except (TypeError, ValueError):
                epsilon_value = None
        eta_epsilon_seconds = (
            _estimate_epsilon_eta_seconds(
                current_epsilon=epsilon_value,
                epsilon_start=self._monitor_epsilon_start,
                epsilon_end=self._monitor_epsilon_end,
                epsilon_decay_steps=self._monitor_epsilon_decay_steps,
                seconds_per_step=seconds_per_step,
            )
            if self._monitor_track_epsilon_eta
            else None
        )

        eta_session_seconds: float | None = None
        if self._monitor_episode_duration_count > 0 and self._monitor_total_episodes is not None:
            average_episode_seconds = (
                self._monitor_episode_duration_sum / float(self._monitor_episode_duration_count)
            )
            remaining_episodes = max(
                self._monitor_total_episodes - len(self._monitor_completed_episode_ids),
                0,
            )
            eta_session_seconds = average_episode_seconds * float(remaining_episodes)

        previous_reward_value = self._monitor_metric_vars.get("reward")
        reward_value = _resolve_reward_metric_value(
            training_line=training_line,
            reward_line=reward_line,
            previous_value=(
                previous_reward_value.get() if previous_reward_value is not None else "-"
            ),
        )
        mapped_values = {
            "episode": status_values.get("episode", "-"),
            "step": status_values.get("step", "-"),
            "reward": reward_value,
            "phase": action_values.get("phase", "-"),
            "total": status_values.get("total", "-"),
            "epsilon": status_values.get("epsilon", "-"),
            "threat_epsilon": status_values.get("threat_epsilon", "-"),
            "updates": status_values.get("updates", "-"),
            "done": status_values.get("done", "-"),
            "terminal": status_values.get("terminal", "-"),
            "eta_epsilon": _format_duration_seconds(eta_epsilon_seconds),
            "eta_session": _format_duration_seconds(eta_session_seconds),
        }
        for key, value in mapped_values.items():
            variable = self._monitor_metric_vars.get(key)
            if variable is not None:
                variable.set(value)

        if epsilon_value is None:
            self._epsilon_progress_value.set(0.0)
            self._epsilon_progress_text.set(
                _format_epsilon_progress_text(
                    current_epsilon=None,
                    epsilon_end=self._monitor_epsilon_end,
                )
            )
        else:
            clamped_epsilon = max(0.0, min(1.0, epsilon_value))
            self._epsilon_progress_value.set(clamped_epsilon * 100.0)
            self._epsilon_progress_text.set(
                _format_epsilon_progress_text(
                    current_epsilon=clamped_epsilon,
                    epsilon_end=self._monitor_epsilon_end,
                )
            )

        reward_step_key: tuple[int, int] | None = None
        if current_episode is not None and current_step is not None:
            reward_step_key = (current_episode, current_step)

        should_record_reward = reward_step_key is None or (
            new_step_event and reward_step_key != self._monitor_last_reward_step_key
        )
        if should_record_reward:
            try:
                reward_sample = float(reward_value)
            except (TypeError, ValueError):
                reward_sample = None
            if reward_sample is not None:
                self._reward_history.append(reward_sample)
                if len(self._reward_history) > _REWARD_HISTORY_LIMIT:
                    self._reward_history = self._reward_history[-_REWARD_HISTORY_LIMIT:]
                if reward_step_key is not None:
                    self._monitor_last_reward_step_key = reward_step_key
        self._draw_reward_graph()

    def _draw_reward_graph(self) -> None:
        canvas = self._reward_canvas
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        canvas.delete("all")

        left = 28
        right = width - 10
        top = 10
        bottom = height - 18
        canvas.create_rectangle(left, top, right, bottom, outline="#263546")
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + int((bottom - top) * ratio)
            canvas.create_line(left, y, right, y, fill="#172231")
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
            x = left + int((right - left) * ratio)
            canvas.create_line(x, top, x, bottom, fill="#111a24")
        canvas.create_text(
            left,
            top - 2,
            text=f"reward pulse (last {_REWARD_HISTORY_LIMIT} steps)",
            fill="#7cf5a6",
            anchor="sw",
            font=("Segoe UI", 8, "bold"),
        )

        points = self._reward_history
        if not points:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="waiting for reward data",
                fill=_PALETTE["muted"],
                font=_FONTS["small"],
            )
            return

        min_reward = min(min(points), 0.0)
        max_reward = max(max(points), 0.0)
        if math.isclose(min_reward, max_reward):
            padding = max(1.0, abs(max_reward) * 0.1)
            min_reward -= padding
            max_reward += padding
        reward_span = max(max_reward - min_reward, 1e-9)

        canvas.create_text(
            left - 8,
            top,
            text=f"{max_reward:.2f}",
            fill=_PALETTE["muted"],
            anchor="e",
            font=_FONTS["small"],
        )
        canvas.create_text(
            left - 8,
            bottom,
            text=f"{min_reward:.2f}",
            fill=_PALETTE["muted"],
            anchor="e",
            font=_FONTS["small"],
        )
        zero_ratio = (0.0 - min_reward) / reward_span
        zero_y = bottom - ((bottom - top) * zero_ratio)
        canvas.create_line(left, zero_y, right, zero_y, fill="#29445a", dash=(3, 3))

        point_count = len(points)
        if point_count == 1:
            x = (left + right) / 2
            y_ratio = (points[0] - min_reward) / reward_span
            y = bottom - ((bottom - top) * y_ratio)
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#7cf5a6", outline="")
            return

        coordinates: list[float] = []
        for index, value in enumerate(points):
            x = left + ((right - left) * (index / float(point_count - 1)))
            y_ratio = (value - min_reward) / reward_span
            y = bottom - ((bottom - top) * y_ratio)
            coordinates.extend((x, y))

        canvas.create_line(*coordinates, fill="#0f4022", width=6, capstyle="round", joinstyle="round")
        canvas.create_line(*coordinates, fill="#7cf5a6", width=2, capstyle="round", joinstyle="round")
        latest_x = coordinates[-2]
        latest_y = coordinates[-1]
        canvas.create_oval(
            latest_x - 4,
            latest_y - 4,
            latest_x + 4,
            latest_y + 4,
            fill="#d6ffe5",
            outline="#7cf5a6",
        )

    def _run_process(self, command: list[str]) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(_REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._event_queue.put(("process_started", int(self._process.pid)))
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._event_queue.put(("line", line.rstrip("\n")))
            return_code = self._process.wait()
            self._event_queue.put(("done", return_code))
        except OSError as error:
            self._event_queue.put(("line", f"Failed to launch command: {error}"))
            self._event_queue.put(("done", 1))
        finally:
            self._process = None

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        process.terminate()
        self._event_queue.put(("line", "Process termination requested."))
        self._status_text.set("STOPPING")
        self._status_label.configure(style="StatusStop.TLabel")

    def _append_output(self, text: str, *, tag: str | None = None) -> None:
        if tag is None:
            self._output.insert("end", f"{text}\n")
        else:
            self._output.insert("end", f"{text}\n", tag)
        self._output.see("end")

    def _animate_status(self) -> None:
        if self._process is not None:
            running_suffix = "." * ((self._status_phase % 3) + 1)
            self._status_text.set(f"RUNNING{running_suffix}")
            self._status_phase += 1
        elif self._status_text.get().startswith("RUNNING"):
            self._status_text.set("READY")
            self._status_label.configure(style="StatusReady.TLabel")
        self.after(250, self._animate_status)

    def _drain_event_queue(self) -> None:
        while True:
            try:
                kind, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                line = str(payload)
                tag = "error" if "error" in line.lower() or "failed" in line.lower() else None
                self._append_output(line, tag=tag)
            elif kind == "process_started":
                pid = int(payload)
                self._runner_monitor_pid = pid
                self._runner_monitor_status.set(f"runner_session=attached pid={pid} via=gui")
            elif kind == "done":
                code = int(payload)
                self._append_output(f"[process exited with code {code}]", tag="system")
                self._run_button.configure(state="normal")
                self._stop_button.configure(state="disabled")
                self._preview_button.configure(state="normal")
                self._status_text.set("READY" if code == 0 else "FAILED")
                self._status_label.configure(
                    style="StatusReady.TLabel" if code == 0 else "StatusStop.TLabel"
                )
                self._clear_monitor_files()
                self._refresh_preview()
        self.after(100, self._drain_event_queue)

    def _on_close(self) -> None:
        if self._process is not None:
            self._stop_process()
        self._stop_state_monitor()
        self._clear_monitor_files()
        self._hide_reward_tooltip(force=True)
        self._hide_phase_tooltip(force=True)
        self.destroy()


def main() -> None:
    app = HybridRunnerGui()
    app.mainloop()


if __name__ == "__main__":
    main()

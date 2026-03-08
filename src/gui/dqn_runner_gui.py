"""Simple Tkinter launcher for DQN run/evaluation workflows."""

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
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from src.env import dqn_policy_runner
from src.training import evaluate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATH_LIKE_DESTS = {"exe", "checkpoint", "checkpoint_a", "checkpoint_b", "json_out"}
_HIDDEN_GUI_DESTS = {"external_status_file"}
_MAX_FORM_COLUMNS = 5
_CHECKPOINT_DIR = _REPO_ROOT / "artifacts" / "checkpoints"
_STATUS_KV_PATTERN = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")
_EPSILON_HISTORY_LIMIT = 240
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
    if _is_numeric_action(action) or _is_boolean_optional(action) or action.choices is not None:
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
    if current_value:
        current_path = Path(current_value)
        if current_path.suffix:
            return current_path.parent
        return current_path
    if dest in {"checkpoint", "checkpoint_a", "checkpoint_b", "json_out"}:
        return _CHECKPOINT_DIR
    return _REPO_ROOT


def _run_dqn_preset_overrides() -> dict[str, dict[str, object]]:
    return {
        "defaults": {},
        "reward survival": {
            "reward_survival": 0.25,
            "reward_step_penalty": 0.005,
            "reward_fail_penalty": 3.0,
            "reward_danger_tile_penalty": 0.15,
            "reward_safe_tile_bonus": 0.05,
        },
        "reward exploration": {
            "reward_survival": 0.05,
            "reward_step_penalty": 0.003,
            "reward_currency_delta": 0.03,
            "reward_score_delta": 0.003,
            "reward_resource_proximity": 0.08,
            "reward_points_collected": 0.004,
            "reward_phase_progress": 0.2,
        },
    }


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


def _parse_status_values(line: str) -> dict[str, str]:
    return {key: value for key, value in _STATUS_KV_PATTERN.findall(line)}


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
            if _is_boolean_optional(action):
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

    def _apply_selected_preset(self) -> None:
        preset_name = self._selected_preset.get()
        overrides = self._presets.get(preset_name)
        if overrides is None:
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

        for dest, value in overrides.items():
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
        if dest == "exe":
            path = filedialog.askopenfilename(
                parent=self,
                title="Select executable",
                initialdir=initial_dir,
                filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            )
        elif dest in {"checkpoint_a", "checkpoint_b"}:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select checkpoint",
                initialdir=initial_dir,
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
        elif dest == "checkpoint" and self._profile_id != "run-dqn":
            path = filedialog.askopenfilename(
                parent=self,
                title="Select checkpoint",
                initialdir=initial_dir,
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
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
            if _is_boolean_optional(action):
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


class DqnRunnerGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("868 DQN Launcher")
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
        self._status_snapshot: tuple[str, str] = ("", "")
        self._monitor_training_line = tk.StringVar(value="training=idle")
        self._monitor_action_line = tk.StringVar(value="action=idle")
        self._monitor_metric_vars: dict[str, tk.StringVar] = {}
        self._epsilon_history: list[float] = []

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
            text="Train, evaluate, compare, and launch from one surface with every CLI flag exposed.",
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
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=0)

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
        preview_entry.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))

        self._preview_button = ttk.Button(
            controls,
            text="Preview Command",
            command=self._refresh_preview,
            style="Secondary.TButton",
        )
        self._preview_button.grid(row=2, column=0, sticky="w")
        self._run_button = ttk.Button(
            controls,
            text="Launch",
            command=self._run_selected,
            style="Primary.TButton",
        )
        self._run_button.grid(row=2, column=1)
        self._stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self._stop_process,
            state="disabled",
            style="Danger.TButton",
        )
        self._stop_button.grid(row=2, column=2, sticky="e")

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

        style.configure("Primary.TButton", background="#1f8f81", foreground="#ffffff", borderwidth=0, padding=(10, 5))
        style.map("Primary.TButton", background=[("active", "#28a594"), ("disabled", "#345955")])
        style.configure("Secondary.TButton", background="#2a3a4d", foreground=_PALETTE["text"], borderwidth=0, padding=(8, 5))
        style.map("Secondary.TButton", background=[("active", "#35485f")])
        style.configure("Danger.TButton", background="#5a2630", foreground="#ffdce1", borderwidth=0, padding=(8, 5))
        style.map("Danger.TButton", background=[("active", "#70313d"), ("disabled", "#3f2730")])

    def _build_profiles(self) -> None:
        run_parser = dqn_policy_runner._build_parser()
        evaluate_parser = evaluate._build_parser()
        eval_run_parser = _get_subparser(evaluate_parser, command_name="run")
        eval_compare_parser = _get_subparser(evaluate_parser, command_name="compare")

        profiles = (
            (
                "run-dqn",
                "DQN Run (train/eval)",
                run_parser,
                ("-m", "src.env.dqn_policy_runner"),
                _run_dqn_preset_overrides(),
            ),
            (
                "eval-run",
                "Evaluate Run",
                eval_run_parser,
                ("-m", "src.training.evaluate", "run"),
                None,
            ),
            (
                "eval-compare",
                "Evaluate Compare",
                eval_compare_parser,
                ("-m", "src.training.evaluate", "compare"),
                None,
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
        self._refresh_tab_visuals()

    def _build_monitor_tab(self) -> None:
        monitor = ttk.Frame(self._notebook, padding=(10, 10, 10, 10), style="Surface.TFrame")
        monitor.columnconfigure(0, weight=1)
        monitor.rowconfigure(1, weight=0)
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
            ("Total", "total"),
            ("Epsilon", "epsilon"),
            ("Updates", "updates"),
            ("Done", "done"),
            ("Terminal", "terminal"),
        )
        for idx, (label, key) in enumerate(metric_keys):
            row = idx // 4
            col = idx % 4
            card = ttk.Frame(metric_grid, style="MetricCard.TFrame", padding=(8, 6, 8, 6))
            card.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
            self._monitor_metric_vars[key] = tk.StringVar(value="-")
            ttk.Label(card, text=label, style="MetricName.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=self._monitor_metric_vars[key], style="MetricValue.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
            )

        ttk.Label(monitor, text="training_line", style="FormLabel.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(
            monitor,
            textvariable=self._monitor_training_line,
            style="FormHelp.TLabel",
        ).grid(row=2, column=0, sticky="e", padx=(120, 0))

        graph_shell = ttk.Frame(monitor, style="Surface.TFrame")
        graph_shell.grid(row=3, column=0, sticky="nsew", pady=(20, 0))
        graph_shell.columnconfigure(0, weight=1)
        graph_shell.rowconfigure(1, weight=1)
        ttk.Label(graph_shell, textvariable=self._monitor_action_line, style="FormHelp.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self._epsilon_canvas = tk.Canvas(
            graph_shell,
            height=180,
            bg="#0d141d",
            highlightthickness=1,
            highlightbackground="#243244",
        )
        self._epsilon_canvas.grid(row=1, column=0, sticky="nsew")
        self._epsilon_canvas.bind("<Configure>", lambda _: self._draw_epsilon_graph())

        self._notebook.add(monitor, text="Live Monitor")

    def _refresh_tab_visuals(self) -> None:
        selected = self._notebook.select()
        for tab_id in self._notebook.tabs():
            self._notebook.tab(tab_id, padding=(18, 10) if tab_id == selected else (10, 5))

    def _on_tab_changed(self) -> None:
        self._refresh_tab_visuals()
        self._refresh_preview()

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

    def _run_selected(self) -> None:
        if self._process is not None:
            messagebox.showerror("Process Running", "A command is already running.")
            return
        try:
            command = self._current_form().build_command()
        except ValueError as error:
            messagebox.showerror("Invalid Settings", str(error))
            return
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
            self._external_status_file = None
            return command
        is_dqn_runner = command[1:3] == ["-m", "src.env.dqn_policy_runner"]
        is_eval_compare = command[1:4] == ["-m", "src.training.evaluate", "compare"]
        if not is_dqn_runner and not is_eval_compare:
            self._external_status_file = None
            return command

        previous_status_file = self._external_status_file
        if previous_status_file is not None:
            previous_status_file.unlink(missing_ok=True)
        status_file = self._create_status_file()
        self._external_status_file = status_file
        self._status_snapshot = ("", "")
        self._monitor_training_line.set("training=starting")
        self._monitor_action_line.set("action=idle reason=launch")
        self._epsilon_history.clear()
        for variable in self._monitor_metric_vars.values():
            variable.set("-")

        filtered: list[str] = []
        skip_next = False
        for index, token in enumerate(command):
            if skip_next:
                skip_next = False
                continue
            if token == "--external-status-file":
                skip_next = index + 1 < len(command)
                continue
            if token in {"--tui", "--no-tui"}:
                continue
            filtered.append(token)

        if is_dqn_runner or is_eval_compare:
            filtered.append("--no-tui")
        filtered.extend(["--external-status-file", str(status_file)])
        return filtered

    def _create_status_file(self) -> Path:
        handle, file_path = tempfile.mkstemp(prefix="868-gui-status-", suffix=".json")
        os.close(handle)
        return Path(file_path)

    def _poll_external_status(self) -> None:
        status_file = self._external_status_file
        if status_file is not None and status_file.exists():
            try:
                payload = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                training_line = str(payload.get("training_line", ""))
                action_line = str(payload.get("action_line", ""))
                snapshot = (training_line, action_line)
                if snapshot != self._status_snapshot:
                    self._status_snapshot = snapshot
                    self._monitor_training_line.set(training_line or "training=idle")
                    self._monitor_action_line.set(action_line or "action=idle")
                    self._update_monitor_metrics(training_line)
        self.after(200, self._poll_external_status)

    def _update_monitor_metrics(self, training_line: str) -> None:
        status_values = _parse_status_values(training_line)
        mapped_values = {
            "episode": status_values.get("episode", "-"),
            "step": status_values.get("step", "-"),
            "reward": status_values.get("reward", "-"),
            "total": status_values.get("total", "-"),
            "epsilon": status_values.get("epsilon", "-"),
            "updates": status_values.get("updates", "-"),
            "done": status_values.get("done", "-"),
            "terminal": status_values.get("terminal", "-"),
        }
        for key, value in mapped_values.items():
            variable = self._monitor_metric_vars.get(key)
            if variable is not None:
                variable.set(value)

        epsilon_text = mapped_values["epsilon"]
        try:
            epsilon_value = float(epsilon_text)
        except (TypeError, ValueError):
            return
        epsilon_value = max(0.0, min(1.0, epsilon_value))
        self._epsilon_history.append(epsilon_value)
        if len(self._epsilon_history) > _EPSILON_HISTORY_LIMIT:
            self._epsilon_history = self._epsilon_history[-_EPSILON_HISTORY_LIMIT:]
        self._draw_epsilon_graph()

    def _draw_epsilon_graph(self) -> None:
        canvas = self._epsilon_canvas
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        canvas.delete("all")

        left = 28
        right = width - 10
        top = 10
        bottom = height - 18
        canvas.create_rectangle(left, top, right, bottom, outline="#263546")
        for ratio in (0.25, 0.5, 0.75):
            y = top + int((bottom - top) * ratio)
            canvas.create_line(left, y, right, y, fill="#1a2432")

        canvas.create_text(left - 8, top, text="1.0", fill=_PALETTE["muted"], anchor="e", font=_FONTS["small"])
        canvas.create_text(
            left - 8,
            bottom,
            text="0.0",
            fill=_PALETTE["muted"],
            anchor="e",
            font=_FONTS["small"],
        )
        canvas.create_text(
            left,
            top - 2,
            text="epsilon",
            fill=_PALETTE["accent"],
            anchor="sw",
            font=("Segoe UI", 8, "bold"),
        )

        points = self._epsilon_history
        if len(points) < 2:
            return
        span = len(points) - 1
        line_points: list[float] = []
        for index, value in enumerate(points):
            x = left + ((right - left) * index / span)
            y = top + ((bottom - top) * (1.0 - value))
            line_points.extend((x, y))
        canvas.create_line(*line_points, fill=_PALETTE["accent"], width=2, smooth=True)
        canvas.create_oval(
            line_points[-2] - 3,
            line_points[-1] - 3,
            line_points[-2] + 3,
            line_points[-1] + 3,
            fill=_PALETTE["accent_alt"],
            outline="",
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
                if self._external_status_file is not None:
                    self._external_status_file.unlink(missing_ok=True)
                    self._external_status_file = None
                self._refresh_preview()
        self.after(100, self._drain_event_queue)

    def _on_close(self) -> None:
        if self._process is not None:
            self._stop_process()
        if self._external_status_file is not None:
            self._external_status_file.unlink(missing_ok=True)
            self._external_status_file = None
        self.destroy()


def main() -> None:
    app = DqnRunnerGui()
    app.mainloop()


if __name__ == "__main__":
    main()

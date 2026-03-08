"""Simple Tkinter launcher for DQN run/evaluation workflows."""

from __future__ import annotations

import argparse
import math
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.env import dqn_policy_runner
from src.training import evaluate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATH_LIKE_DESTS = {"exe", "checkpoint", "checkpoint_a", "checkpoint_b", "json_out"}
_MAX_FORM_COLUMNS = 5


def _iter_parser_actions(parser: argparse.ArgumentParser) -> tuple[argparse.Action, ...]:
    actions: list[argparse.Action] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            continue
        if not action.option_strings:
            continue
        actions.append(action)
    return tuple(actions)


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
        return 30
    if _is_numeric_action(action):
        return 8
    if action.choices is not None:
        longest_choice = max(len(str(choice)) for choice in action.choices)
        return max(10, min(longest_choice + 3, 18))
    return 16


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
    ) -> None:
        super().__init__(master, padding=(4, 4, 4, 2))
        self._profile_id = profile_id
        self._module_args = module_args
        self._fields: list[_FormField] = []
        self._default_help = "Focus a field to show option help."
        self._help_text = tk.StringVar(value=self._default_help)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        body = ttk.Frame(self, padding=(4, 2, 4, 0))
        body.grid(row=0, column=0, sticky="nsew")

        actions = _iter_parser_actions(parser)
        max_columns = _max_form_columns(len(actions))
        for column_index in range(max_columns):
            body.columnconfigure(column_index, weight=1, uniform="arg-columns")

        row_index = 0
        column_index = 0
        for action in actions:
            span = min(_field_column_span(action, max_columns=max_columns), max_columns)
            if column_index + span > max_columns:
                row_index += 1
                column_index = 0

            card = ttk.Frame(body, padding=(3, 1, 3, 1))
            card.grid(
                row=row_index,
                column=column_index,
                columnspan=span,
                sticky="ew",
                padx=2,
                pady=1,
            )
            card.columnconfigure(0, weight=1)

            help_text = action.help or ""
            label_text = _display_label(action)
            if _is_boolean_optional(action):
                variable = tk.BooleanVar(value=bool(action.default))
                widget = ttk.Checkbutton(card, variable=variable, text=label_text)
                widget.grid(row=0, column=0, sticky="w")
                self._bind_help(widget, help_text)
                self._fields.append(_FormField(action=action, variable=variable))
            else:
                ttk.Label(card, text=label_text).grid(row=0, column=0, sticky="w")
                variable = tk.StringVar(value=_default_text(action))
                widget = self._build_value_widget(card, action=action, variable=variable)
                self._bind_help(widget, help_text)
                self._fields.append(_FormField(action=action, variable=variable))

            column_index += span
            if column_index >= max_columns:
                row_index += 1
                column_index = 0

        help_label = ttk.Label(
            self,
            textvariable=self._help_text,
            foreground="#555555",
            padding=(8, 4, 8, 2),
            anchor="w",
        )
        help_label.grid(row=1, column=0, sticky="ew")

    def _bind_help(self, widget: tk.Misc, help_text: str) -> None:
        if not help_text:
            return

        def _on_focus_in(_: tk.Event[tk.Misc]) -> None:
            self._help_text.set(help_text)

        def _on_focus_out(_: tk.Event[tk.Misc]) -> None:
            self._help_text.set(self._default_help)

        widget.bind("<FocusIn>", _on_focus_in)
        widget.bind("<FocusOut>", _on_focus_out)

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
                width=max(_widget_width_for_action(action), 24),
            )
            widget.grid(row=0, column=0, sticky="ew")
            browse_button = ttk.Button(
                row,
                text="...",
                width=3,
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
        initial_dir = str(Path(current_value).parent) if current_value else str(_REPO_ROOT)
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
        self.geometry("1240x860")
        self.minsize(1040, 700)

        self._event_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._forms: dict[str, _ArgForm] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=4)
        self.rowconfigure(3, weight=1)

        header = ttk.Label(
            self,
            text="Run DQN training/eval and KPI evaluations from one GUI (all CLI flags exposed).",
            padding=(10, 8),
        )
        header.grid(row=0, column=0, sticky="w")

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self._notebook.bind("<<NotebookTabChanged>>", lambda _: self._refresh_preview())

        self._build_profiles()

        controls = ttk.Frame(self, padding=(10, 0, 10, 8))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self._command_preview = tk.StringVar(value="")
        preview_entry = ttk.Entry(controls, textvariable=self._command_preview)
        preview_entry.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        self._preview_button = ttk.Button(controls, text="Preview Command", command=self._refresh_preview)
        self._preview_button.grid(row=1, column=0, sticky="w")
        self._run_button = ttk.Button(controls, text="Run", command=self._run_selected)
        self._run_button.grid(row=1, column=1)
        self._stop_button = ttk.Button(controls, text="Stop", command=self._stop_process, state="disabled")
        self._stop_button.grid(row=1, column=2, sticky="e")

        output_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        output_frame.grid(row=3, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self._output = tk.Text(output_frame, wrap="none", height=10)
        self._output.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self._output.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self._output.configure(yscrollcommand=output_scroll.set)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_event_queue)
        self._refresh_preview()

    def _build_profiles(self) -> None:
        run_parser = dqn_policy_runner._build_parser()
        evaluate_parser = evaluate._build_parser()
        eval_run_parser = _get_subparser(evaluate_parser, command_name="run")
        eval_compare_parser = _get_subparser(evaluate_parser, command_name="compare")

        profiles = (
            ("run-dqn", "DQN Run (train/eval)", run_parser, ("-m", "src.env.dqn_policy_runner")),
            (
                "eval-run",
                "Evaluate Run",
                eval_run_parser,
                ("-m", "src.training.evaluate", "run"),
            ),
            (
                "eval-compare",
                "Evaluate Compare",
                eval_compare_parser,
                ("-m", "src.training.evaluate", "compare"),
            ),
        )

        for profile_id, title, parser, module_args in profiles:
            form = _ArgForm(
                self._notebook,
                profile_id=profile_id,
                parser=parser,
                module_args=module_args,
            )
            self._forms[profile_id] = form
            self._notebook.add(form, text=title)

    def _current_form(self) -> _ArgForm:
        selected_tab = self._notebook.select()
        current_widget = self.nametowidget(selected_tab)
        if not isinstance(current_widget, _ArgForm):
            raise RuntimeError("Selected tab is not a valid form.")
        return current_widget

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

        self._output.insert("end", f">>> {_format_command(command)}\n")
        self._output.see("end")
        self._run_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._preview_button.configure(state="disabled")

        self._reader_thread = threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
        )
        self._reader_thread.start()

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

    def _drain_event_queue(self) -> None:
        while True:
            try:
                kind, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._output.insert("end", f"{payload}\n")
                self._output.see("end")
            elif kind == "done":
                code = int(payload)
                self._output.insert("end", f"[process exited with code {code}]\n")
                self._output.see("end")
                self._run_button.configure(state="normal")
                self._stop_button.configure(state="disabled")
                self._preview_button.configure(state="normal")
                self._refresh_preview()
        self.after(100, self._drain_event_queue)

    def _on_close(self) -> None:
        if self._process is not None:
            self._stop_process()
        self.destroy()


def main() -> None:
    app = DqnRunnerGui()
    app.mainloop()


if __name__ == "__main__":
    main()

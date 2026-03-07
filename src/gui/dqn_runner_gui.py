"""Simple Tkinter launcher for DQN run/evaluation workflows."""

from __future__ import annotations

import argparse
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
        super().__init__(master)
        self._profile_id = profile_id
        self._module_args = module_args
        self._fields: list[_FormField] = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        canvas = tk.Canvas(self, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        body = ttk.Frame(canvas, padding=10)
        body.columnconfigure(1, weight=1)
        window = canvas.create_window((0, 0), window=body, anchor="nw")

        def _sync_scrollregion(_: tk.Event[tk.Misc]) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_inner_width(event: tk.Event[tk.Misc]) -> None:
            canvas.itemconfigure(window, width=event.width)

        body.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_inner_width)

        row_index = 0
        for action in _iter_parser_actions(parser):
            field_label = _primary_option(action)
            help_text = action.help or ""

            ttk.Label(body, text=field_label).grid(row=row_index, column=0, sticky="nw", padx=(0, 8))
            if _is_boolean_optional(action):
                variable = tk.BooleanVar(value=bool(action.default))
                checkbox = ttk.Checkbutton(body, variable=variable, text=help_text)
                checkbox.grid(row=row_index, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
                self._fields.append(_FormField(action=action, variable=variable))
                row_index += 1
                continue

            if action.choices is not None:
                default_value = _default_text(action)
                initial = default_value or str(next(iter(action.choices)))
                variable = tk.StringVar(value=initial)
                widget = ttk.Combobox(
                    body,
                    textvariable=variable,
                    values=[str(choice) for choice in action.choices],
                    state="readonly",
                )
            else:
                variable = tk.StringVar(value=_default_text(action))
                widget = ttk.Entry(body, textvariable=variable, width=48)

            widget.grid(row=row_index, column=1, sticky="ew", padx=(0, 8))
            if action.dest in _PATH_LIKE_DESTS:
                browse_button = ttk.Button(
                    body,
                    text="Browse",
                    command=lambda var=variable, dest=action.dest: self._browse_path(var, dest),
                )
                browse_button.grid(row=row_index, column=2, sticky="e", padx=(0, 8))
            if help_text:
                ttk.Label(body, text=help_text, foreground="#555555").grid(
                    row=row_index + 1,
                    column=1,
                    columnspan=2,
                    sticky="w",
                    padx=(0, 8),
                    pady=(0, 8),
                )
                row_index += 2
            else:
                row_index += 1
            self._fields.append(_FormField(action=action, variable=variable))

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
        self.geometry("1100x820")
        self.minsize(900, 680)

        self._event_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._forms: dict[str, _ArgForm] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
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

        self._output = tk.Text(output_frame, wrap="none")
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


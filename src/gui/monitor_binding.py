"""Live monitor binding and status parsing helpers for the Hybrid GUI."""

from __future__ import annotations

import ctypes
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.env.runner_tui import ACTIVE_RUNNER_SESSION_FILE

STATUS_KV_PATTERN = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")
TEXTUAL_MARKUP_PATTERN = re.compile(r"\[[^\]]+\]")
LIVE_MONITOR_RUNNER_MODULES = {
    "src.env.heuristic_policy_runner",
    "src.env.random_policy_runner",
    "src.hybrid.runner",
}
LIVE_MONITOR_TUI_MODULE = "src.memory.state_monitor_tui"
MONITOR_ACTION_TO_KEY_LABEL = {
    "move_up": "UP",
    "move_left": "LEFT",
    "move_down": "DOWN",
    "move_right": "RIGHT",
    "space": "SPACE",
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


def parse_status_values(line: str) -> dict[str, str]:
    return {key: value for key, value in STATUS_KV_PATTERN.findall(line)}


def strip_textual_markup(text: str) -> str:
    return TEXTUAL_MARKUP_PATTERN.sub("", text)


def resolve_reward_metric_value(
    *,
    training_line: str,
    reward_line: str,
    previous_value: str = "-",
) -> str:
    status_values = parse_status_values(training_line)
    reward_value = status_values.get("reward")
    if reward_value:
        return reward_value
    reward_values = parse_status_values(reward_line)
    reward_total = reward_values.get("total")
    if reward_total:
        return reward_total
    return previous_value


def monitor_action_card_values(action_line: str) -> dict[str, str]:
    action_values = parse_status_values(action_line)
    return {
        "action": action_values.get("action", "-"),
        "reason": action_values.get("reason", "-"),
        "phase": action_values.get("phase", "-"),
        "target": action_values.get("next_target", "-"),
        "loss": action_values.get("loss", "-"),
    }


def parse_next_available_actions(next_available_actions_line: str) -> tuple[str, ...]:
    status_values = parse_status_values(next_available_actions_line)
    raw_actions = status_values.get("next_available_actions", "")
    if not raw_actions or raw_actions == "-":
        return ()
    return tuple(
        token
        for token in (item.strip() for item in raw_actions.split(","))
        if token and token != "-" and not token.startswith("...(")
    )


def monitor_key_label_for_action(action_name: str) -> str | None:
    return MONITOR_ACTION_TO_KEY_LABEL.get(str(action_name).strip())


def status_key_label(key: str) -> str:
    return str(key).replace("_", " ")


def format_reward_breakdown_tooltip(reward_line: str) -> str:
    reward_values = parse_status_values(reward_line)
    if not reward_values:
        return "Reward breakdown unavailable"
    label_width = max(len(status_key_label(key)) for key in reward_values)
    lines = ["reward breakdown"]
    for key, value in reward_values.items():
        lines.append(f"{status_key_label(key):<{label_width}}  {value}")
    return "\n".join(lines)


def format_phase_breakdown_tooltip(action_line: str) -> str:
    action_values = monitor_action_card_values(action_line)
    fields = (
        ("Reason", action_values["reason"]),
        ("Action", action_values["action"]),
        ("Phase", action_values["phase"]),
        ("Target", action_values["target"]),
    )
    if all(value == "-" for _, value in fields):
        return "Phase detail unavailable"
    label_width = max(len(label) for label, _ in fields)
    lines = ["phase detail"]
    for label, value in fields:
        lines.append(f"{label:<{label_width}}  {value}")
    return "\n".join(lines)


def parse_episode_progress(
    raw_episode: str,
    *,
    fallback_total: int | None = None,
) -> tuple[int | None, int | None]:
    text = str(raw_episode).strip()
    if not text:
        return None, fallback_total
    if "/" in text:
        current_text, _, total_text = text.partition("/")
        try:
            current = int(current_text)
            total = int(total_text)
        except ValueError:
            return None, fallback_total
        return current if current >= 1 else None, total if total >= 1 else fallback_total
    try:
        episode_number = int(text)
    except ValueError:
        match = re.search(r"(\d+)$", text)
        if match is None:
            return None, fallback_total
        episode_number = int(match.group(1))
    return episode_number if episode_number >= 1 else None, fallback_total


def parse_step_value(raw_step: str) -> int | None:
    text = str(raw_step).strip()
    if not text:
        return None
    if "/" in text:
        text, _, _ = text.partition("/")
    try:
        step = int(text)
    except ValueError:
        return None
    return step if step >= 1 else None


def parse_bool_value(raw_value: str) -> bool | None:
    text = str(raw_value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def format_duration_seconds(total_seconds: float | None) -> str:
    if total_seconds is None or total_seconds < 0:
        return "-"
    rounded = int(round(total_seconds))
    if rounded <= 0:
        return "0s"
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def estimate_epsilon_eta_seconds(
    *,
    current_epsilon: float | None,
    epsilon_start: float | None,
    epsilon_end: float | None,
    epsilon_decay_steps: int | None,
    seconds_per_step: float | None,
) -> float | None:
    if (
        current_epsilon is None
        or epsilon_start is None
        or epsilon_end is None
        or epsilon_decay_steps is None
        or epsilon_decay_steps <= 0
        or seconds_per_step is None
        or seconds_per_step <= 0
    ):
        return None
    if epsilon_start <= epsilon_end:
        return 0.0 if current_epsilon <= epsilon_end else None
    clamped = max(epsilon_end, min(epsilon_start, current_epsilon))
    remaining_fraction = (clamped - epsilon_end) / (epsilon_start - epsilon_end)
    remaining_steps = max(0.0, float(epsilon_decay_steps) * remaining_fraction)
    return remaining_steps * seconds_per_step


def format_epsilon_progress_text(
    *,
    current_epsilon: float | None,
    epsilon_end: float | None,
) -> str:
    if current_epsilon is None:
        if epsilon_end is None:
            return "-"
        clamped_end = max(0.0, min(1.0, epsilon_end))
        return f"end {clamped_end * 100.0:.1f}%"
    clamped_current = max(0.0, min(1.0, current_epsilon))
    current_text = f"{clamped_current * 100.0:.1f}%"
    if epsilon_end is None:
        return current_text
    clamped_end = max(0.0, min(1.0, epsilon_end))
    return f"{current_text} -> {clamped_end * 100.0:.1f}%"


@dataclass(frozen=True)
class RunningPythonProcess:
    pid: int
    parent_pid: int | None
    executable_name: str
    command_line: str


@dataclass(frozen=True)
class LiveMonitorSessionBinding:
    runner_pid: int
    runner_module: str
    status_file: Path
    control_file: Path | None
    executable_name: str | None
    source_pid: int
    source_module: str


def split_windows_command_line(command_line: str) -> tuple[str, ...]:
    text = str(command_line).strip()
    if not text:
        return ()
    if os.name != "nt":
        return tuple(token.strip('"') for token in shlex.split(text, posix=False))

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    argc = ctypes.c_int(0)
    argv_pointer = shell32.CommandLineToArgvW(text, ctypes.byref(argc))
    if not argv_pointer:
        return ()
    try:
        return tuple(str(argv_pointer[index]) for index in range(argc.value))
    finally:
        kernel32.LocalFree(ctypes.cast(argv_pointer, ctypes.c_void_p))


def normalize_cli_option_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip('"')
    return normalized or None


def extract_cli_option_value(arguments: tuple[str, ...], option_name: str) -> str | None:
    for index, token in enumerate(arguments):
        if token == option_name:
            if index + 1 >= len(arguments):
                return None
            return normalize_cli_option_value(arguments[index + 1])
        if token.startswith(f"{option_name}="):
            return normalize_cli_option_value(token.partition("=")[2])
    return None


def parse_module_invocation(arguments: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    for index, token in enumerate(arguments):
        if token == "-m" and index + 1 < len(arguments):
            return str(arguments[index + 1]), tuple(arguments[index + 2 :])
    return (None, ())


def is_live_monitor_runner_module(module_name: str | None, module_args: tuple[str, ...]) -> bool:
    del module_args
    return module_name in LIVE_MONITOR_RUNNER_MODULES


def normalize_executable_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return Path(normalized).name.lower()


def build_live_monitor_binding_for_runner(
    *,
    process: RunningPythonProcess,
    runner_module: str,
    runner_args: tuple[str, ...],
    children: tuple[RunningPythonProcess, ...],
    parsed_arguments_by_pid: dict[int, tuple[str | None, tuple[str, ...]]],
    monitor_bindings_by_runner_pid: dict[int, tuple[LiveMonitorSessionBinding, ...]],
) -> LiveMonitorSessionBinding | None:
    executable_name = extract_cli_option_value(runner_args, "--exe")
    status_file_text = extract_cli_option_value(runner_args, "--external-status-file")
    control_file_text = extract_cli_option_value(runner_args, "--external-control-file")
    source_pid = process.pid
    source_module = runner_module

    if status_file_text is None:
        declared_monitor_bindings = monitor_bindings_by_runner_pid.get(process.pid, ())
        if declared_monitor_bindings:
            selected_binding = select_live_monitor_binding(
                declared_monitor_bindings,
                preferred_runner_pid=process.pid,
                preferred_executable_name=executable_name,
            )
            if selected_binding is not None:
                return LiveMonitorSessionBinding(
                    runner_pid=process.pid,
                    runner_module=runner_module,
                    status_file=selected_binding.status_file,
                    control_file=selected_binding.control_file,
                    executable_name=(executable_name or selected_binding.executable_name),
                    source_pid=selected_binding.source_pid,
                    source_module=selected_binding.source_module,
                )

        child_candidates: list[tuple[int, str, str | None, str | None]] = []
        for child in children:
            child_module, child_args = parsed_arguments_by_pid.get(child.pid, (None, ()))
            if child_module != LIVE_MONITOR_TUI_MODULE:
                continue
            child_status_file = extract_cli_option_value(child_args, "--external-status-file")
            if child_status_file is None:
                continue
            child_candidates.append(
                (
                    child.pid,
                    child_status_file,
                    extract_cli_option_value(child_args, "--external-control-file"),
                    extract_cli_option_value(child_args, "--exe"),
                )
            )
        if child_candidates:
            child_candidates.sort(reverse=True)
            source_pid, status_file_text, child_control_file, child_executable_name = child_candidates[0]
            source_module = LIVE_MONITOR_TUI_MODULE
            if control_file_text is None:
                control_file_text = child_control_file
            if executable_name is None:
                executable_name = child_executable_name

    if status_file_text is None:
        return None

    return LiveMonitorSessionBinding(
        runner_pid=process.pid,
        runner_module=runner_module,
        status_file=Path(status_file_text),
        control_file=(Path(control_file_text) if control_file_text else None),
        executable_name=executable_name,
        source_pid=source_pid,
        source_module=source_module,
    )


def load_live_monitor_binding_from_status_file(
    *,
    status_file: Path | None,
    fallback_control_file: Path | None = None,
) -> LiveMonitorSessionBinding | None:
    if status_file is None or not status_file.exists():
        return None
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    try:
        runner_pid = int(payload.get("runner_pid", 0))
    except (TypeError, ValueError):
        runner_pid = 0
    if runner_pid <= 0:
        return None

    runner_module = str(payload.get("runner_module", "") or "").strip() or "unknown"
    executable_name = normalize_cli_option_value(str(payload.get("runner_executable_name", "") or ""))
    control_file_text = normalize_cli_option_value(str(payload.get("runner_control_file", "") or ""))
    control_file = Path(control_file_text) if control_file_text else fallback_control_file
    return LiveMonitorSessionBinding(
        runner_pid=runner_pid,
        runner_module=runner_module,
        status_file=status_file,
        control_file=control_file,
        executable_name=executable_name,
        source_pid=runner_pid,
        source_module=runner_module,
    )


def load_active_runner_session_binding(
    *,
    registry_file: Path = ACTIVE_RUNNER_SESSION_FILE,
) -> LiveMonitorSessionBinding | None:
    if not registry_file.exists():
        return None
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    status_file_text = normalize_cli_option_value(str(payload.get("status_file", "") or ""))
    if status_file_text is None:
        return None
    status_file = Path(status_file_text)
    if not status_file.exists():
        return None
    try:
        runner_pid = int(payload.get("runner_pid", 0))
    except (TypeError, ValueError):
        runner_pid = 0
    if runner_pid <= 0:
        return None
    control_file_text = normalize_cli_option_value(str(payload.get("control_file", "") or ""))
    runner_module = str(payload.get("runner_module", "") or "").strip() or "unknown"
    executable_name = normalize_cli_option_value(str(payload.get("runner_executable_name", "") or ""))
    return LiveMonitorSessionBinding(
        runner_pid=runner_pid,
        runner_module=runner_module,
        status_file=status_file,
        control_file=(Path(control_file_text) if control_file_text else None),
        executable_name=executable_name,
        source_pid=runner_pid,
        source_module="active_registry",
    )


def select_live_monitor_binding(
    bindings: tuple[LiveMonitorSessionBinding, ...],
    *,
    preferred_runner_pid: int | None = None,
    preferred_executable_name: str | None = None,
) -> LiveMonitorSessionBinding | None:
    if not bindings:
        return None

    preferred_executable_key = normalize_executable_name(preferred_executable_name)

    def _sort_key(binding: LiveMonitorSessionBinding) -> tuple[int, int, int, int, int, int]:
        executable_match = (
            1
            if preferred_executable_key is not None
            and normalize_executable_name(binding.executable_name) == preferred_executable_key
            else 0
        )
        pid_match = 1 if preferred_runner_pid is not None and binding.runner_pid == preferred_runner_pid else 0
        direct_binding = 1 if binding.source_module == binding.runner_module else 0
        has_control_file = 1 if binding.control_file is not None else 0
        return (
            pid_match,
            executable_match,
            direct_binding,
            has_control_file,
            binding.runner_pid,
            binding.source_pid,
        )

    return max(bindings, key=_sort_key)


def discover_live_monitor_session_binding(
    processes: tuple[RunningPythonProcess, ...],
    *,
    preferred_runner_pid: int | None = None,
    preferred_executable_name: str | None = None,
) -> LiveMonitorSessionBinding | None:
    if not processes:
        return None

    children_by_parent_pid: dict[int, list[RunningPythonProcess]] = {}
    parsed_arguments_by_pid: dict[int, tuple[str | None, tuple[str, ...]]] = {}
    monitor_bindings_by_runner_pid: dict[int, list[LiveMonitorSessionBinding]] = {}
    for process in processes:
        parsed_arguments = parse_module_invocation(split_windows_command_line(process.command_line))
        parsed_arguments_by_pid[process.pid] = parsed_arguments
        module_name, module_args = parsed_arguments
        if module_name == LIVE_MONITOR_TUI_MODULE:
            runner_pid_text = extract_cli_option_value(module_args, "--runner-pid")
            status_file_text = extract_cli_option_value(module_args, "--external-status-file")
            if runner_pid_text is not None and status_file_text is not None:
                try:
                    declared_runner_pid = int(runner_pid_text)
                except (TypeError, ValueError):
                    declared_runner_pid = 0
                if declared_runner_pid > 0:
                    monitor_bindings_by_runner_pid.setdefault(declared_runner_pid, []).append(
                        LiveMonitorSessionBinding(
                            runner_pid=declared_runner_pid,
                            runner_module="unknown",
                            status_file=Path(status_file_text),
                            control_file=(
                                Path(control_file_text)
                                if (control_file_text := extract_cli_option_value(
                                    module_args, "--external-control-file"
                                ))
                                else None
                            ),
                            executable_name=extract_cli_option_value(module_args, "--exe"),
                            source_pid=process.pid,
                            source_module=LIVE_MONITOR_TUI_MODULE,
                        )
                    )
        if process.parent_pid is not None:
            children_by_parent_pid.setdefault(process.parent_pid, []).append(process)

    bindings: list[LiveMonitorSessionBinding] = []
    for process in processes:
        runner_module, runner_args = parsed_arguments_by_pid.get(process.pid, (None, ()))
        if not is_live_monitor_runner_module(runner_module, runner_args):
            continue
        if runner_module is None:
            continue
        binding = build_live_monitor_binding_for_runner(
            process=process,
            runner_module=runner_module,
            runner_args=runner_args,
            children=tuple(children_by_parent_pid.get(process.pid, ())),
            parsed_arguments_by_pid=parsed_arguments_by_pid,
            monitor_bindings_by_runner_pid={
                pid: tuple(candidate_bindings)
                for pid, candidate_bindings in monitor_bindings_by_runner_pid.items()
            },
        )
        if binding is not None:
            bindings.append(binding)

    return select_live_monitor_binding(
        tuple(bindings),
        preferred_runner_pid=preferred_runner_pid,
        preferred_executable_name=preferred_executable_name,
    )


def list_running_python_processes() -> tuple[RunningPythonProcess, ...]:
    if os.name != "nt":
        return ()

    powershell_command = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and ($_.Name -match '^(python|pythonw|py)\\.exe$') } | "
        "Select-Object ProcessId, ParentProcessId, Name, CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", powershell_command],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    if completed.returncode != 0:
        return ()
    stdout_text = completed.stdout.strip()
    if not stdout_text:
        return ()

    try:
        payload = json.loads(stdout_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()

    rows = payload if isinstance(payload, list) else [payload]
    processes: list[RunningPythonProcess] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get("ProcessId"))
        except (TypeError, ValueError):
            continue
        try:
            parent_pid = int(row.get("ParentProcessId")) if row.get("ParentProcessId") is not None else None
        except (TypeError, ValueError):
            parent_pid = None
        command_line = str(row.get("CommandLine") or "").strip()
        if not command_line:
            continue
        processes.append(
            RunningPythonProcess(
                pid=pid,
                parent_pid=parent_pid,
                executable_name=str(row.get("Name") or ""),
                command_line=command_line,
            )
        )
    return tuple(processes)

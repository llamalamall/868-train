"""Status-file parsing helpers for the Hybrid GUI live monitor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

STATUS_KV_PATTERN = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")
TEXTUAL_MARKUP_PATTERN = re.compile(r"\[[^\]]+\]")
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
class LiveMonitorSessionBinding:
    runner_pid: int
    runner_module: str
    status_file: Path
    control_file: Path | None
    executable_name: str | None


def normalize_cli_option_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip('"')
    return normalized or None


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
    )

"""Hybrid GUI preset and checkpoint helper functions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
HYBRID_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "hybrid"
APPDATA_GAME_SAVE_DIR = (
    Path(os.environ["APPDATA"]) / "868-hack"
    if os.environ.get("APPDATA")
    else None
)
AUTO_LATEST_BETA_META_CHECKPOINT = "__AUTO_LATEST_BETA_META_CHECKPOINT__"


def initial_browse_dir(*, dest: str, current_value: str) -> Path:
    if current_value:
        current_path = Path(current_value)
        if current_path.suffix:
            return current_path.parent
        return current_path
    if dest in {"checkpoint", "checkpoint_root", "resume_checkpoint", "warmstart_checkpoint"}:
        return HYBRID_CHECKPOINT_DIR
    if (
        dest == "restore_save_file"
        and APPDATA_GAME_SAVE_DIR is not None
        and APPDATA_GAME_SAVE_DIR.exists()
    ):
        return APPDATA_GAME_SAVE_DIR
    return REPO_ROOT


def run_hybrid_preset_overrides(*, command_name: str) -> dict[str, dict[str, object]]:
    if command_name == "movement-test":
        return {
            "defaults": {},
            "gate a smoke": {
                "episodes": 3,
                "max_steps": 180,
                "no_enemies": True,
                "threat_trigger_distance": 2,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "long route validation": {
                "episodes": 10,
                "max_steps": 320,
                "no_enemies": True,
                "threat_trigger_distance": 2,
                "prog_actions": False,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
        }
    if command_name == "train-meta-no-enemies":
        return {
            "defaults": {},
            "gate b baseline": {
                "episodes": 120,
                "max_steps": 350,
                "no_enemies": True,
                "meta_epsilon_start": 0.6,
                "meta_epsilon_end": 0.05,
                "meta_epsilon_decay_steps": 5000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "quick tune": {
                "episodes": 60,
                "max_steps": 280,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "beta efficient warmstart": {
                "episodes": 120,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "run_tag": "hybrid-meta-beta-efficient",
            },
            "gamma logic rerun": {
                "episodes": 120,
                "max_steps": 350,
                "no_enemies": True,
                "meta_gamma": 0.99,
                "meta_learning_rate": 0.001,
                "meta_epsilon_start": 0.6,
                "meta_epsilon_end": 0.05,
                "meta_epsilon_decay_steps": 5000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "run_tag": "hybrid-meta-gamma-fixedlogic",
            },
            "efficient anti-churn": {
                "episodes": 120,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "run_tag": "hybrid-meta-efficient-fixedlogic",
            },
            "meta ack sweep balanced": {
                "episodes": 30,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "post_action_delay": 0.03,
                "action_ack_timeout": 0.50,
                "action_ack_poll_interval": 0.02,
                "action_ack_backoff_max_level": 0,
                "run_tag": "hybrid-meta-ack-balanced",
            },
            "meta ack sweep conservative": {
                "episodes": 30,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "post_action_delay": 0.05,
                "action_ack_timeout": 0.70,
                "action_ack_poll_interval": 0.02,
                "action_ack_backoff_max_level": 0,
                "run_tag": "hybrid-meta-ack-conservative",
            },
            "meta ack sweep fast poll": {
                "episodes": 30,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "post_action_delay": 0.02,
                "action_ack_timeout": 0.50,
                "action_ack_poll_interval": 0.01,
                "action_ack_backoff_max_level": 0,
                "run_tag": "hybrid-meta-ack-fastpoll",
            },
            "beta efficient conservative ack": {
                "episodes": 120,
                "max_steps": 320,
                "no_enemies": True,
                "meta_learning_rate": 0.0005,
                "meta_epsilon_decay_steps": 3000,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "post_action_delay": 0.05,
                "action_ack_timeout": 0.70,
                "action_ack_poll_interval": 0.02,
                "action_ack_backoff_max_level": 0,
                "run_tag": "hybrid-meta-beta-efficient-conservative-ack",
            },
        }
    if command_name == "train-full-hierarchical":
        return {
            "defaults": {},
            "gate c baseline": {
                "episodes": 200,
                "max_steps": 450,
                "no_enemies": False,
                "meta_freeze_episodes": 25,
                "joint_finetune": True,
                "threat_trigger_distance": 2,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "threat warmup heavy": {
                "episodes": 240,
                "max_steps": 450,
                "no_enemies": False,
                "meta_freeze_episodes": 60,
                "joint_finetune": True,
                "threat_learning_rate": 0.0007,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "beta full fixed meta": {
                "episodes": 320,
                "max_steps": 450,
                "no_enemies": False,
                "warmstart_checkpoint": AUTO_LATEST_BETA_META_CHECKPOINT,
                "meta_freeze_episodes": 0,
                "joint_finetune": False,
                "threat_learning_rate": 0.0007,
                "threat_epsilon_start": 0.80,
                "threat_epsilon_end": 0.05,
                "threat_epsilon_decay_steps": 15000,
                "threat_trigger_distance": 2,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "run_tag": "hybrid-full-beta-fixedmeta",
            },
            "beta full long warmup": {
                "episodes": 320,
                "max_steps": 450,
                "no_enemies": False,
                "warmstart_checkpoint": AUTO_LATEST_BETA_META_CHECKPOINT,
                "meta_freeze_episodes": 80,
                "joint_finetune": True,
                "threat_learning_rate": 0.0007,
                "threat_epsilon_start": 0.80,
                "threat_epsilon_end": 0.05,
                "threat_epsilon_decay_steps": 15000,
                "threat_trigger_distance": 2,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
                "run_tag": "hybrid-full-beta-longwarmup",
            },
        }
    if command_name == "eval-hybrid":
        return {
            "defaults": {},
            "eval quick": {
                "episodes": 10,
                "max_steps": 300,
                "no_enemies": False,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "eval stress": {
                "episodes": 25,
                "max_steps": 500,
                "no_enemies": False,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
            "beta verification": {
                "episodes": 30,
                "max_steps": 450,
                "no_enemies": False,
                "phase_lock_min_steps": 6,
                "target_stall_release_steps": 4,
            },
        }
    return {"defaults": {}}


def _parse_saved_at_utc(value: object, *, fallback_path: Path) -> datetime:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
    return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)


def _is_completed_training_state(training_state: dict[str, object]) -> bool:
    requested_raw = training_state.get("episodes_requested")
    completed_raw = training_state.get("episodes_completed")
    results_raw = training_state.get("results")
    try:
        requested = int(requested_raw) if requested_raw is not None else 0
    except (TypeError, ValueError):
        requested = 0
    try:
        completed = int(completed_raw) if completed_raw is not None else 0
    except (TypeError, ValueError):
        completed = 0
    if requested <= 0 and isinstance(results_raw, list):
        requested = len(results_raw)
    if completed <= 0 and isinstance(results_raw, list):
        completed = len(results_raw)
    return requested > 0 and completed >= requested


def latest_completed_meta_checkpoint(*, checkpoint_root: Path | None = None) -> Path:
    resolved_root = checkpoint_root or HYBRID_CHECKPOINT_DIR
    if not resolved_root.exists():
        raise ValueError(
            f"No completed train-meta-no-enemies hybrid checkpoints found under {resolved_root}."
        )
    candidates: list[tuple[datetime, Path, bool]] = []
    for child in resolved_root.iterdir():
        if not child.is_dir():
            continue
        config_path = child / "hybrid_config.json"
        training_state_path = child / "training_state.json"
        if not config_path.exists() or not training_state_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(config, dict) or not isinstance(training_state, dict):
            continue
        if str(config.get("command") or "").strip() != "train-meta-no-enemies":
            continue
        if not _is_completed_training_state(training_state):
            continue
        saved_at = _parse_saved_at_utc(training_state.get("saved_at_utc"), fallback_path=child)
        candidates.append((saved_at, child, "beta" in child.name.lower()))

    if not candidates:
        raise ValueError(
            f"No completed train-meta-no-enemies hybrid checkpoints found under {resolved_root}."
        )

    beta_candidates = [item for item in candidates if item[2]]
    selected_pool = beta_candidates or candidates
    _saved_at, selected_path, _is_beta = max(selected_pool, key=lambda item: (item[0], item[1].name))
    return selected_path


def resolve_preset_overrides(
    *,
    overrides: dict[str, object],
    latest_checkpoint_resolver: Callable[[], Path] | None = None,
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    latest_meta_checkpoint: str | None = None
    checkpoint_resolver = latest_checkpoint_resolver or latest_completed_meta_checkpoint
    for dest, value in overrides.items():
        if value == AUTO_LATEST_BETA_META_CHECKPOINT:
            if latest_meta_checkpoint is None:
                latest_meta_checkpoint = str(checkpoint_resolver())
            resolved[dest] = latest_meta_checkpoint
            continue
        resolved[dest] = value
    return resolved

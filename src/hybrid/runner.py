"""CLI runner for hybrid hierarchical training and evaluation workflows."""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from src.env.runner_tui import RunnerTuiSession
from src.env.random_policy_runner import (
    _default_game_save_target_path,
    _resolve_restore_save_source_path,
    _restore_selected_save_file,
)
from src.hybrid.astar_controller import AStarMovementController
from src.hybrid.checkpoint import HybridCheckpointManager
from src.hybrid.coordinator import HybridCoordinator, HybridCoordinatorConfig
from src.hybrid.env import HybridLiveEnv, HybridLiveEnvConfig
from src.hybrid.meta_controller import MetaControllerDQN, MetaDQNConfig
from src.hybrid.rewards import HybridMetaRewardWeights, HybridRewardSuite
from src.hybrid.threat_controller import ThreatControllerDRQN, ThreatDRQNConfig
from src.hybrid.types import ObjectivePhase, ThreatOverride


def _game_tick_ms_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:  # pragma: no cover - argparse emits user-facing error.
        raise argparse.ArgumentTypeError("game tick must be an integer.") from error
    if parsed < 1 or parsed > 16:
        raise argparse.ArgumentTypeError("game tick ms must be between 1 and 16.")
    return parsed


def _format_monitor_actions(actions: object, *, limit: int = 8) -> str:
    if not isinstance(actions, (tuple, list)):
        return "-"
    normalized = tuple(str(item).strip() for item in actions if str(item).strip())
    if not normalized:
        return "-"
    return ",".join(normalized)


def _terminal_is_fail(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    normalized = reason.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in ("fail", "loss", "dead", "player_health"))


def _format_monitor_target(target: object) -> str:
    if target is None:
        return "none"
    target_x = getattr(target, "x", None)
    target_y = getattr(target, "y", None)
    if isinstance(target_x, int) and isinstance(target_y, int):
        return f"({target_x},{target_y})"
    return "none"


def _format_monitor_action_line(
    *,
    action: str,
    reason: str,
    phase: ObjectivePhase,
    target: object,
) -> str:
    return (
        "action={action} phase={phase} next_target={target} reason={reason}"
    ).format(
        action=action,
        phase=phase.value,
        target=_format_monitor_target(target),
        reason=reason,
    )


def _format_monitor_training_line(
    *,
    episode_id: str,
    step: int,
    total_reward: float,
    meta_epsilon: float,
    updates_applied: int,
    threat_epsilon: float | None = None,
    reward: float | None = None,
    waiting: bool = False,
    done: bool | None = None,
    terminal_reason: str | None = None,
) -> str:
    parts = [
        f"episode={episode_id}",
        f"step={step}",
    ]
    if reward is not None:
        parts.append(f"reward={reward:.3f}")
    parts.append(f"total={total_reward:.3f}")
    parts.append(f"epsilon={meta_epsilon:.4f}")
    if threat_epsilon is not None:
        parts.append(f"threat_epsilon={threat_epsilon:.4f}")
    parts.append(f"updates={updates_applied}")
    if waiting:
        parts.append("waiting=step")
    else:
        parts.append(f"done={bool(done)}")
        parts.append(f"terminal={terminal_reason or '-'}")
    return " ".join(parts)


@dataclass(frozen=True)
class HybridEpisodeSummary:
    """Per-episode rollout summary."""

    episode_id: str
    steps: int
    done: bool
    terminal_reason: str | None
    total_reward: float
    meta_reward_total: float
    threat_reward_total: float
    invalid_actions: int
    premature_exit_attempts: int
    route_length_total: int
    route_replans: int
    hit_step_limit: bool
    terminal_classification: str
    phase_switches: int
    threat_active_steps: int


def _add_common_runner_args(
    parser: argparse.ArgumentParser,
    *,
    default_episodes: int,
    default_max_steps: int,
    default_no_enemies: bool,
) -> None:
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument(
        "--launch-exe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch --exe when not already running before attempting attach.",
    )
    parser.add_argument("--episodes", type=int, default=default_episodes, help="Episode count.")
    parser.add_argument("--max-steps", type=int, default=default_max_steps, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--movement-keys",
        choices=("arrows", "wasd", "numpad"),
        default="arrows",
        help="Movement key mapping profile.",
    )
    parser.add_argument(
        "--siphon-key",
        choices=("space", "z"),
        default="space",
        help="Key used by siphon action.",
    )
    parser.add_argument(
        "--prog-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prog-slot actions (prog_slot_1..prog_slot_10).",
    )
    parser.add_argument(
        "--reset-sequence",
        default="confirm",
        nargs="?",
        const="",
        help="Comma-separated reset actions; pass without value to disable reset sequence.",
    )
    parser.add_argument(
        "--focus-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Focus game window during attach/reacquire.",
    )
    parser.add_argument(
        "--window-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use window-targeted input instead of global SendInput.",
    )
    parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch live state monitor TUI in a separate window.",
    )
    parser.add_argument(
        "--tui-interval",
        type=float,
        default=0.5,
        help="Polling interval for live TUI (seconds).",
    )
    parser.add_argument(
        "--external-status-file",
        default=None,
        help="Optional JSON file path to receive status updates for GUI monitor mode.",
    )
    parser.add_argument(
        "--external-control-file",
        default=None,
        help="Optional JSON file path for pause/step/resume controls.",
    )
    parser.add_argument(
        "--step-through",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause before each action and wait for monitor step token.",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=3.0,
        help="Per-step watchdog timeout in seconds.",
    )
    parser.add_argument(
        "--game-tick-ms",
        type=_game_tick_ms_arg,
        default=1,
        help="Target game loop tick size in milliseconds (1..16).",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=15.0,
        help="Reset watchdog timeout in seconds.",
    )
    parser.add_argument(
        "--post-action-delay",
        type=float,
        default=0.01,
        help="Fixed delay after dispatching each action before reading state.",
    )
    parser.add_argument(
        "--wait-for-action-processing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll for post-action state evidence before finalizing step.",
    )
    parser.add_argument(
        "--action-ack-timeout",
        type=float,
        default=0.35,
        help="Additional wait limit (seconds) to observe post-action state change.",
    )
    parser.add_argument(
        "--action-ack-poll-interval",
        type=float,
        default=0.05,
        help="Polling interval (seconds) while waiting for post-action state change.",
    )
    parser.add_argument(
        "--prog-backoff-steps",
        type=int,
        default=3,
        help="Fallback prog-slot backoff steps after ineffective attempts.",
    )
    parser.add_argument(
        "--require-non-terminal-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reset() to return non-terminal state before stepping.",
    )
    parser.add_argument(
        "--disable-idle-frame-delay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch idle frame delay for faster runtime pacing.",
    )
    parser.add_argument(
        "--disable-background-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable animated background motion effect.",
    )
    parser.add_argument(
        "--disable-wall-animations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze wall/tile animation counter.",
    )
    parser.add_argument(
        "--no-enemies",
        action=argparse.BooleanOptionalAction,
        default=default_no_enemies,
        help="Enable runtime enemy suppression.",
    )
    parser.add_argument(
        "--run-tag",
        default="hybrid",
        help="Suffix tag for run directory naming.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/hybrid",
        help="Root directory for saved hybrid checkpoint bundles.",
    )
    parser.add_argument(
        "--restore-save-file",
        default=None,
        help=(
            "Optional source save file to restore before each new-game confirm/reset action. "
            "When set, the file is copied to %APPDATA%\\868-hack\\savegame_868."
        ),
    )
    parser.add_argument(
        "--restore-save-delay",
        type=float,
        default=0.35,
        help="Delay in seconds after restoring save file before the next episode reset/new game.",
    )
    parser.add_argument(
        "--threat-trigger-distance",
        type=int,
        default=2,
        help="Threat activation distance threshold in grid tiles.",
    )
    parser.add_argument(
        "--print-reward-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-step split reward components.",
    )
    parser.add_argument(
        "--meta-reward-objective-complete",
        type=float,
        default=1.50,
        help="Meta reward: objective completion bonus.",
    )
    parser.add_argument(
        "--meta-reward-phase-progress",
        type=float,
        default=0.25,
        help="Meta reward: distance progress shaping coefficient.",
    )
    parser.add_argument(
        "--meta-reward-step-cost",
        type=float,
        default=0.01,
        help="Meta reward: per-step cost coefficient.",
    )
    parser.add_argument(
        "--meta-reward-premature-exit-penalty",
        type=float,
        default=1.25,
        help="Meta reward: premature-exit penalty.",
    )
    parser.add_argument(
        "--meta-reward-sector-advance",
        type=float,
        default=1.00,
        help="Meta reward: sector-advance bonus coefficient.",
    )
    parser.add_argument(
        "--meta-reward-final-sector-win",
        type=float,
        default=25.00,
        help="Meta reward: large bonus for exiting sector 8 and winning the run.",
    )


def _add_meta_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--meta-gamma", type=float, default=0.99, help="Meta DQN gamma.")
    parser.add_argument("--meta-learning-rate", type=float, default=0.001, help="Meta DQN learning rate.")
    parser.add_argument("--meta-replay-capacity", type=int, default=12_000, help="Meta replay capacity.")
    parser.add_argument("--meta-min-replay-size", type=int, default=256, help="Meta min replay size.")
    parser.add_argument("--meta-batch-size", type=int, default=64, help="Meta batch size.")
    parser.add_argument("--meta-target-sync-interval", type=int, default=250, help="Meta target sync interval.")
    parser.add_argument("--meta-epsilon-start", type=float, default=0.60, help="Meta epsilon start.")
    parser.add_argument("--meta-epsilon-end", type=float, default=0.05, help="Meta epsilon end.")
    parser.add_argument("--meta-epsilon-decay-steps", type=int, default=5_000, help="Meta epsilon decay steps.")
    parser.add_argument("--meta-hidden-size", type=int, default=64, help="Meta network hidden size.")
    parser.add_argument("--meta-feature-count", type=int, default=18, help="Meta feature vector size.")


def _add_threat_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threat-gamma", type=float, default=0.98, help="Threat DRQN gamma.")
    parser.add_argument("--threat-learning-rate", type=float, default=0.001, help="Threat DRQN learning rate.")
    parser.add_argument("--threat-replay-capacity", type=int, default=20_000, help="Threat replay capacity.")
    parser.add_argument("--threat-min-replay-size", type=int, default=512, help="Threat min replay size.")
    parser.add_argument("--threat-batch-size", type=int, default=64, help="Threat batch size.")
    parser.add_argument("--threat-target-sync-interval", type=int, default=300, help="Threat target sync interval.")
    parser.add_argument("--threat-epsilon-start", type=float, default=0.50, help="Threat epsilon start.")
    parser.add_argument("--threat-epsilon-end", type=float, default=0.05, help="Threat epsilon end.")
    parser.add_argument(
        "--threat-epsilon-decay-steps",
        type=int,
        default=7_500,
        help="Threat epsilon decay steps.",
    )
    parser.add_argument("--threat-hidden-size", type=int, default=96, help="Threat GRU hidden size.")
    parser.add_argument("--threat-feature-count", type=int, default=20, help="Threat feature vector size.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hybrid hierarchical workflows (A* movement + meta DQN + threat DRQN)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    movement_test = subparsers.add_parser(
        "movement-test",
        description="Phase 1: deterministic A* movement test with no enemies.",
    )
    _add_common_runner_args(
        movement_test,
        default_episodes=5,
        default_max_steps=250,
        default_no_enemies=True,
    )

    train_meta = subparsers.add_parser(
        "train-meta-no-enemies",
        description="Phase 2: train meta-controller with no enemies (threat controller frozen).",
    )
    _add_common_runner_args(
        train_meta,
        default_episodes=120,
        default_max_steps=350,
        default_no_enemies=True,
    )
    _add_meta_model_args(train_meta)
    train_meta.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional existing hybrid run directory to resume from.",
    )

    train_full = subparsers.add_parser(
        "train-full-hierarchical",
        description="Phase 3: train full hierarchy with enemies enabled.",
    )
    _add_common_runner_args(
        train_full,
        default_episodes=200,
        default_max_steps=450,
        default_no_enemies=False,
    )
    _add_meta_model_args(train_full)
    _add_threat_model_args(train_full)
    train_full.add_argument(
        "--warmstart-checkpoint",
        default=None,
        help="Hybrid run directory used to warm-start meta controller (typically phase 2 output).",
    )
    train_full.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional full bundle checkpoint directory to resume both controllers.",
    )
    train_full.add_argument(
        "--meta-freeze-episodes",
        type=int,
        default=25,
        help="Episodes to freeze meta updates while threat DRQN warms up.",
    )
    train_full.add_argument(
        "--joint-finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After warmup, continue training meta together with threat controller.",
    )

    eval_parser = subparsers.add_parser(
        "eval-hybrid",
        description="Evaluate a saved hybrid checkpoint bundle without exploration updates.",
    )
    _add_common_runner_args(
        eval_parser,
        default_episodes=20,
        default_max_steps=450,
        default_no_enemies=False,
    )
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Hybrid run directory containing meta/threat checkpoints.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.episodes) < 1:
        parser.error("--episodes must be >= 1.")
    if int(args.max_steps) < 1:
        parser.error("--max-steps must be >= 1.")
    if float(args.step_timeout) <= 0:
        parser.error("--step-timeout must be > 0.")
    if float(args.reset_timeout) <= 0:
        parser.error("--reset-timeout must be > 0.")
    if float(args.action_ack_poll_interval) < 0:
        parser.error("--action-ack-poll-interval must be >= 0.")
    if float(args.action_ack_timeout) < 0:
        parser.error("--action-ack-timeout must be >= 0.")
    if float(args.restore_save_delay) < 0:
        parser.error("--restore-save-delay must be >= 0.")
    if int(args.prog_backoff_steps) < 0:
        parser.error("--prog-backoff-steps must be >= 0.")
    if bool(args.step_through) and not bool(args.tui):
        parser.error("--step-through requires --tui.")
    restore_source = _resolve_restore_save_source_path(args)
    if restore_source is not None:
        if not restore_source.exists():
            parser.error(f"--restore-save-file not found: {restore_source}.")
        if not restore_source.is_file():
            parser.error(f"--restore-save-file must be a file: {restore_source}.")
    if args.command == "train-full-hierarchical" and not args.resume_checkpoint and not args.warmstart_checkpoint:
        parser.error(
            "--warmstart-checkpoint is required for train-full-hierarchical "
            "unless --resume-checkpoint is used."
        )


def _build_meta_config(args: argparse.Namespace) -> MetaDQNConfig:
    return MetaDQNConfig(
        gamma=float(getattr(args, "meta_gamma", 0.99)),
        learning_rate=float(getattr(args, "meta_learning_rate", 0.001)),
        replay_capacity=int(getattr(args, "meta_replay_capacity", 12_000)),
        min_replay_size=int(getattr(args, "meta_min_replay_size", 256)),
        batch_size=int(getattr(args, "meta_batch_size", 64)),
        target_sync_interval=int(getattr(args, "meta_target_sync_interval", 250)),
        epsilon_start=float(getattr(args, "meta_epsilon_start", 0.60)),
        epsilon_end=float(getattr(args, "meta_epsilon_end", 0.05)),
        epsilon_decay_steps=int(getattr(args, "meta_epsilon_decay_steps", 5_000)),
        hidden_size=int(getattr(args, "meta_hidden_size", 64)),
        feature_count=int(getattr(args, "meta_feature_count", 18)),
    )


def _build_threat_config(args: argparse.Namespace) -> ThreatDRQNConfig:
    return ThreatDRQNConfig(
        gamma=float(getattr(args, "threat_gamma", 0.98)),
        learning_rate=float(getattr(args, "threat_learning_rate", 0.001)),
        replay_capacity=int(getattr(args, "threat_replay_capacity", 20_000)),
        min_replay_size=int(getattr(args, "threat_min_replay_size", 512)),
        batch_size=int(getattr(args, "threat_batch_size", 64)),
        target_sync_interval=int(getattr(args, "threat_target_sync_interval", 300)),
        epsilon_start=float(getattr(args, "threat_epsilon_start", 0.50)),
        epsilon_end=float(getattr(args, "threat_epsilon_end", 0.05)),
        epsilon_decay_steps=int(getattr(args, "threat_epsilon_decay_steps", 7_500)),
        feature_count=int(getattr(args, "threat_feature_count", 20)),
        hidden_size=int(getattr(args, "threat_hidden_size", 96)),
    )


def _build_meta_reward_weights(args: argparse.Namespace) -> HybridMetaRewardWeights:
    return HybridMetaRewardWeights(
        objective_complete=float(getattr(args, "meta_reward_objective_complete", 1.50)),
        phase_progress=float(getattr(args, "meta_reward_phase_progress", 0.25)),
        step_cost=float(getattr(args, "meta_reward_step_cost", 0.01)),
        premature_exit_penalty=float(getattr(args, "meta_reward_premature_exit_penalty", 1.25)),
        sector_advance=float(getattr(args, "meta_reward_sector_advance", 1.00)),
        final_sector_win=float(getattr(args, "meta_reward_final_sector_win", 25.00)),
    )


def _next_run_directory(*, root: Path, tag: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    day_prefix = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{re.escape(day_prefix)}-(\d{{2}})-")
    max_index = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        max_index = max(max_index, int(match.group(1)))
    run_id = f"{day_prefix}-{max_index + 1:02d}-{tag}"
    return root / run_id


def _build_hybrid_env(
    args: argparse.Namespace,
    *,
    effective_window_input: bool,
    pre_reset_hook: Callable[[], None] | None = None,
) -> HybridLiveEnv:
    reset_sequence = tuple(
        action.strip()
        for action in str(args.reset_sequence).split(",")
        if action.strip()
    )
    env_config = HybridLiveEnvConfig(
        step_timeout_seconds=float(args.step_timeout),
        reset_timeout_seconds=float(args.reset_timeout),
        post_action_delay_seconds=float(args.post_action_delay),
        wait_for_action_processing=bool(args.wait_for_action_processing),
        action_ack_timeout_seconds=float(args.action_ack_timeout),
        action_ack_poll_interval_seconds=float(args.action_ack_poll_interval),
        prog_slot_backoff_steps=int(args.prog_backoff_steps),
        require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
        game_tick_ms=int(args.game_tick_ms),
        disable_idle_frame_delay=bool(args.disable_idle_frame_delay),
        disable_background_motion=bool(args.disable_background_motion),
        disable_wall_animations=bool(args.disable_wall_animations),
    )
    return HybridLiveEnv.from_live_process(
        executable_name=str(args.exe),
        config=env_config,
        movement_keys=str(args.movement_keys),
        include_prog_actions=bool(args.prog_actions),
        siphon_key=str(args.siphon_key),
        reset_sequence=reset_sequence if reset_sequence else None,
        launch_process_if_missing=bool(args.launch_exe),
        focus_window_on_attach=bool(args.focus_window),
        window_targeted_input=bool(effective_window_input),
        no_enemies_mode=bool(args.no_enemies),
        pre_reset_hook=pre_reset_hook,
    )


def _format_reward_line(
    *,
    total: float,
    meta_total: float,
    threat_total: float,
    objective: ObjectivePhase,
    override: ThreatOverride,
) -> str:
    return (
        "reward total={total:+.3f} meta={meta:+.3f} threat={threat:+.3f} "
        "objective={objective} override={override}"
    ).format(
        total=total,
        meta=meta_total,
        threat=threat_total,
        objective=objective.value,
        override=override.value,
    )


def _classify_terminal_reason(
    *,
    done: bool,
    terminal_reason: str | None,
    hit_step_limit: bool,
) -> str:
    if hit_step_limit:
        return "step_limit"
    if not done:
        return "incomplete"
    normalized = str(terminal_reason or "").strip().lower()
    if not normalized:
        return "terminal_other"
    if normalized in {"state:start_screen", "state:victory"}:
        return "non_death_terminal"
    if _terminal_is_fail(normalized):
        return "fail_or_death"
    return "terminal_other"


def _terminal_reason_key(reason: str | None) -> str:
    normalized = str(reason or "").strip()
    return normalized if normalized else "none"


def _build_training_summary(results: tuple[HybridEpisodeSummary, ...]) -> dict[str, Any]:
    if not results:
        return {
            "episodes": 0,
            "done_episodes": 0,
            "non_death_terminal_episodes": 0,
            "hit_step_limit_episodes": 0,
            "done_rate": 0.0,
            "non_death_terminal_rate": 0.0,
            "hit_step_limit_rate": 0.0,
            "avg_steps": 0.0,
            "avg_total_reward": 0.0,
            "avg_meta_reward": 0.0,
            "avg_threat_reward": 0.0,
            "avg_invalid_actions": 0.0,
            "avg_premature_exit_attempts": 0.0,
            "avg_route_length": 0.0,
            "avg_route_replans": 0.0,
            "avg_phase_switches": 0.0,
            "avg_threat_active_steps": 0.0,
            "terminal_reason_counts": {},
            "terminal_classification_counts": {},
        }

    episode_count = len(results)
    terminal_reason_counts = Counter(_terminal_reason_key(item.terminal_reason) for item in results)
    terminal_classification_counts = Counter(item.terminal_classification for item in results)
    done_count = sum(1 for item in results if item.done)
    non_death_terminal_count = terminal_classification_counts.get("non_death_terminal", 0)
    hit_step_limit_count = sum(1 for item in results if item.hit_step_limit)

    return {
        "episodes": episode_count,
        "done_episodes": done_count,
        "non_death_terminal_episodes": non_death_terminal_count,
        "hit_step_limit_episodes": hit_step_limit_count,
        "done_rate": done_count / episode_count,
        "non_death_terminal_rate": non_death_terminal_count / episode_count,
        "hit_step_limit_rate": hit_step_limit_count / episode_count,
        "avg_steps": mean(item.steps for item in results),
        "avg_total_reward": mean(item.total_reward for item in results),
        "avg_meta_reward": mean(item.meta_reward_total for item in results),
        "avg_threat_reward": mean(item.threat_reward_total for item in results),
        "avg_invalid_actions": mean(item.invalid_actions for item in results),
        "avg_premature_exit_attempts": mean(item.premature_exit_attempts for item in results),
        "avg_route_length": mean(item.route_length_total for item in results),
        "avg_route_replans": mean(item.route_replans for item in results),
        "avg_phase_switches": mean(item.phase_switches for item in results),
        "avg_threat_active_steps": mean(item.threat_active_steps for item in results),
        "terminal_reason_counts": dict(sorted(terminal_reason_counts.items())),
        "terminal_classification_counts": dict(sorted(terminal_classification_counts.items())),
    }


def _print_results(results: tuple[HybridEpisodeSummary, ...]) -> None:
    print(
        "episode_id\tsteps\tdone\ttotal_reward\tmeta_reward\tthreat_reward\t"
        "invalid_actions\tpremature_exit\troute_len\treplans\tphase_switches\t"
        "threat_active_steps\thit_step_limit\tterminal_classification\tterminal_reason"
    )
    for result in results:
        print(
            "{episode}\t{steps}\t{done}\t{total:.3f}\t{meta:.3f}\t{threat:.3f}\t"
            "{invalid}\t{premature}\t{route_len}\t{replans}\t{phase_switches}\t"
            "{threat_steps}\t{hit_step_limit}\t{terminal_classification}\t{reason}".format(
                episode=result.episode_id,
                steps=result.steps,
                done=result.done,
                total=result.total_reward,
                meta=result.meta_reward_total,
                threat=result.threat_reward_total,
                invalid=result.invalid_actions,
                premature=result.premature_exit_attempts,
                route_len=result.route_length_total,
                replans=result.route_replans,
                phase_switches=result.phase_switches,
                threat_steps=result.threat_active_steps,
                hit_step_limit=result.hit_step_limit,
                terminal_classification=result.terminal_classification,
                reason=result.terminal_reason or "",
            )
        )
    summary = _build_training_summary(results)
    print(
        "\nsummary episodes={episodes} avg_steps={avg_steps:.2f} avg_total_reward={avg_total:.3f} "
        "done_rate={done_rate:.2%} non_death_terminal_rate={non_death_rate:.2%} "
        "hit_step_limit_rate={hit_limit_rate:.2%} avg_invalid_actions={avg_invalid:.2f} "
        "avg_premature_exit={avg_premature:.2f} avg_phase_switches={avg_phase_switches:.2f} "
        "avg_threat_active_steps={avg_threat_steps:.2f}".format(
            episodes=summary["episodes"],
            avg_steps=summary["avg_steps"],
            avg_total=summary["avg_total_reward"],
            done_rate=summary["done_rate"],
            non_death_rate=summary["non_death_terminal_rate"],
            hit_limit_rate=summary["hit_step_limit_rate"],
            avg_invalid=summary["avg_invalid_actions"],
            avg_premature=summary["avg_premature_exit_attempts"],
            avg_phase_switches=summary["avg_phase_switches"],
            avg_threat_steps=summary["avg_threat_active_steps"],
        )
    )


def _serialize_results(results: tuple[HybridEpisodeSummary, ...]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": item.episode_id,
            "steps": item.steps,
            "done": item.done,
            "terminal_reason": item.terminal_reason,
            "total_reward": item.total_reward,
            "meta_reward_total": item.meta_reward_total,
            "threat_reward_total": item.threat_reward_total,
            "invalid_actions": item.invalid_actions,
            "premature_exit_attempts": item.premature_exit_attempts,
            "route_length_total": item.route_length_total,
            "route_replans": item.route_replans,
            "hit_step_limit": item.hit_step_limit,
            "terminal_classification": item.terminal_classification,
            "phase_switches": item.phase_switches,
            "threat_active_steps": item.threat_active_steps,
        }
        for item in results
    ]


def _build_training_state_payload(
    *,
    results: tuple[HybridEpisodeSummary, ...],
    episodes_requested: int,
    max_steps: int,
    saved_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "episodes_requested": int(episodes_requested),
        "episodes_completed": len(results),
        "max_steps_per_episode": int(max_steps),
        "saved_at_utc": saved_at_utc or (datetime.utcnow().isoformat() + "Z"),
        "summary": _build_training_summary(results),
        "results": _serialize_results(results),
    }


def _build_hybrid_config_payload(
    args: argparse.Namespace,
    *,
    command: str,
    restore_save_source: Path | None,
    meta_reward_weights: HybridMetaRewardWeights,
) -> dict[str, Any]:
    return {
        "command": command,
        "run_tag": str(args.run_tag),
        "no_enemies_mode": bool(args.no_enemies),
        "movement_keys": str(args.movement_keys),
        "siphon_key": str(args.siphon_key),
        "prog_actions": bool(args.prog_actions),
        "restore_save_file": (
            str(restore_save_source)
            if restore_save_source is not None
            else None
        ),
        "restore_save_delay": float(args.restore_save_delay),
        "threat_trigger_distance": int(args.threat_trigger_distance),
        "warmstart_checkpoint": (
            str(args.warmstart_checkpoint)
            if getattr(args, "warmstart_checkpoint", None)
            else None
        ),
        "resume_checkpoint": (
            str(args.resume_checkpoint)
            if getattr(args, "resume_checkpoint", None)
            else None
        ),
        "meta_reward_weights": asdict(meta_reward_weights),
        "meta_config": vars(_build_meta_config(args)),
        "threat_config": vars(_build_threat_config(args)),
    }


def _run_rollouts(
    *,
    env: HybridLiveEnv,
    coordinator: HybridCoordinator,
    reward_suite: HybridRewardSuite,
    episodes: int,
    max_steps: int,
    train_meta: bool,
    train_threat: bool,
    use_meta: bool,
    use_threat: bool,
    explore_meta: bool,
    explore_threat: bool,
    tui: RunnerTuiSession,
    monitor_enabled: bool,
    print_reward_breakdown: bool,
) -> tuple[HybridEpisodeSummary, ...]:
    results: list[HybridEpisodeSummary] = []

    for _ in range(episodes):
        state = env.reset()
        coordinator.start_episode()
        episode_id = env.current_episode_id or f"episode-{len(results) + 1:05d}"
        steps = 0
        done = False
        terminal_reason: str | None = None
        total_reward = 0.0
        meta_reward_total = 0.0
        threat_reward_total = 0.0
        invalid_actions = 0
        premature_exit_attempts = 0
        route_length_total = 0
        route_replans = 0
        phase_switches = 0
        threat_active_steps = 0
        updates_applied = 0
        last_phase: ObjectivePhase | None = None
        last_target_signature: tuple[str, int, int] | None = None

        while steps < max_steps and not done:
            available_actions = env.available_actions(state)
            if not available_actions:
                available_actions = tuple(action for action in env.action_space if action != "cancel")
                if not available_actions:
                    available_actions = env.action_space
            trace = coordinator.decide(
                state=state,
                available_actions=available_actions,
                use_meta_controller=use_meta,
                use_threat_controller=use_threat,
                explore_meta=explore_meta,
                explore_threat=explore_threat,
            )
            if last_phase is not None and trace.decision.objective.phase != last_phase:
                phase_switches += 1
            last_phase = trace.decision.objective.phase
            if trace.threat_active:
                threat_active_steps += 1
            target = trace.decision.objective.target_position
            if target is not None:
                route_length_total += max(int(trace.objective_distance_before or 0), 0)
                target_signature = (
                    trace.decision.objective.phase.value,
                    int(target.x),
                    int(target.y),
                )
                if last_target_signature is not None and target_signature != last_target_signature:
                    route_replans += 1
                last_target_signature = target_signature

            threat_monitor_epsilon = (
                coordinator.threat_controller.epsilon
                if use_threat
                else None
            )
            if monitor_enabled:
                tui.wait_for_step_gate(
                    training_line=_format_monitor_training_line(
                        episode_id=episode_id,
                        step=steps + 1,
                        total_reward=total_reward,
                        meta_epsilon=coordinator.meta_controller.epsilon,
                        threat_epsilon=threat_monitor_epsilon,
                        updates_applied=updates_applied,
                        waiting=True,
                    ),
                    action_line=_format_monitor_action_line(
                        action=trace.decision.action,
                        phase=trace.decision.objective.phase,
                        target=target,
                        reason=trace.decision.reason,
                    ),
                )

            next_state, _env_reward, done, info = env.step(trace.decision.action)
            terminal_reason = str(info.get("terminal_reason") or terminal_reason or "").strip() or None
            if trace.decision.used_fallback or info.get("invalid_action_reason") is not None:
                invalid_actions += 1
            if bool(info.get("premature_exit_attempt", False)):
                premature_exit_attempts += 1
            coordinator.observe_step_result(trace=trace, info=info)

            meta_breakdown = reward_suite.compute_meta_reward(
                previous_state=state,
                current_state=next_state,
                objective_phase=trace.decision.objective.phase,
                done=done,
                info={
                    **info,
                    "objective_target": target,
                    "objective_distance_before": trace.objective_distance_before,
                },
            )
            threat_breakdown = reward_suite.compute_threat_reward(
                previous_state=state,
                current_state=next_state,
                done=done,
                threat_override=trace.decision.threat_override,
                info={
                    **info,
                    "invalid_override": bool(
                        trace.decision.used_fallback
                        and trace.decision.threat_override != ThreatOverride.ROUTE_DEFAULT
                    ),
                    "rejoined_route": bool(
                        trace.decision.threat_override == ThreatOverride.ROUTE_DEFAULT
                        and not trace.threat_active
                    ),
                },
            )
            step_reward = float(meta_breakdown.total + threat_breakdown.total)
            total_reward += step_reward
            meta_reward_total += float(meta_breakdown.total)
            threat_reward_total += float(threat_breakdown.total)

            next_allowed_phases = coordinator.allowed_meta_phases(next_state)
            next_scripted_phase, next_target = coordinator.resolve_objective_for_phase(
                state=next_state,
                phase=next_allowed_phases[0],
            )
            next_meta_features = coordinator.meta_feature_vector(
                state=next_state,
                scripted_phase=next_scripted_phase,
                target=next_target,
            )
            next_objective_distance = coordinator.objective_distance(
                state=next_state,
                target=next_target,
            )
            next_threat_features = coordinator.threat_feature_vector(
                state=next_state,
                route_action=None,
                objective_distance=next_objective_distance,
            )

            if train_meta:
                update = coordinator.meta_controller.observe(
                    features=trace.meta_features,
                    chosen_phase=trace.decision.objective.phase,
                    reward=meta_breakdown.total,
                    next_features=next_meta_features,
                    done=done,
                    next_allowed_phases=next_allowed_phases,
                )
                if update.did_update:
                    updates_applied += 1
            if train_threat:
                update = coordinator.threat_controller.observe(
                    features=trace.threat_features,
                    chosen_override=trace.decision.threat_override,
                    reward=threat_breakdown.total,
                    next_features=next_threat_features,
                    done=done,
                    next_allowed_overrides=tuple(ThreatOverride),
                )
                if update.did_update:
                    updates_applied += 1

            next_available_actions = env.available_actions(next_state)
            if monitor_enabled:
                tui.consume_manual_step_flag()
                tui.update(
                    training_line=_format_monitor_training_line(
                        episode_id=episode_id,
                        step=steps + 1,
                        reward=step_reward,
                        total_reward=total_reward,
                        meta_epsilon=coordinator.meta_controller.epsilon,
                        threat_epsilon=threat_monitor_epsilon,
                        updates_applied=updates_applied,
                        done=done,
                        terminal_reason=terminal_reason,
                    ),
                    action_line=_format_monitor_action_line(
                        action=trace.decision.action,
                        phase=trace.decision.objective.phase,
                        target=target,
                        reason=trace.decision.reason,
                    ),
                    reward_line=_format_reward_line(
                        total=step_reward,
                        meta_total=meta_breakdown.total,
                        threat_total=threat_breakdown.total,
                        objective=trace.decision.objective.phase,
                        override=trace.decision.threat_override,
                    ),
                    next_available_actions_line="next_available_actions={actions}".format(
                        actions=_format_monitor_actions(next_available_actions),
                    ),
                )
            if print_reward_breakdown:
                print(
                    _format_reward_line(
                        total=step_reward,
                        meta_total=meta_breakdown.total,
                        threat_total=threat_breakdown.total,
                        objective=trace.decision.objective.phase,
                        override=trace.decision.threat_override,
                    )
                )
            state = next_state
            steps += 1

        hit_step_limit = steps >= max_steps and not done
        terminal_classification = _classify_terminal_reason(
            done=done,
            terminal_reason=terminal_reason,
            hit_step_limit=hit_step_limit,
        )
        results.append(
            HybridEpisodeSummary(
                episode_id=episode_id,
                steps=steps,
                done=done,
                terminal_reason=terminal_reason,
                total_reward=total_reward,
                meta_reward_total=meta_reward_total,
                threat_reward_total=threat_reward_total,
                invalid_actions=invalid_actions,
                premature_exit_attempts=premature_exit_attempts,
                route_length_total=route_length_total,
                route_replans=route_replans,
                hit_step_limit=hit_step_limit,
                terminal_classification=terminal_classification,
                phase_switches=phase_switches,
                threat_active_steps=threat_active_steps,
            )
        )
    return tuple(results)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    monitor_enabled = bool(args.tui) or bool(args.external_status_file)
    effective_window_input = bool(args.window_input) or bool(args.step_through) or bool(args.tui)
    if effective_window_input and not bool(args.window_input):
        mode = "step-through" if bool(args.step_through) else "tui"
        print(
            f"{mode} enabled: using window-targeted input so actions still go to the game "
            "while monitor window has focus."
        )
    if bool(args.no_enemies):
        print("no_enemies_mode_enabled\tenemy entities will be suppressed.")
    restore_save_source = _resolve_restore_save_source_path(args)
    restore_save_delay_seconds = max(float(args.restore_save_delay), 0.0)
    restore_save_target = (
        _default_game_save_target_path()
        if restore_save_source is not None
        else None
    )
    if restore_save_source is not None and restore_save_target is not None:
        print(
            "savegame_restore_enabled\tsource={source}\ttarget={target}\tdelay_seconds={delay:.3f}".format(
                source=restore_save_source,
                target=restore_save_target,
                delay=restore_save_delay_seconds,
            )
        )

    def _restore_save_before_reset() -> None:
        if restore_save_source is None or restore_save_target is None:
            return
        _restore_selected_save_file(
            source_path=restore_save_source,
            target_path=restore_save_target,
        )
        print(
            "savegame_restored_before_reset\tsource={source}\ttarget={target}\tdelay_seconds={delay:.3f}".format(
                source=restore_save_source,
                target=restore_save_target,
                delay=restore_save_delay_seconds,
            )
        )
        if restore_save_delay_seconds > 0:
            time.sleep(restore_save_delay_seconds)

    env: HybridLiveEnv | None = None
    tui = RunnerTuiSession(
        executable_name=str(args.exe),
        enabled=monitor_enabled,
        interval_seconds=float(args.tui_interval),
        step_through=bool(args.step_through),
        launch_monitor=bool(args.tui),
        external_status_file=(str(args.external_status_file) if args.external_status_file else None),
        external_control_file=(str(args.external_control_file) if args.external_control_file else None),
    )
    try:
        env = _build_hybrid_env(
            args,
            effective_window_input=effective_window_input,
            pre_reset_hook=(
                _restore_save_before_reset
                if restore_save_source is not None and restore_save_target is not None
                else None
            ),
        )
        tui.start()

        command = str(args.command)
        meta_reward_weights = _build_meta_reward_weights(args)
        reward_suite = HybridRewardSuite(meta_weights=meta_reward_weights)
        coordinator_config = HybridCoordinatorConfig(
            threat_trigger_distance=max(int(args.threat_trigger_distance), 1),
            exit_after_siphons_when_scripted=False,
        )
        coordinator = HybridCoordinator(
            meta_controller=MetaControllerDQN(config=_build_meta_config(args), seed=args.seed),
            threat_controller=ThreatControllerDRQN(config=_build_threat_config(args), seed=args.seed),
            movement_controller=AStarMovementController(),
            config=coordinator_config,
        )

        if command == "eval-hybrid":
            loaded_meta, loaded_threat, _bundle_config, _training_state = HybridCheckpointManager.load_bundle(
                run_directory=str(args.checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=loaded_threat,
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )
        elif getattr(args, "resume_checkpoint", None):
            loaded_meta, loaded_threat, _bundle_config, _training_state = HybridCheckpointManager.load_bundle(
                run_directory=str(args.resume_checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=loaded_threat,
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )
        elif command == "train-full-hierarchical" and getattr(args, "warmstart_checkpoint", None):
            loaded_meta, _loaded_threat, _bundle_config, _training_state = HybridCheckpointManager.load_bundle(
                run_directory=str(args.warmstart_checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=ThreatControllerDRQN(config=_build_threat_config(args), seed=args.seed),
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )

        if command == "movement-test":
            train_meta = False
            train_threat = False
            use_meta = False
            use_threat = False
            explore_meta = False
            explore_threat = False
        elif command == "train-meta-no-enemies":
            train_meta = True
            train_threat = False
            use_meta = True
            use_threat = False
            explore_meta = True
            explore_threat = False
        elif command == "train-full-hierarchical":
            train_meta = False
            train_threat = True
            use_meta = True
            use_threat = True
            explore_meta = True
            explore_threat = True
        elif command == "eval-hybrid":
            train_meta = False
            train_threat = False
            use_meta = True
            use_threat = True
            explore_meta = False
            explore_threat = False
        else:  # pragma: no cover - argparse guards command values.
            raise ValueError(f"Unsupported command: {command}")

        results: list[HybridEpisodeSummary] = []
        if command == "train-full-hierarchical":
            freeze_episodes = max(int(args.meta_freeze_episodes), 0)
            if freeze_episodes > 0:
                warmup_results = _run_rollouts(
                    env=env,
                    coordinator=coordinator,
                    reward_suite=reward_suite,
                    episodes=min(int(args.episodes), freeze_episodes),
                    max_steps=int(args.max_steps),
                    train_meta=False,
                    train_threat=True,
                    use_meta=True,
                    use_threat=True,
                    explore_meta=True,
                    explore_threat=True,
                    tui=tui,
                    monitor_enabled=monitor_enabled,
                    print_reward_breakdown=bool(args.print_reward_breakdown),
                )
                results.extend(warmup_results)

            remaining = max(int(args.episodes) - len(results), 0)
            if remaining > 0:
                finetune_meta = bool(args.joint_finetune)
                finetune_results = _run_rollouts(
                    env=env,
                    coordinator=coordinator,
                    reward_suite=reward_suite,
                    episodes=remaining,
                    max_steps=int(args.max_steps),
                    train_meta=finetune_meta,
                    train_threat=True,
                    use_meta=True,
                    use_threat=True,
                    explore_meta=True,
                    explore_threat=True,
                    tui=tui,
                    monitor_enabled=monitor_enabled,
                    print_reward_breakdown=bool(args.print_reward_breakdown),
                )
                results.extend(finetune_results)
        else:
            rollout_results = _run_rollouts(
                env=env,
                coordinator=coordinator,
                reward_suite=reward_suite,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                train_meta=train_meta,
                train_threat=train_threat,
                use_meta=use_meta,
                use_threat=use_threat,
                explore_meta=explore_meta,
                explore_threat=explore_threat,
                tui=tui,
                monitor_enabled=monitor_enabled,
                print_reward_breakdown=bool(args.print_reward_breakdown),
            )
            results.extend(rollout_results)

        finalized = tuple(results)
        _print_results(finalized)

        if command in {"train-meta-no-enemies", "train-full-hierarchical"}:
            checkpoint_root = Path(str(args.checkpoint_root))
            run_directory = _next_run_directory(root=checkpoint_root, tag=str(args.run_tag))
            bundle = HybridCheckpointManager.save_bundle(
                run_directory=run_directory,
                meta_controller=coordinator.meta_controller,
                threat_controller=coordinator.threat_controller,
                hybrid_config=_build_hybrid_config_payload(
                    args,
                    command=command,
                    restore_save_source=restore_save_source,
                    meta_reward_weights=meta_reward_weights,
                ),
                training_state=_build_training_state_payload(
                    results=finalized,
                    episodes_requested=int(args.episodes),
                    max_steps=int(args.max_steps),
                ),
            )
            print(f"hybrid_checkpoint_saved\t{bundle.run_directory}")
    finally:
        if env is not None:
            env.close()
        tui.close()


if __name__ == "__main__":
    main()

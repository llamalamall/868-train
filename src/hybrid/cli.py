"""CLI argument helpers for Hybrid workflows."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.env.runner_common import game_tick_ms_arg, resolve_restore_save_source_path
from src.hybrid.checkpoint import HybridCheckpointManager
from src.hybrid.env import HybridLiveEnv, HybridLiveEnvConfig
from src.hybrid.meta_controller import MetaDQNConfig
from src.hybrid.rewards import HybridMetaRewardWeights, HybridThreatRewardWeights
from src.hybrid.threat_controller import ThreatDRQNConfig

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
        default=False,
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
        type=game_tick_ms_arg,
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
        "--post-action-delay-backoff",
        type=float,
        default=0.02,
        help="Additional post-action delay (seconds) added per active ack-timeout backoff level.",
    )
    parser.add_argument(
        "--action-ack-timeout-backoff",
        type=float,
        default=0.10,
        help="Additional action-ack timeout (seconds) added per active ack-timeout backoff level.",
    )
    parser.add_argument(
        "--action-ack-backoff-max-level",
        type=int,
        default=3,
        help="Maximum adaptive ack-timeout backoff level.",
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
        "--phase-lock-min-steps",
        type=int,
        default=6,
        help="Minimum steps to keep a resource/exit phase lock before allowing a phase switch.",
    )
    parser.add_argument(
        "--target-stall-release-steps",
        type=int,
        default=4,
        help="Consecutive non-progress steps before releasing the current locked objective.",
    )
    parser.add_argument(
        "--print-reward-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-step split reward components.",
    )
    parser.add_argument(
        "--victory-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the debugger-backed victory transition monitor during hybrid runs.",
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
    parser.add_argument(
        "--meta-reward-currency-gain",
        type=float,
        default=0.10,
        help="Meta reward: coefficient applied to positive currency gain.",
    )
    parser.add_argument(
        "--meta-reward-energy-gain",
        type=float,
        default=0.10,
        help="Meta reward: coefficient applied to positive energy gain.",
    )
    parser.add_argument(
        "--meta-reward-score-gain",
        type=float,
        default=0.02,
        help="Meta reward: coefficient applied to positive score gain.",
    )
    parser.add_argument(
        "--meta-reward-prog-gain",
        type=float,
        default=1.50,
        help="Meta reward: coefficient applied to positive prog inventory gain.",
    )
    parser.add_argument(
        "--meta-reward-step-limit-penalty",
        type=float,
        default=5.00,
        help="Meta reward: penalty applied on the final transition of a step-capped episode.",
    )
    parser.add_argument(
        "--meta-reward-stagnation-penalty",
        type=float,
        default=0.05,
        help="Meta reward: penalty applied when the current objective stalls past the grace window.",
    )
    parser.add_argument(
        "--meta-reward-stagnation-grace-steps",
        type=int,
        default=3,
        help="Meta reward: consecutive non-progress steps allowed before stagnation penalty starts.",
    )
    parser.add_argument(
        "--threat-reward-survival",
        type=float,
        default=0.05,
        help="Threat reward: survival bonus applied on non-terminal threat-relevant steps.",
    )
    parser.add_argument(
        "--threat-reward-damage-taken-penalty",
        type=float,
        default=0.35,
        help="Threat reward: penalty coefficient applied to damage taken.",
    )
    parser.add_argument(
        "--threat-reward-fail-penalty",
        type=float,
        default=2.50,
        help="Threat reward: penalty applied on fail/death terminals.",
    )
    parser.add_argument(
        "--threat-reward-route-rejoin-bonus",
        type=float,
        default=0.15,
        help="Threat reward: bonus for route rejoin events after a threat override.",
    )
    parser.add_argument(
        "--threat-reward-invalid-override-penalty",
        type=float,
        default=0.10,
        help="Threat reward: penalty for invalid threat overrides.",
    )
    parser.add_argument(
        "--threat-reward-enemy-damaged",
        type=float,
        default=0.20,
        help="Threat reward: positive coefficient applied to enemy HP damage dealt.",
    )
    parser.add_argument(
        "--threat-reward-enemy-cleared",
        type=float,
        default=0.75,
        help="Threat reward: positive coefficient applied when an enemy is removed.",
    )
    parser.add_argument(
        "--threat-reward-spawn-debt-penalty",
        type=float,
        default=0.15,
        help="Threat reward: penalty coefficient applied to risky siphon spawn debt or enemy-count growth.",
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
    parser.add_argument(
        "--meta-phase-override-credit-mode",
        choices=("executed", "skip_overridden"),
        default="skip_overridden",
        help="How to handle meta updates when the coordinator executes a different phase than the meta policy requested.",
    )


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
    if float(args.post_action_delay_backoff) < 0:
        parser.error("--post-action-delay-backoff must be >= 0.")
    if float(args.action_ack_timeout_backoff) < 0:
        parser.error("--action-ack-timeout-backoff must be >= 0.")
    if int(args.action_ack_backoff_max_level) < 0:
        parser.error("--action-ack-backoff-max-level must be >= 0.")
    if float(args.restore_save_delay) < 0:
        parser.error("--restore-save-delay must be >= 0.")
    if int(args.prog_backoff_steps) < 0:
        parser.error("--prog-backoff-steps must be >= 0.")
    if int(args.phase_lock_min_steps) < 0:
        parser.error("--phase-lock-min-steps must be >= 0.")
    if int(args.target_stall_release_steps) < 0:
        parser.error("--target-stall-release-steps must be >= 0.")
    if int(getattr(args, "meta_reward_stagnation_grace_steps", 0)) < 0:
        parser.error("--meta-reward-stagnation-grace-steps must be >= 0.")
    if bool(args.step_through) and not bool(args.tui):
        parser.error("--step-through requires --tui.")
    restore_source = resolve_restore_save_source_path(args)
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
    checkpoint_args: list[tuple[str, str, tuple[str, ...] | None]] = []
    if args.command == "eval-hybrid" and getattr(args, "checkpoint", None):
        checkpoint_args.append(
            ("--checkpoint", str(args.checkpoint), HybridCheckpointManager.BUNDLE_REQUIRED_FILES)
        )
    if getattr(args, "resume_checkpoint", None):
        checkpoint_args.append(
            (
                "--resume-checkpoint",
                str(args.resume_checkpoint),
                HybridCheckpointManager.BUNDLE_REQUIRED_FILES,
            )
        )
    if args.command == "train-full-hierarchical" and getattr(args, "warmstart_checkpoint", None):
        checkpoint_args.append(
            (
                "--warmstart-checkpoint",
                str(args.warmstart_checkpoint),
                HybridCheckpointManager.WARMSTART_REQUIRED_FILES,
            )
        )
    for option_name, path_value, required_files in checkpoint_args:
        try:
            HybridCheckpointManager.validate_bundle_directory(
                run_directory=path_value,
                required_files=required_files,
                label=option_name,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            parser.error(str(error))


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
        currency_gain=float(getattr(args, "meta_reward_currency_gain", 0.10)),
        energy_gain=float(getattr(args, "meta_reward_energy_gain", 0.10)),
        score_gain=float(getattr(args, "meta_reward_score_gain", 0.02)),
        prog_gain=float(getattr(args, "meta_reward_prog_gain", 1.50)),
        step_limit_penalty=float(getattr(args, "meta_reward_step_limit_penalty", 5.00)),
        stagnation_penalty=float(getattr(args, "meta_reward_stagnation_penalty", 0.05)),
        stagnation_grace_steps=int(getattr(args, "meta_reward_stagnation_grace_steps", 3)),
    )


def _build_threat_reward_weights(args: argparse.Namespace) -> HybridThreatRewardWeights:
    return HybridThreatRewardWeights(
        survival=float(getattr(args, "threat_reward_survival", 0.05)),
        damage_taken_penalty=float(getattr(args, "threat_reward_damage_taken_penalty", 0.35)),
        fail_penalty=float(getattr(args, "threat_reward_fail_penalty", 2.50)),
        route_rejoin_bonus=float(getattr(args, "threat_reward_route_rejoin_bonus", 0.15)),
        invalid_override_penalty=float(
            getattr(args, "threat_reward_invalid_override_penalty", 0.10)
        ),
        enemy_damaged=float(getattr(args, "threat_reward_enemy_damaged", 0.20)),
        enemy_cleared=float(getattr(args, "threat_reward_enemy_cleared", 0.75)),
        spawn_debt_penalty=float(getattr(args, "threat_reward_spawn_debt_penalty", 0.15)),
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
        post_action_delay_backoff_seconds=float(args.post_action_delay_backoff),
        action_ack_timeout_backoff_seconds=float(args.action_ack_timeout_backoff),
        action_ack_backoff_max_level=int(args.action_ack_backoff_max_level),
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


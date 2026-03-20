"""CLI runner for DQN training/evaluation against the live game environment."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.agent.dqn_agent import DQNAgent, DQNConfig
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import (
    _build_action_config,
    _build_reward_config,
    _build_reward_fn,
    format_reward_breakdown_line,
)
from src.env.runner_tui import RunnerTuiSession
from src.training.rewards import RewardWeights
from src.training.train import LearningEpisodeRolloutResult, run_dqn_training


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


_FAIL_TERMINAL_REASON_TOKENS: tuple[str, ...] = ("fail", "loss", "dead", "start_screen")
_APP_SAVE_FOLDER_NAME = "868-HACK"
_APP_SAVE_FILE_NAME = "savegame_868"


def _default_game_save_target_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / _APP_SAVE_FOLDER_NAME / _APP_SAVE_FILE_NAME
    return Path.home() / "AppData" / "Roaming" / _APP_SAVE_FOLDER_NAME / _APP_SAVE_FILE_NAME


def _resolve_restore_save_source_path(args: argparse.Namespace) -> Path | None:
    if not args.restore_save_file:
        return None
    return Path(str(args.restore_save_file)).expanduser().resolve()


def _reason_indicates_fail_terminal(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    normalized = reason.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _FAIL_TERMINAL_REASON_TOKENS)


def _event_indicates_fail_terminal(event: dict[str, Any]) -> bool:
    if not bool(event.get("done", False)):
        return False
    return _reason_indicates_fail_terminal(event.get("terminal_reason"))


def _restore_selected_save_file(*, source_path: Path, target_path: Path) -> None:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DQN training/evaluation episodes against live game env."
    )
    default_weights = RewardWeights()

    parser.add_argument(
        "--mode",
        choices=("train", "eval"),
        default="train",
        help="train: update model and write checkpoint. eval: run greedy policy from a checkpoint.",
    )
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument(
        "--launch-exe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch --exe when not already running before attempting attach.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
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
        help="Key used by the siphon action.",
    )
    parser.add_argument(
        "--prog-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prog-slot actions (prog_slot_1..prog_slot_10 mapped to 1..0).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path for load/save. In train mode, if omitted, an automatic path under "
            "artifacts/checkpoints is used. In eval mode, this is required."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="In train mode, also save periodic checkpoints every N episodes (0 disables).",
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
        "--reset-sequence",
        default="confirm",
        nargs="?",
        const="",
        help="Comma-separated reset actions; pass without a value to disable reset key sequence.",
    )
    parser.add_argument(
        "--focus-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Focus the game window during attach/reacquire (default: enabled).",
    )
    parser.add_argument(
        "--window-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use window-targeted PostMessage input instead of global SendInput.",
    )
    parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch live state monitor TUI in a separate console window.",
    )
    parser.add_argument(
        "--tui-interval",
        type=float,
        default=0.5,
        help="Polling interval for the live TUI (seconds).",
    )
    parser.add_argument(
        "--external-status-file",
        default=None,
        help=(
            "Optional JSON file path to receive live training/action status lines. "
            "Useful for GUI monitoring without launching the external TUI."
        ),
    )
    parser.add_argument(
        "--external-control-file",
        default=None,
        help=(
            "Optional JSON file path for external pause/step/resume session controls. "
            "Useful for GUI monitoring without launching the external TUI."
        ),
    )
    parser.add_argument(
        "--step-through",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause before each action and wait for Enter in the TUI to advance.",
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
        help="Target game loop tick size in milliseconds (1..16). Lower values speed up gameplay.",
    )
    parser.add_argument(
        "--disable-idle-frame-delay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch SDL_Delay(1) in the main loop to SDL_Delay(0) for faster runtime pacing.",
    )
    parser.add_argument(
        "--disable-background-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable animated background motion effect via runtime flag patching.",
    )
    parser.add_argument(
        "--disable-wall-animations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze wall/tile palette animation counter in the renderer.",
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
        help="Poll for state evidence that each action was processed before finalizing a step.",
    )
    parser.add_argument(
        "--action-ack-timeout",
        type=float,
        default=0.35,
        help="Max additional wait time (seconds) to observe post-action state change.",
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
        help="Require reset() to observe a non-terminal state before starting steps.",
    )
    parser.add_argument(
        "--no-enemies",
        action="store_true",
        default=False,
        help=(
            "Suppress active enemy entities each step so training can focus on "
            "phase progression without combat pressure."
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Q-model SGD learning rate.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=20_000,
        help="Replay buffer capacity.",
    )
    parser.add_argument(
        "--min-replay-size",
        type=int,
        default=256,
        help="Minimum replay items before updates begin.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Replay minibatch size.",
    )
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=500,
        help="Number of optimization steps between target syncs.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.8,
        help="Initial epsilon for exploration.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon for exploration.",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=5_000,
        help="Linear epsilon decay horizon in env steps.",
    )

    parser.add_argument(
        "--reward-survival",
        type=float,
        default=default_weights.survival,
        help="Survival reward applied for non-terminal steps.",
    )
    parser.add_argument(
        "--reward-step-penalty",
        type=float,
        default=default_weights.step_penalty,
        help="Per-step penalty magnitude applied each transition.",
    )
    parser.add_argument(
        "--reward-health-delta",
        type=float,
        default=default_weights.health_delta,
        help="Weight multiplied by (current_health - previous_health).",
    )
    parser.add_argument(
        "--reward-currency-delta",
        type=float,
        default=default_weights.currency_delta,
        help="Weight multiplied by (current_currency - previous_currency).",
    )
    parser.add_argument(
        "--reward-energy-delta",
        type=float,
        default=default_weights.energy_delta,
        help="Weight multiplied by (current_energy - previous_energy).",
    )
    parser.add_argument(
        "--reward-score-delta",
        type=float,
        default=default_weights.score_delta,
        help="Weight multiplied by (current_score - previous_score) when score is available.",
    )
    parser.add_argument(
        "--reward-siphon-collected",
        type=float,
        default=default_weights.siphon_collected,
        help="Reward per siphon removed from map.",
    )
    parser.add_argument(
        "--reward-enemy-damaged",
        type=float,
        default=default_weights.enemy_damaged,
        help="Reward per enemy HP point reduced when an enemy survives the step.",
    )
    parser.add_argument(
        "--reward-enemy-cleared",
        type=float,
        default=default_weights.enemy_cleared,
        help="Reward per live enemy removed from map.",
    )
    parser.add_argument(
        "--reward-phase-progress",
        type=float,
        default=default_weights.phase_progress,
        help="Weight for progress toward active objective (siphon->high-priority siphon target->enemy->exit).",
    )
    parser.add_argument(
        "--reward-backtrack-penalty",
        type=float,
        default=default_weights.backtrack_penalty,
        help="Penalty weight for increased distance from the active objective.",
    )
    parser.add_argument(
        "--reward-map-clear-bonus",
        type=float,
        default=default_weights.map_clear_bonus,
        help="Bonus when player reaches exit after siphons/enemies are cleared and high-priority targets are siphoned.",
    )
    parser.add_argument(
        "--reward-premature-exit-penalty",
        type=float,
        default=default_weights.premature_exit_penalty,
        help="Penalty when stepping onto exit before objectives are complete.",
    )
    parser.add_argument(
        "--reward-invalid-action-penalty",
        type=float,
        default=default_weights.invalid_action_penalty,
        help="Penalty when action appears ineffective/invalid.",
    )
    parser.add_argument(
        "--reward-fail-penalty",
        type=float,
        default=default_weights.fail_penalty,
        help="Terminal fail penalty magnitude (applied as negative).",
    )
    parser.add_argument(
        "--reward-safe-tile-bonus",
        type=float,
        default=default_weights.safe_tile_bonus,
        help="Bonus on steps where the current tile is not threatened by enemies.",
    )
    parser.add_argument(
        "--reward-danger-tile-penalty",
        type=float,
        default=default_weights.danger_tile_penalty,
        help="Penalty on steps where the current tile is threatened by enemies.",
    )
    parser.add_argument(
        "--reward-resource-proximity",
        type=float,
        default=default_weights.resource_proximity,
        help="Reward per tile of progress toward nearest harvest target.",
    )
    parser.add_argument(
        "--reward-prog-collected-base",
        type=float,
        default=default_weights.prog_collected_base,
        help="Base reward for each newly collected prog before priority bonus.",
    )
    parser.add_argument(
        "--reward-points-collected",
        type=float,
        default=default_weights.points_collected,
        help="Reward per map-point unit removed from available board points.",
    )
    parser.add_argument(
        "--reward-damage-taken-penalty",
        type=float,
        default=default_weights.damage_taken_penalty,
        help="Penalty multiplier applied to negative health deltas.",
    )
    parser.add_argument(
        "--reward-sector-advance",
        type=float,
        default=default_weights.sector_advance,
        help="Reward per positive sector index transition.",
    )
    parser.add_argument(
        "--reward-clip-abs",
        type=float,
        default=5.0,
        help="Absolute value used to clip final reward per step.",
    )
    parser.add_argument(
        "--print-reward-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-step reward component breakdown during execution.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episodes < 1:
        parser.error("--episodes must be >= 1.")
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1.")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0.")
    if args.mode == "eval" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode=eval.")
    if float(args.restore_save_delay) < 0:
        parser.error("--restore-save-delay must be >= 0.")
    restore_source = _resolve_restore_save_source_path(args)
    if restore_source is not None:
        if not restore_source.exists():
            parser.error(f"--restore-save-file not found: {restore_source}.")
        if not restore_source.is_file():
            parser.error(f"--restore-save-file must be a file: {restore_source}.")


def _build_dqn_config(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        gamma=float(args.gamma),
        learning_rate=float(args.learning_rate),
        replay_capacity=int(args.replay_capacity),
        min_replay_size=int(args.min_replay_size),
        batch_size=int(args.batch_size),
        target_sync_interval=int(args.target_sync_interval),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay_steps=int(args.epsilon_decay_steps),
    )


def _default_checkpoint_path(*, now_utc: datetime | None = None) -> Path:
    stamp = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / "checkpoints" / f"dqn-{stamp}.json"


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(str(args.checkpoint))
    return _default_checkpoint_path()


def _periodic_checkpoint_path(base_checkpoint: Path, *, episode_index: int) -> Path:
    return base_checkpoint.with_name(
        f"{base_checkpoint.stem}.ep{episode_index:05d}{base_checkpoint.suffix or '.json'}"
    )


def _build_train_metadata(
    *,
    args: argparse.Namespace,
    episode_count: int,
    save_kind: str,
) -> dict[str, Any]:
    return {
        "mode": "train",
        "episodes_requested": int(args.episodes),
        "episodes_completed": int(episode_count),
        "max_steps_per_episode": int(args.max_steps),
        "save_kind": save_kind,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _print_results(results: tuple[LearningEpisodeRolloutResult, ...]) -> None:
    print("episode_id\tsteps\tdone\ttotal_reward\tupdates\tloss\tepsilon\tterminal_reason")
    for result in results:
        loss_str = "" if result.last_loss is None else f"{result.last_loss:.6f}"
        print(
            f"{result.episode_id}\t{result.steps}\t{result.done}\t"
            f"{result.total_reward:.3f}\t{result.updates_applied}\t"
            f"{loss_str}\t{result.epsilon:.4f}\t{result.terminal_reason or ''}"
        )

    avg_steps = mean(result.steps for result in results)
    avg_reward = mean(result.total_reward for result in results)
    done_rate = sum(1 for result in results if result.done) / len(results)
    avg_updates = mean(result.updates_applied for result in results)
    print(
        f"\nsummary episodes={len(results)} avg_steps={avg_steps:.2f} "
        f"avg_reward={avg_reward:.3f} done_rate={done_rate:.2%} avg_updates={avg_updates:.2f}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if bool(args.step_through) and not bool(args.tui):
        parser.error("--step-through requires --tui.")
    monitor_enabled = bool(args.tui) or bool(args.external_status_file)
    effective_window_input = bool(args.window_input) or bool(args.step_through) or bool(args.tui)
    if effective_window_input and not bool(args.window_input):
        mode = "step-through" if bool(args.step_through) else "tui"
        print(
            f"{mode} enabled: using window-targeted input so actions still go to the game "
            "while the TUI window has focus."
        )
    if bool(args.no_enemies):
        print("no_enemies_mode_enabled\tenemy entities will be suppressed.")

    reset_sequence = tuple(
        action.strip() for action in str(args.reset_sequence).split(",") if action.strip()
    )
    reward_config = _build_reward_config(args)
    reward_fn = _build_reward_fn(
        reward_config=reward_config,
        print_breakdown=bool(args.print_reward_breakdown),
    )
    env: GameEnv | None = None
    tui = RunnerTuiSession(
        executable_name=str(args.exe),
        runner_module="src.env.dqn_policy_runner",
        enabled=monitor_enabled,
        interval_seconds=float(args.tui_interval),
        step_through=bool(args.step_through),
        launch_monitor=bool(args.tui),
        external_status_file=(str(args.external_status_file) if args.external_status_file else None),
        external_control_file=(str(args.external_control_file) if args.external_control_file else None),
    )

    checkpoint_path = _resolve_checkpoint_path(args)
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

    try:
        env = GameEnv.from_live_process(
            executable_name=args.exe,
            config=GameEnvConfig(
                step_timeout_seconds=args.step_timeout,
                reset_timeout_seconds=args.reset_timeout,
                post_action_poll_delay_seconds=max(float(args.post_action_delay), 0.0),
                wait_for_action_processing=bool(args.wait_for_action_processing),
                action_ack_timeout_seconds=max(float(args.action_ack_timeout), 0.0),
                action_ack_poll_interval_seconds=max(float(args.action_ack_poll_interval), 0.0),
                prog_slot_backoff_steps=max(int(args.prog_backoff_steps), 0),
                require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
            ),
            reset_sequence=reset_sequence if reset_sequence else None,
            launch_process_if_missing=bool(args.launch_exe),
            focus_window_on_attach=bool(args.focus_window),
            window_targeted_input=effective_window_input,
            action_config=_build_action_config(
                args.movement_keys,
                include_prog_actions=bool(args.prog_actions),
                siphon_key=str(args.siphon_key),
            ),
            pre_reset_hook=(
                _restore_save_before_reset
                if restore_save_source is not None and restore_save_target is not None
                else None
            ),
            reward_fn=reward_fn,
            game_tick_ms=int(args.game_tick_ms),
            no_enemies_mode=bool(args.no_enemies),
            disable_idle_frame_delay=bool(args.disable_idle_frame_delay),
            disable_background_motion=bool(args.disable_background_motion),
            disable_wall_animations=bool(args.disable_wall_animations),
        )
        tui.start()

        def _on_step(event: dict[str, Any]) -> None:
            if not monitor_enabled:
                return
            tui.consume_manual_step_flag()
            tui.update(
                training_line=(
                    "episode={episode} step={step} reward={reward:.3f} total={total:.3f} "
                    "epsilon={epsilon:.4f} updates={updates} done={done} terminal={terminal}".format(
                        episode=event.get("episode_id"),
                        step=int(event.get("step_index", 0)) + 1,
                        reward=float(event.get("reward", 0.0)),
                        total=float(event.get("total_reward", 0.0)),
                        epsilon=float(event.get("epsilon", 0.0)),
                        updates=int(event.get("updates_applied", 0)),
                        done=bool(event.get("done", False)),
                        terminal=event.get("terminal_reason") or "-",
                    )
                ),
                action_line="action={action} reason={reason} loss={loss}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "dqn_select_action",
                    loss=(
                        "{0:.6f}".format(float(event.get("last_loss")))
                        if event.get("last_loss") is not None
                        else "-"
                    ),
                ),
                reward_line=format_reward_breakdown_line(event),
                next_available_actions_line="next_available_actions={actions}".format(
                    actions=_format_monitor_actions(event.get("next_available_actions")),
                ),
            )

        def _on_before_step(event: dict[str, Any]) -> None:
            tui.wait_for_step_gate(
                training_line=(
                    "episode={episode} step={step} total={total:.3f} "
                    "epsilon={epsilon:.4f} updates={updates} waiting=step".format(
                        episode=event.get("episode_id"),
                        step=int(event.get("step_index", 0)) + 1,
                        total=float(event.get("total_reward", 0.0)),
                        epsilon=float(event.get("epsilon", 0.0)),
                        updates=int(event.get("updates_applied", 0)),
                    )
                ),
                action_line="action={action} reason={reason}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "dqn_select_action",
                ),
            )

        assert env is not None
        if checkpoint_path.exists():
            agent = DQNAgent.load_checkpoint(checkpoint_path)
        elif args.mode == "train":
            policy_actions = tuple(action for action in env.action_space if action != "cancel")
            if not policy_actions:
                policy_actions = env.action_space
            agent = DQNAgent(
                action_space=policy_actions,
                config=_build_dqn_config(args),
                seed=args.seed,
            )
        else:
            parser.error(f"Checkpoint not found: {checkpoint_path}.")
            return

        overlap = set(agent.action_space) & set(env.action_space)
        if not overlap:
            parser.error(
                "No overlapping actions between loaded DQN checkpoint and environment action_space."
            )
            return

        results: list[LearningEpisodeRolloutResult] = []
        if args.mode == "train":
            for episode_index in range(1, int(args.episodes) + 1):
                episode_result = run_dqn_training(
                    env=env,
                    agent=agent,
                    episodes=1,
                    max_steps_per_episode=int(args.max_steps),
                    explore=True,
                    learn=True,
                    before_step_callback=_on_before_step if monitor_enabled else None,
                    step_callback=(
                        _on_step
                        if monitor_enabled or restore_save_source is not None
                        else None
                    ),
                )[0]
                results.append(episode_result)

                reached_step_limit_without_terminal = (
                    not bool(episode_result.done)
                    and int(episode_result.steps) >= int(args.max_steps)
                )
                if reached_step_limit_without_terminal:
                    print(
                        "episode_step_limit_reached\tepisode={episode}\tsteps={steps}\t"
                        "action=cancel_then_reset".format(
                            episode=episode_result.episode_id,
                            steps=episode_result.steps,
                        )
                    )
                    if "cancel" in env.action_space:
                        try:
                            env.step("cancel")
                        except Exception as error:
                            print(
                                "episode_step_limit_cancel_failed\tepisode={episode}\terror={error}".format(
                                    episode=episode_result.episode_id,
                                    error=error,
                                )
                            )
                    else:
                        print(
                            "episode_step_limit_cancel_unavailable\tepisode={episode}".format(
                                episode=episode_result.episode_id,
                            )
                        )

                    if restore_save_source is not None and restore_save_target is not None:
                        try:
                            _restore_save_before_reset()
                        except Exception as error:
                            print(
                                "episode_step_limit_restore_failed\tepisode={episode}\terror={error}".format(
                                    episode=episode_result.episode_id,
                                    error=error,
                                )
                            )

                    try:
                        env.reset()
                    except Exception as error:
                        print(
                            "episode_step_limit_reset_failed\tepisode={episode}\terror={error}".format(
                                episode=episode_result.episode_id,
                                error=error,
                            )
                        )

                if args.checkpoint_every and episode_index % int(args.checkpoint_every) == 0:
                    periodic_path = _periodic_checkpoint_path(
                        checkpoint_path, episode_index=episode_index
                    )
                    agent.save_checkpoint(
                        periodic_path,
                        metadata=_build_train_metadata(
                            args=args,
                            episode_count=episode_index,
                            save_kind="periodic",
                        ),
                    )
                    print(f"checkpoint_saved\t{periodic_path}")

            final_checkpoint = agent.save_checkpoint(
                checkpoint_path,
                metadata=_build_train_metadata(
                    args=args,
                    episode_count=len(results),
                    save_kind="final",
                ),
            )
            print(f"checkpoint_saved\t{final_checkpoint}")
        else:
            results = list(
                run_dqn_training(
                    env=env,
                    agent=agent,
                    episodes=int(args.episodes),
                    max_steps_per_episode=int(args.max_steps),
                    explore=False,
                    learn=False,
                    before_step_callback=_on_before_step if monitor_enabled else None,
                    step_callback=(
                        _on_step
                        if monitor_enabled or restore_save_source is not None
                        else None
                    ),
                )
            )
    finally:
        if env is not None:
            env.close()
        tui.close()

    _print_results(tuple(results))


if __name__ == "__main__":
    main()

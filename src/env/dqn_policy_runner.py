"""CLI runner for DQN training/evaluation against the live game environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.agent.dqn_agent import DQNAgent, DQNConfig
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import _build_action_config, _build_reward_config, _build_reward_fn
from src.env.runner_tui import RunnerTuiSession
from src.training.rewards import RewardWeights
from src.training.train import LearningEpisodeRolloutResult, run_dqn_training


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
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--movement-keys",
        choices=("arrows", "wasd", "numpad"),
        default="arrows",
        help="Movement key mapping profile.",
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
        default=False,
        help="Use window-targeted PostMessage input instead of global SendInput.",
    )
    parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch live state monitor TUI in a separate console window.",
    )
    parser.add_argument(
        "--tui-interval",
        type=float,
        default=0.5,
        help="Polling interval for the live TUI (seconds).",
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
        "--reset-timeout",
        type=float,
        default=15.0,
        help="Reset watchdog timeout in seconds.",
    )
    parser.add_argument(
        "--require-non-terminal-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reset() to observe a non-terminal state before starting steps.",
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
        default=0.01,
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
        default=200,
        help="Number of optimization steps between target syncs.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
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
        default=10_000,
        help="Linear epsilon decay horizon in env steps.",
    )

    parser.add_argument(
        "--reward-survival",
        type=float,
        default=default_weights.survival,
        help="Survival reward applied for non-terminal steps.",
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
        "--reward-fail-penalty",
        type=float,
        default=default_weights.fail_penalty,
        help="Terminal fail penalty magnitude (applied as negative).",
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
    effective_window_input = bool(args.window_input) or bool(args.step_through) or bool(args.tui)
    if effective_window_input and not bool(args.window_input):
        mode = "step-through" if bool(args.step_through) else "tui"
        print(
            f"{mode} enabled: using window-targeted input so actions still go to the game "
            "while the TUI window has focus."
        )

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
        enabled=bool(args.tui),
        interval_seconds=float(args.tui_interval),
        step_through=bool(args.step_through),
    )

    checkpoint_path = _resolve_checkpoint_path(args)
    try:
        env = GameEnv.from_live_process(
            executable_name=args.exe,
            config=GameEnvConfig(
                step_timeout_seconds=args.step_timeout,
                reset_timeout_seconds=args.reset_timeout,
                require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
            ),
            reset_sequence=reset_sequence if reset_sequence else None,
            focus_window_on_attach=bool(args.focus_window),
            window_targeted_input=effective_window_input,
            action_config=_build_action_config(
                args.movement_keys,
                include_prog_actions=bool(args.prog_actions),
            ),
            reward_fn=reward_fn,
        )
        tui.start()

        def _on_step(event: dict[str, Any]) -> None:
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
            )

        def _on_before_step(event: dict[str, Any]) -> None:
            tui.wait_for_step_advance(
                training_line=(
                    "episode={episode} step={step} total={total:.3f} "
                    "epsilon={epsilon:.4f} updates={updates} waiting=enter".format(
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
                    before_step_callback=_on_before_step if bool(args.step_through) else None,
                    step_callback=_on_step if bool(args.tui) else None,
                )[0]
                results.append(episode_result)

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
                    before_step_callback=_on_before_step if bool(args.step_through) else None,
                    step_callback=_on_step if bool(args.tui) else None,
                )
            )
    finally:
        if env is not None:
            env.close()
        tui.close()

    _print_results(tuple(results))


if __name__ == "__main__":
    main()

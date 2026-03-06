"""CLI runner for random-policy validation against live game environment."""

from __future__ import annotations

import argparse
from statistics import mean

from src.controller.action_api import ActionConfig
from src.env.game_env import GameEnv, GameEnvConfig, run_random_policy


def _build_action_config(movement_keys: str) -> ActionConfig:
    default_config = ActionConfig()
    bindings = dict(default_config.action_key_bindings)
    key_codes = dict(default_config.key_codes)

    if movement_keys == "wasd":
        bindings.update(
            {
                "move_up": "W",
                "move_down": "S",
                "move_left": "A",
                "move_right": "D",
            }
        )
    elif movement_keys == "numpad":
        bindings.update(
            {
                "move_up": "NUMPAD8",
                "move_down": "NUMPAD2",
                "move_left": "NUMPAD4",
                "move_right": "NUMPAD6",
            }
        )
    elif movement_keys != "arrows":
        raise ValueError(
            "movement_keys must be one of: arrows, wasd, numpad."
        )

    return ActionConfig(
        action_key_bindings=bindings,
        key_codes=key_codes,
        timings=default_config.timings,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random-policy episodes against live game env.")
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--movement-keys",
        choices=("arrows", "wasd", "numpad"),
        default="arrows",
        help="Movement key mapping profile.",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated action names to sample from. Default excludes 'cancel'.",
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
        default=False,
        help="Require reset() to observe a non-terminal state before starting steps.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reset_sequence = tuple(
        action.strip()
        for action in str(args.reset_sequence).split(",")
        if action.strip()
    )

    config = GameEnvConfig(
        step_timeout_seconds=args.step_timeout,
        reset_timeout_seconds=args.reset_timeout,
        require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
    )
    action_config = _build_action_config(args.movement_keys)
    env = GameEnv.from_live_process(
        executable_name=args.exe,
        config=config,
        reset_sequence=reset_sequence if reset_sequence else None,
        focus_window_on_attach=bool(args.focus_window),
        window_targeted_input=bool(args.window_input),
        action_config=action_config,
    )
    try:
        if args.actions:
            policy_actions = tuple(
                action.strip()
                for action in str(args.actions).split(",")
                if action.strip()
            )
        else:
            policy_actions = tuple(action for action in env.action_space if action != "cancel")
            if not policy_actions:
                policy_actions = env.action_space

        results = run_random_policy(
            env=env,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
            actions=policy_actions,
        )
    finally:
        env.close()

    print("episode_id\tsteps\tdone\ttotal_reward\tterminal_reason")
    for result in results:
        print(
            f"{result.episode_id}\t{result.steps}\t{result.done}\t"
            f"{result.total_reward:.3f}\t{result.terminal_reason or ''}"
        )

    avg_steps = mean(result.steps for result in results)
    avg_reward = mean(result.total_reward for result in results)
    done_rate = sum(1 for result in results if result.done) / len(results)
    print(
        f"\nsummary episodes={len(results)} avg_steps={avg_steps:.2f} "
        f"avg_reward={avg_reward:.3f} done_rate={done_rate:.2%}"
    )


if __name__ == "__main__":
    main()

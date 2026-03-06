"""CLI runner for heuristic-policy validation against live game environment."""

from __future__ import annotations

import argparse
from statistics import mean

from src.agent.baseline_heuristic import HeuristicBaselineAgent, HeuristicBaselineConfig
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import _build_action_config, _build_reward_config, _build_reward_fn
from src.training.rewards import RewardWeights
from src.training.train import run_agent_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run heuristic-policy episodes against live game env."
    )
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for tie-breaks.")
    parser.add_argument(
        "--movement-keys",
        choices=("arrows", "wasd", "numpad"),
        default="arrows",
        help="Movement key mapping profile.",
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
    parser.add_argument(
        "--low-health-threshold",
        type=int,
        default=3,
        help="Health threshold where heuristic prefers waiting to conserve state.",
    )
    parser.add_argument(
        "--avoid-enemy-distance",
        type=int,
        default=1,
        help="When nearest enemy is within this Manhattan distance, move away if possible.",
    )
    default_weights = RewardWeights()
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reset_sequence = tuple(
        action.strip() for action in str(args.reset_sequence).split(",") if action.strip()
    )
    reward_config = _build_reward_config(args)
    reward_fn = _build_reward_fn(
        reward_config=reward_config,
        print_breakdown=bool(args.print_reward_breakdown),
    )
    env = GameEnv.from_live_process(
        executable_name=args.exe,
        config=GameEnvConfig(
            step_timeout_seconds=args.step_timeout,
            reset_timeout_seconds=args.reset_timeout,
            require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
        ),
        reset_sequence=reset_sequence if reset_sequence else None,
        focus_window_on_attach=bool(args.focus_window),
        window_targeted_input=bool(args.window_input),
        action_config=_build_action_config(args.movement_keys),
        reward_fn=reward_fn,
    )
    try:
        results = run_agent_policy(
            env=env,
            agent=HeuristicBaselineAgent(
                config=HeuristicBaselineConfig(
                    low_health_threshold=int(args.low_health_threshold),
                    avoid_enemy_distance=int(args.avoid_enemy_distance),
                )
            ),
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
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

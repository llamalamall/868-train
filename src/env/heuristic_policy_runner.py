"""CLI runner for heuristic-policy validation against live game environment."""

from __future__ import annotations

import argparse
import logging
from statistics import mean
from typing import Any

from src.agent.baseline_heuristic import HeuristicBaselineAgent, HeuristicBaselineConfig
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import (
    _build_action_config,
    _build_reward_config,
    _build_reward_fn,
    format_reward_breakdown_line,
)
from src.env.runner_tui import RunnerTuiSession
from src.training.rewards import RewardWeights
from src.training.train import run_agent_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run heuristic-policy episodes against live game env."
    )
    heuristic_defaults = HeuristicBaselineConfig()
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument(
        "--launch-exe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch --exe when not already running before attempting attach.",
    )
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
        "--prog-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prog-slot actions (prog_slot_1..prog_slot_10 mapped to 1..0).",
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
        "--post-action-delay",
        type=float,
        default=0.2,
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
        "--low-health-threshold",
        type=int,
        default=heuristic_defaults.low_health_threshold,
        help="Health threshold where heuristic prefers waiting to conserve state.",
    )
    parser.add_argument(
        "--enemy-prediction-horizon-steps",
        type=int,
        default=heuristic_defaults.enemy_prediction_horizon_steps,
        help="Enemy lookahead depth used to reject dangerous moves.",
    )
    default_weights = RewardWeights()
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
    parser.add_argument(
        "--verbose-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable heuristic action-choice logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if bool(args.step_through) and not bool(args.tui):
        parser.error("--step-through requires --tui.")
    effective_window_input = bool(args.window_input) or bool(args.step_through) or bool(args.tui)
    if effective_window_input and not bool(args.window_input):
        mode = "step-through" if bool(args.step_through) else "tui"
        print(
            f"{mode} enabled: using window-targeted input so actions still go to the game "
            "while the TUI window has focus."
        )
    if args.verbose_actions:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

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
            ),
            reward_fn=reward_fn,
        )
        tui.start()

        def _on_step(event: dict[str, Any]) -> None:
            tui.consume_manual_step_flag()
            tui.update(
                training_line=(
                    "episode={episode} step={step} reward={reward:.3f} total={total:.3f} "
                    "done={done} terminal={terminal}".format(
                        episode=event.get("episode_id"),
                        step=int(event.get("step_index", 0)) + 1,
                        reward=float(event.get("reward", 0.0)),
                        total=float(event.get("total_reward", 0.0)),
                        done=bool(event.get("done", False)),
                        terminal=event.get("terminal_reason") or "-",
                    )
                ),
                action_line="action={action} reason={reason}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "heuristic_select",
                ),
                reward_line=format_reward_breakdown_line(event),
            )

        def _on_before_step(event: dict[str, Any]) -> None:
            tui.wait_for_step_gate(
                training_line=(
                    "episode={episode} step={step} total={total:.3f} waiting=step".format(
                        episode=event.get("episode_id"),
                        step=int(event.get("step_index", 0)) + 1,
                        total=float(event.get("total_reward", 0.0)),
                    )
                ),
                action_line="action={action} reason={reason}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "heuristic_select",
                ),
            )

        assert env is not None
        results = run_agent_policy(
            env=env,
            agent=HeuristicBaselineAgent(
                config=HeuristicBaselineConfig(
                    low_health_threshold=int(args.low_health_threshold),
                    enemy_prediction_horizon_steps=max(
                        int(args.enemy_prediction_horizon_steps),
                        0,
                    ),
                    verbose_action_logging=bool(args.verbose_actions),
                )
            ),
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
            before_step_callback=_on_before_step if bool(args.tui) else None,
            step_callback=_on_step if bool(args.tui) else None,
        )
    finally:
        if env is not None:
            env.close()
        tui.close()

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

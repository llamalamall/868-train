"""CLI runner for random-policy validation against live game environment."""

from __future__ import annotations

import argparse
from typing import Any
from statistics import mean

from src.controller.action_api import ActionConfig
from src.env.game_env import GameEnv, GameEnvConfig, RewardFunction, run_random_policy
from src.env.runner_tui import RunnerTuiSession
from src.state.schema import GameStateSnapshot
from src.training.rewards import RewardConfig, RewardWeights, compute_reward

_WASD_KEY_CODES = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
}

_NUMPAD_KEY_CODES = {
    "NUMPAD2": 0x62,
    "NUMPAD4": 0x64,
    "NUMPAD6": 0x66,
    "NUMPAD8": 0x68,
}
_PROG_SLOT_ACTION_BINDINGS = {
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


def _build_action_config(movement_keys: str, *, include_prog_actions: bool = True) -> ActionConfig:
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
        key_codes.update(_WASD_KEY_CODES)
    elif movement_keys == "numpad":
        bindings.update(
            {
                "move_up": "NUMPAD8",
                "move_down": "NUMPAD2",
                "move_left": "NUMPAD4",
                "move_right": "NUMPAD6",
            }
        )
        key_codes.update(_NUMPAD_KEY_CODES)
    elif movement_keys != "arrows":
        raise ValueError(
            "movement_keys must be one of: arrows, wasd, numpad."
        )

    if include_prog_actions:
        bindings.update(_PROG_SLOT_ACTION_BINDINGS)
    else:
        bindings = {
            action_name: key_name
            for action_name, key_name in bindings.items()
            if not action_name.startswith("prog_slot_")
        }

    return ActionConfig(
        action_key_bindings=bindings,
        key_codes=key_codes,
        timings=default_config.timings,
    )


def _build_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        weights=RewardWeights(
            survival=float(args.reward_survival),
            step_penalty=float(args.reward_step_penalty),
            health_delta=float(args.reward_health_delta),
            currency_delta=float(args.reward_currency_delta),
            energy_delta=float(args.reward_energy_delta),
            score_delta=float(args.reward_score_delta),
            siphon_collected=float(args.reward_siphon_collected),
            enemy_cleared=float(args.reward_enemy_cleared),
            phase_progress=float(args.reward_phase_progress),
            map_clear_bonus=float(args.reward_map_clear_bonus),
            premature_exit_penalty=float(args.reward_premature_exit_penalty),
            invalid_action_penalty=float(args.reward_invalid_action_penalty),
            fail_penalty=float(args.reward_fail_penalty),
            safe_tile_bonus=float(args.reward_safe_tile_bonus),
            danger_tile_penalty=float(args.reward_danger_tile_penalty),
            resource_proximity=float(args.reward_resource_proximity),
            prog_collected_base=float(args.reward_prog_collected_base),
            points_collected=float(args.reward_points_collected),
            damage_taken_penalty=float(args.reward_damage_taken_penalty),
        ),
        reward_clip_abs=float(args.reward_clip_abs),
    )


def _build_reward_fn(
    *,
    reward_config: RewardConfig,
    print_breakdown: bool = False,
) -> RewardFunction:
    def reward_fn(
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        result = compute_reward(
            previous_state=previous_state,
            current_state=current_state,
            done=done,
            config=reward_config,
            info=info,
        )
        info["reward_breakdown"] = {
            "survival": result.breakdown.survival,
            "step_penalty": result.breakdown.step_penalty,
            "health_change": result.breakdown.health_change,
            "currency_change": result.breakdown.currency_change,
            "energy_change": result.breakdown.energy_change,
            "score_change": result.breakdown.score_change,
            "siphon_collected": result.breakdown.siphon_collected,
            "enemy_cleared": result.breakdown.enemy_cleared,
            "phase_progress": result.breakdown.phase_progress,
            "map_clear_bonus": result.breakdown.map_clear_bonus,
            "premature_exit_penalty": result.breakdown.premature_exit_penalty,
            "invalid_action_penalty": result.breakdown.invalid_action_penalty,
            "fail_penalty": result.breakdown.fail_penalty,
            "safe_tile_bonus": result.breakdown.safe_tile_bonus,
            "danger_tile_penalty": result.breakdown.danger_tile_penalty,
            "resource_proximity": result.breakdown.resource_proximity,
            "prog_collected": result.breakdown.prog_collected,
            "points_collected": result.breakdown.points_collected,
            "damage_taken_penalty": result.breakdown.damage_taken_penalty,
            "total": result.total,
        }
        if print_breakdown:
            print(
                "reward step={step} action={action} total={total:.3f} "
                "survival={survival:.3f} step_penalty={step_penalty:.3f} health={health:.3f} "
                "currency={currency:.3f} energy={energy:.3f} score={score:.3f} "
                "siphon={siphon:.3f} enemy={enemy:.3f} "
                "safe={safe:.3f} danger={danger:.3f} proximity={proximity:.3f} "
                "prog={prog:.3f} points={points:.3f} damage_taken={damage:.3f} "
                "premature_exit={premature:.3f} invalid={invalid:.3f} "
                "fail_penalty={fail_penalty:.3f} done={done}".format(
                    step=info.get("step_index"),
                    action=info.get("action"),
                    total=result.total,
                    survival=result.breakdown.survival,
                    step_penalty=result.breakdown.step_penalty,
                    health=result.breakdown.health_change,
                    currency=result.breakdown.currency_change,
                    energy=result.breakdown.energy_change,
                    score=result.breakdown.score_change,
                    siphon=result.breakdown.siphon_collected,
                    enemy=result.breakdown.enemy_cleared,
                    safe=result.breakdown.safe_tile_bonus,
                    danger=result.breakdown.danger_tile_penalty,
                    proximity=result.breakdown.resource_proximity,
                    prog=result.breakdown.prog_collected,
                    points=result.breakdown.points_collected,
                    damage=result.breakdown.damage_taken_penalty,
                    premature=result.breakdown.premature_exit_penalty,
                    invalid=result.breakdown.invalid_action_penalty,
                    fail_penalty=result.breakdown.fail_penalty,
                    done=done,
                )
            )
        return result.total

    return reward_fn


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_reward_breakdown_line(event: dict[str, Any]) -> str:
    """Format a compact, human-readable reward dump from callback event payload."""
    breakdown = event.get("reward_breakdown")
    if not isinstance(breakdown, dict):
        return "reward unavailable"

    ordered_fields = (
        ("total", "total"),
        ("survival", "survival"),
        ("step_penalty", "step"),
        ("health_change", "health"),
        ("currency_change", "currency"),
        ("energy_change", "energy"),
        ("score_change", "score"),
        ("siphon_collected", "siphon"),
        ("enemy_cleared", "enemy"),
        ("phase_progress", "phase"),
        ("map_clear_bonus", "map_clear"),
        ("safe_tile_bonus", "safe"),
        ("danger_tile_penalty", "danger"),
        ("resource_proximity", "proximity"),
        ("prog_collected", "prog"),
        ("points_collected", "points"),
        ("damage_taken_penalty", "damage"),
        ("premature_exit_penalty", "premature_exit"),
        ("invalid_action_penalty", "invalid"),
        ("fail_penalty", "fail"),
    )
    components: list[str] = []
    for key, label in ordered_fields:
        number = _as_float(breakdown.get(key))
        if number is None:
            continue
        if key == "total" or abs(number) > 1e-9:
            components.append(f"{label}={number:+.3f}")
    if not components:
        return "reward breakdown empty"
    return "reward " + " ".join(components)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random-policy episodes against live game env.")
    default_weights = RewardWeights()
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument(
        "--launch-exe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch --exe when not already running before attempting attach.",
    )
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
        "--prog-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include prog-slot actions (prog_slot_1..prog_slot_10 mapped to 1..0).",
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
        "--reward-enemy-cleared",
        type=float,
        default=default_weights.enemy_cleared,
        help="Reward per live enemy removed from map.",
    )
    parser.add_argument(
        "--reward-phase-progress",
        type=float,
        default=default_weights.phase_progress,
        help="Weight for progress toward active objective (siphon->enemy->exit).",
    )
    parser.add_argument(
        "--reward-map-clear-bonus",
        type=float,
        default=default_weights.map_clear_bonus,
        help="Bonus when player reaches exit after all siphons/enemies are cleared.",
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

    reset_sequence = tuple(
        action.strip()
        for action in str(args.reset_sequence).split(",")
        if action.strip()
    )

    config = GameEnvConfig(
        step_timeout_seconds=args.step_timeout,
        reset_timeout_seconds=args.reset_timeout,
        post_action_poll_delay_seconds=max(float(args.post_action_delay), 0.0),
        wait_for_action_processing=bool(args.wait_for_action_processing),
        action_ack_timeout_seconds=max(float(args.action_ack_timeout), 0.0),
        action_ack_poll_interval_seconds=max(float(args.action_ack_poll_interval), 0.0),
        prog_slot_backoff_steps=max(int(args.prog_backoff_steps), 0),
        require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
    )
    action_config = _build_action_config(
        args.movement_keys,
        include_prog_actions=bool(args.prog_actions),
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
            config=config,
            reset_sequence=reset_sequence if reset_sequence else None,
            launch_process_if_missing=bool(args.launch_exe),
            focus_window_on_attach=bool(args.focus_window),
            window_targeted_input=effective_window_input,
            action_config=action_config,
            reward_fn=reward_fn,
        )
        tui.start()

        def _on_step(event: dict[str, Any]) -> None:
            should_emit_reward_dump = bool(args.print_reward_breakdown) or tui.consume_manual_step_flag()
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
                    reason=event.get("action_reason") or "random_policy_sample",
                ),
                reward_line=(format_reward_breakdown_line(event) if should_emit_reward_dump else ""),
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
                    reason=event.get("action_reason") or "random_policy_sample",
                ),
            )

        if args.actions:
            policy_actions = tuple(
                action.strip()
                for action in str(args.actions).split(",")
                if action.strip()
            )
        else:
            assert env is not None
            policy_actions = tuple(action for action in env.action_space if action != "cancel")
            if not policy_actions:
                policy_actions = env.action_space

        assert env is not None
        results = run_random_policy(
            env=env,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
            actions=policy_actions,
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

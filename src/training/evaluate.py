"""Evaluation helpers for baseline and DQN policy checkpoints."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Protocol

from src.agent.dqn_agent import DQNAgent
from src.env.game_env import GameEnv, GameEnvConfig
from src.env.random_policy_runner import _build_action_config, _build_reward_config, _build_reward_fn
from src.env.runner_tui import RunnerTuiSession
from src.state.schema import FieldState, GameStateSnapshot
from src.training.train import BaselineAgent, EpisodeEnv, EpisodeRolloutResult, run_agent_policy

EvaluationEventCallback = Callable[[dict[str, Any]], None]


class ClosableEpisodeEnv(EpisodeEnv, Protocol):
    """Episode environment with optional close hook."""

    def close(self) -> None:
        """Release environment resources."""


@dataclass(frozen=True)
class BaselineMetrics:
    """Comparable KPI summary for one baseline policy."""

    policy_name: str
    episodes: int
    avg_steps: float
    avg_reward: float
    done_rate: float
    terminal_reasons: dict[str, int]


@dataclass(frozen=True)
class DQNEpisodeKpi:
    """Per-episode KPI values used in checkpoint-level evaluation reports."""

    episode_id: str
    steps: int
    done: bool
    terminal_reason: str | None
    failed: bool
    health_delta: float
    currency_gain: float


@dataclass(frozen=True)
class DQNEvaluationReport:
    """Aggregate KPI report for a single DQN checkpoint evaluation."""

    label: str
    checkpoint_path: str
    episodes: int
    max_steps_per_episode: int
    seed: int | None
    fail_rate: float
    avg_episode_length: float
    avg_health_delta: float
    avg_currency_gain: float
    terminal_reasons: dict[str, int]
    episode_results: tuple[DQNEpisodeKpi, ...]


@dataclass(frozen=True)
class DQNCheckpointComparison:
    """Side-by-side KPI comparison for two evaluated checkpoints."""

    checkpoint_a: DQNEvaluationReport
    checkpoint_b: DQNEvaluationReport
    fail_rate_delta: float
    avg_episode_length_delta: float
    avg_health_delta_delta: float
    avg_currency_gain_delta: float


def summarize_rollouts(*, policy_name: str, results: tuple[EpisodeRolloutResult, ...]) -> BaselineMetrics:
    """Convert episode-level rollout output into aggregate metrics."""
    if not results:
        raise ValueError("results must not be empty.")

    terminal_counts: dict[str, int] = {}
    for result in results:
        if result.terminal_reason:
            terminal_counts[result.terminal_reason] = terminal_counts.get(result.terminal_reason, 0) + 1

    return BaselineMetrics(
        policy_name=policy_name,
        episodes=len(results),
        avg_steps=mean(result.steps for result in results),
        avg_reward=mean(result.total_reward for result in results),
        done_rate=sum(1 for result in results if result.done) / len(results),
        terminal_reasons=terminal_counts,
    )


def evaluate_baseline(
    *,
    env: EpisodeEnv,
    policy_name: str,
    agent: BaselineAgent,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
) -> BaselineMetrics:
    """Run one baseline policy and return comparable metrics."""
    results = run_agent_policy(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )
    return summarize_rollouts(policy_name=policy_name, results=results)


def compare_baselines(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    baselines: tuple[tuple[str, BaselineAgent], ...],
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = None,
) -> tuple[BaselineMetrics, ...]:
    """Evaluate multiple baselines under matching rollout settings."""
    if not baselines:
        raise ValueError("baselines must include at least one policy.")

    metrics: list[BaselineMetrics] = []
    for baseline_index, (policy_name, agent) in enumerate(baselines):
        env = env_factory()
        try:
            # Offset seed so each policy has deterministic but independent action sampling.
            policy_seed = None if seed is None else seed + baseline_index
            metrics.append(
                evaluate_baseline(
                    env=env,
                    policy_name=policy_name,
                    agent=agent,
                    episodes=episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    seed=policy_seed,
                )
            )
        finally:
            env.close()

    return tuple(metrics)


def _validate_rollout_settings(*, episodes: int, max_steps_per_episode: int) -> None:
    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be >= 1.")


def _field_to_float(field: FieldState) -> float | None:
    if field.status != "ok":
        return None
    value = field.value
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_or_zero(*, initial: float | None, final: float | None) -> float:
    if initial is None or final is None:
        return 0.0
    return float(final - initial)


def _episode_failed(*, final_state: GameStateSnapshot, terminal_reason: str | None) -> bool:
    if final_state.fail_state.status == "ok" and bool(final_state.fail_state.value):
        return True
    if terminal_reason is None:
        return False
    reason = terminal_reason.strip().lower()
    return any(token in reason for token in ("fail", "loss", "dead", "start_screen"))


def _reset_episode(env: EpisodeEnv, *, episode_seed: int | None) -> GameStateSnapshot:
    if episode_seed is None:
        return env.reset()

    reset_signature: inspect.Signature | None = None
    try:
        reset_signature = inspect.signature(env.reset)
    except (TypeError, ValueError):
        reset_signature = None
    if reset_signature is not None and "seed" in reset_signature.parameters:
        return env.reset(seed=episode_seed)  # type: ignore[call-arg]

    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        seed_fn(episode_seed)
    return env.reset()


def _resolve_eval_actions(*, env: EpisodeEnv, agent: DQNAgent, state: GameStateSnapshot) -> tuple[str, ...]:
    action_space = set(agent.action_space)
    available = tuple(action for action in env.available_actions(state) if action in action_space)
    if available:
        return available

    fallback = tuple(
        action
        for action in env.action_space
        if action in action_space and action not in {"cancel"}
    )
    if fallback:
        return fallback

    overlap = tuple(action for action in env.action_space if action in action_space)
    if overlap:
        return overlap
    raise ValueError("No overlapping actions between DQN checkpoint and environment action space.")


def summarize_dqn_episodes(
    *,
    label: str,
    checkpoint_path: str | Path,
    episodes: tuple[DQNEpisodeKpi, ...],
    max_steps_per_episode: int,
    seed: int | None,
) -> DQNEvaluationReport:
    """Summarize per-episode KPI rows into one checkpoint report."""
    if not episodes:
        raise ValueError("episodes must not be empty.")

    terminal_counts: dict[str, int] = {}
    for episode in episodes:
        if episode.terminal_reason:
            terminal_counts[episode.terminal_reason] = (
                terminal_counts.get(episode.terminal_reason, 0) + 1
            )

    return DQNEvaluationReport(
        label=label,
        checkpoint_path=str(checkpoint_path),
        episodes=len(episodes),
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        fail_rate=sum(1 for episode in episodes if episode.failed) / len(episodes),
        avg_episode_length=mean(episode.steps for episode in episodes),
        avg_health_delta=mean(episode.health_delta for episode in episodes),
        avg_currency_gain=mean(episode.currency_gain for episode in episodes),
        terminal_reasons=terminal_counts,
        episode_results=episodes,
    )


def evaluate_dqn_checkpoint(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    checkpoint_path: str | Path,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = 0,
    label: str | None = None,
    step_callback: EvaluationEventCallback | None = None,
    episode_callback: EvaluationEventCallback | None = None,
) -> DQNEvaluationReport:
    """Run fixed-seed, no-exploration episodes and compute Task-15 KPIs."""
    _validate_rollout_settings(episodes=episodes, max_steps_per_episode=max_steps_per_episode)

    checkpoint = Path(checkpoint_path)
    agent = DQNAgent.load_checkpoint(checkpoint)
    report_label = label or checkpoint.name
    per_episode: list[DQNEpisodeKpi] = []
    env = env_factory()
    try:
        if not (set(agent.action_space) & set(env.action_space)):
            raise ValueError(
                "No overlapping actions between loaded DQN checkpoint and environment action_space."
            )

        for episode_index in range(episodes):
            episode_seed = None if seed is None else int(seed) + episode_index
            state = _reset_episode(env, episode_seed=episode_seed)
            episode_id = env.current_episode_id or f"episode-{episode_index + 1:05d}"
            if episode_callback is not None:
                episode_callback(
                    {
                        "event_type": "episode_start",
                        "checkpoint_label": report_label,
                        "checkpoint_path": str(checkpoint),
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "episodes_total": episodes,
                        "max_steps_per_episode": max_steps_per_episode,
                        "seed": episode_seed,
                    }
                )

            initial_health = _field_to_float(state.health)
            initial_currency = _field_to_float(state.currency)

            steps = 0
            done = False
            terminal_reason: str | None = None
            while steps < max_steps_per_episode and not done:
                allowed_actions = _resolve_eval_actions(env=env, agent=agent, state=state)
                action = agent.select_action(
                    state=state,
                    available_actions=allowed_actions,
                    explore=False,
                )
                state, reward, done, info = env.step(action)
                steps += 1
                reason = info.get("terminal_reason")
                if isinstance(reason, str) and reason.strip():
                    terminal_reason = reason
                if step_callback is not None:
                    step_callback(
                        {
                            "event_type": "step",
                            "checkpoint_label": report_label,
                            "checkpoint_path": str(checkpoint),
                            "episode_id": episode_id,
                            "episode_index": episode_index,
                            "episodes_total": episodes,
                            "step_index": steps - 1,
                            "max_steps_per_episode": max_steps_per_episode,
                            "action": action,
                            "action_reason": getattr(agent, "last_decision_reason", None) or "greedy_q",
                            "reward": float(reward),
                            "done": bool(done),
                            "terminal_reason": terminal_reason,
                        }
                    )

            final_health = _field_to_float(state.health)
            final_currency = _field_to_float(state.currency)
            episode_result = DQNEpisodeKpi(
                episode_id=episode_id,
                steps=steps,
                done=done,
                terminal_reason=terminal_reason,
                failed=_episode_failed(final_state=state, terminal_reason=terminal_reason),
                health_delta=_delta_or_zero(initial=initial_health, final=final_health),
                currency_gain=_delta_or_zero(initial=initial_currency, final=final_currency),
            )
            per_episode.append(episode_result)
            if episode_callback is not None:
                episode_callback(
                    {
                        "event_type": "episode_end",
                        "checkpoint_label": report_label,
                        "checkpoint_path": str(checkpoint),
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "episodes_total": episodes,
                        "steps": episode_result.steps,
                        "done": episode_result.done,
                        "failed": episode_result.failed,
                        "terminal_reason": episode_result.terminal_reason,
                        "health_delta": episode_result.health_delta,
                        "currency_gain": episode_result.currency_gain,
                    }
                )
    finally:
        env.close()

    return summarize_dqn_episodes(
        label=report_label,
        checkpoint_path=checkpoint,
        episodes=tuple(per_episode),
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )


def compare_dqn_checkpoints(
    *,
    env_factory: Callable[[], ClosableEpisodeEnv],
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    episodes: int,
    max_steps_per_episode: int = 200,
    seed: int | None = 0,
    label_a: str | None = None,
    label_b: str | None = None,
) -> DQNCheckpointComparison:
    """Evaluate two checkpoints under matching fixed-seed settings and report KPI deltas."""
    report_a = evaluate_dqn_checkpoint(
        env_factory=env_factory,
        checkpoint_path=checkpoint_a,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        label=label_a,
    )
    report_b = evaluate_dqn_checkpoint(
        env_factory=env_factory,
        checkpoint_path=checkpoint_b,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
        label=label_b,
    )
    return summarize_dqn_comparison(report_a=report_a, report_b=report_b)


def summarize_dqn_comparison(
    *,
    report_a: DQNEvaluationReport,
    report_b: DQNEvaluationReport,
) -> DQNCheckpointComparison:
    """Build comparison KPI deltas from two per-checkpoint evaluation reports."""
    return DQNCheckpointComparison(
        checkpoint_a=report_a,
        checkpoint_b=report_b,
        fail_rate_delta=report_b.fail_rate - report_a.fail_rate,
        avg_episode_length_delta=report_b.avg_episode_length - report_a.avg_episode_length,
        avg_health_delta_delta=report_b.avg_health_delta - report_a.avg_health_delta,
        avg_currency_gain_delta=report_b.avg_currency_gain - report_a.avg_currency_gain,
    )


def evaluation_report_to_json(report: DQNEvaluationReport) -> dict[str, Any]:
    """Convert one checkpoint report to machine-readable JSON payload."""
    payload = asdict(report)
    payload["kind"] = "dqn_evaluation"
    return payload


def comparison_report_to_json(report: DQNCheckpointComparison) -> dict[str, Any]:
    """Convert checkpoint comparison report to machine-readable JSON payload."""
    payload = asdict(report)
    payload["kind"] = "dqn_checkpoint_comparison"
    return payload


def format_evaluation_table(report: DQNEvaluationReport) -> str:
    """Render one checkpoint report as a compact human-readable table."""
    lines = [
        "label\tepisodes\tfail_rate\tavg_episode_length\tavg_health_delta\tavg_currency_gain",
        (
            f"{report.label}\t{report.episodes}\t{report.fail_rate:.2%}\t"
            f"{report.avg_episode_length:.2f}\t{report.avg_health_delta:.3f}\t"
            f"{report.avg_currency_gain:.3f}"
        ),
    ]
    if report.terminal_reasons:
        reasons = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(report.terminal_reasons.items())
        )
        lines.append(f"terminal_reasons\t{reasons}")
    return "\n".join(lines)


def format_comparison_table(report: DQNCheckpointComparison) -> str:
    """Render checkpoint-vs-checkpoint KPI deltas in tabular form."""
    a = report.checkpoint_a
    b = report.checkpoint_b
    lines = [
        "metric\tcheckpoint_a\tcheckpoint_b\tdelta_b_minus_a",
        f"label\t{a.label}\t{b.label}\t-",
        f"fail_rate\t{a.fail_rate:.2%}\t{b.fail_rate:.2%}\t{report.fail_rate_delta:+.2%}",
        (
            f"avg_episode_length\t{a.avg_episode_length:.2f}\t{b.avg_episode_length:.2f}\t"
            f"{report.avg_episode_length_delta:+.2f}"
        ),
        (
            f"avg_health_delta\t{a.avg_health_delta:.3f}\t{b.avg_health_delta:.3f}\t"
            f"{report.avg_health_delta_delta:+.3f}"
        ),
        (
            f"avg_currency_gain\t{a.avg_currency_gain:.3f}\t{b.avg_currency_gain:.3f}\t"
            f"{report.avg_currency_gain_delta:+.3f}"
        ),
    ]
    return "\n".join(lines)


def _add_shared_eval_args(
    parser: argparse.ArgumentParser,
    *,
    focus_window_default: bool,
    window_input_default: bool,
) -> None:
    parser.add_argument("--exe", default="868-HACK.exe", help="Target executable name.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of fixed-seed episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic eval runs.")
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
        help="Comma-separated reset actions; pass without value to disable reset actions.",
    )
    parser.add_argument(
        "--focus-window",
        action=argparse.BooleanOptionalAction,
        default=focus_window_default,
        help="Focus game window during attach/reacquire.",
    )
    parser.add_argument(
        "--window-input",
        action=argparse.BooleanOptionalAction,
        default=window_input_default,
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
        default=True,
        help="Require reset() to observe a non-terminal state before stepping.",
    )
    parser.add_argument(
        "--reward-survival",
        type=float,
        default=0.1,
        help="Survival reward applied for non-terminal steps.",
    )
    parser.add_argument(
        "--reward-health-delta",
        type=float,
        default=0.25,
        help="Weight multiplied by (current_health - previous_health).",
    )
    parser.add_argument(
        "--reward-currency-delta",
        type=float,
        default=0.1,
        help="Weight multiplied by (current_currency - previous_currency).",
    )
    parser.add_argument(
        "--reward-fail-penalty",
        type=float,
        default=5.0,
        help="Terminal fail penalty magnitude (applied as negative).",
    )
    parser.add_argument(
        "--print-reward-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-step reward component breakdown during evaluation.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON summary output.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task-15 DQN evaluation harness with fixed-seed KPI reporting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Evaluate one DQN checkpoint.",
        description="Run fixed-seed evaluation for one checkpoint and print KPIs.",
    )
    _add_shared_eval_args(
        run_parser,
        focus_window_default=True,
        window_input_default=False,
    )
    run_parser.add_argument("--checkpoint", required=True, help="Path to DQN checkpoint JSON.")
    run_parser.add_argument("--label", default=None, help="Optional display label in KPI table.")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two DQN checkpoints under identical evaluation settings.",
        description="Run fixed-seed KPI evaluation for checkpoint A and B, then print deltas.",
    )
    _add_shared_eval_args(
        compare_parser,
        focus_window_default=False,
        window_input_default=True,
    )
    compare_parser.add_argument("--checkpoint-a", required=True, help="Baseline checkpoint path.")
    compare_parser.add_argument("--checkpoint-b", required=True, help="Candidate checkpoint path.")
    compare_parser.add_argument("--label-a", default=None, help="Optional table label for A.")
    compare_parser.add_argument("--label-b", default=None, help="Optional table label for B.")
    compare_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch live state monitor TUI during compare runs.",
    )
    compare_parser.add_argument(
        "--tui-interval",
        type=float,
        default=0.5,
        help="Polling interval for compare TUI state monitor (seconds).",
    )
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.episodes) < 1:
        parser.error("--episodes must be >= 1.")
    if int(args.max_steps) < 1:
        parser.error("--max-steps must be >= 1.")
    if args.command == "compare" and float(args.tui_interval) <= 0:
        parser.error("--tui-interval must be > 0.")


def _build_live_env_factory(args: argparse.Namespace) -> Callable[[], GameEnv]:
    reset_sequence = tuple(
        action.strip() for action in str(args.reset_sequence).split(",") if action.strip()
    )
    reward_config = _build_reward_config(args)
    reward_fn = _build_reward_fn(
        reward_config=reward_config,
        print_breakdown=bool(args.print_reward_breakdown),
    )

    def _factory() -> GameEnv:
        return GameEnv.from_live_process(
            executable_name=str(args.exe),
            config=GameEnvConfig(
                step_timeout_seconds=float(args.step_timeout),
                reset_timeout_seconds=float(args.reset_timeout),
                require_non_terminal_on_reset=bool(args.require_non_terminal_reset),
            ),
            reset_sequence=reset_sequence if reset_sequence else None,
            focus_window_on_attach=bool(args.focus_window),
            window_targeted_input=bool(args.window_input),
            action_config=_build_action_config(
                str(args.movement_keys),
                include_prog_actions=bool(args.prog_actions),
            ),
            reward_fn=reward_fn,
        )

    return _factory


def _emit_json_output(payload: dict[str, Any], *, output_path: str | None) -> None:
    json_text = json.dumps(payload, indent=2, sort_keys=True)
    print()
    print(json_text)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{json_text}\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(parser, args)
    env_factory = _build_live_env_factory(args)

    if args.command == "run":
        report = evaluate_dqn_checkpoint(
            env_factory=env_factory,
            checkpoint_path=args.checkpoint,
            episodes=int(args.episodes),
            max_steps_per_episode=int(args.max_steps),
            seed=args.seed,
            label=args.label,
        )
        print(format_evaluation_table(report))
        _emit_json_output(evaluation_report_to_json(report), output_path=args.json_out)
        return

    if args.command == "compare":
        label_a = args.label_a or Path(str(args.checkpoint_a)).name
        label_b = args.label_b or Path(str(args.checkpoint_b)).name
        tui = RunnerTuiSession(
            executable_name=str(args.exe),
            enabled=bool(args.tui),
            interval_seconds=float(args.tui_interval),
        )

        def _on_step(event: dict[str, Any]) -> None:
            tui.update(
                training_line=(
                    "compare checkpoint={label} episode={episode}/{episode_total} "
                    "step={step}/{step_max} reward={reward:.3f} done={done} terminal={terminal}".format(
                        label=event.get("checkpoint_label"),
                        episode=int(event.get("episode_index", 0)) + 1,
                        episode_total=int(event.get("episodes_total", 0)),
                        step=int(event.get("step_index", 0)) + 1,
                        step_max=int(event.get("max_steps_per_episode", 0)),
                        reward=float(event.get("reward", 0.0)),
                        done=bool(event.get("done", False)),
                        terminal=event.get("terminal_reason") or "-",
                    )
                ),
                action_line="action={action} reason={reason}".format(
                    action=event.get("action"),
                    reason=event.get("action_reason") or "greedy_q",
                ),
            )

        def _on_episode(event: dict[str, Any]) -> None:
            event_type = str(event.get("event_type", ""))
            if event_type == "episode_start":
                tui.update(
                    training_line=(
                        "compare checkpoint={label} episode={episode}/{episode_total} "
                        "status=running seed={seed}".format(
                            label=event.get("checkpoint_label"),
                            episode=int(event.get("episode_index", 0)) + 1,
                            episode_total=int(event.get("episodes_total", 0)),
                            seed=event.get("seed"),
                        )
                    ),
                    action_line="action=idle reason=episode_start",
                )
                return
            if event_type == "episode_end":
                tui.update(
                    training_line=(
                        "compare checkpoint={label} episode={episode}/{episode_total} "
                        "steps={steps} failed={failed} health_delta={health:+.3f} "
                        "currency_gain={currency:+.3f} terminal={terminal}".format(
                            label=event.get("checkpoint_label"),
                            episode=int(event.get("episode_index", 0)) + 1,
                            episode_total=int(event.get("episodes_total", 0)),
                            steps=int(event.get("steps", 0)),
                            failed=bool(event.get("failed", False)),
                            health=float(event.get("health_delta", 0.0)),
                            currency=float(event.get("currency_gain", 0.0)),
                            terminal=event.get("terminal_reason") or "-",
                        )
                    ),
                    action_line="action=idle reason=episode_complete",
                )

        try:
            tui.start()
            tui.update(
                training_line="compare checkpoint={label} status=starting".format(label=label_a),
                action_line="action=idle reason=checkpoint_start",
            )
            report_a = evaluate_dqn_checkpoint(
                env_factory=env_factory,
                checkpoint_path=args.checkpoint_a,
                episodes=int(args.episodes),
                max_steps_per_episode=int(args.max_steps),
                seed=args.seed,
                label=label_a,
                step_callback=_on_step if bool(args.tui) else None,
                episode_callback=_on_episode if bool(args.tui) else None,
            )
            tui.update(
                training_line="compare checkpoint={label} status=complete".format(label=label_a),
                action_line="action=idle reason=checkpoint_complete",
            )
            tui.update(
                training_line="compare checkpoint={label} status=starting".format(label=label_b),
                action_line="action=idle reason=checkpoint_start",
            )
            report_b = evaluate_dqn_checkpoint(
                env_factory=env_factory,
                checkpoint_path=args.checkpoint_b,
                episodes=int(args.episodes),
                max_steps_per_episode=int(args.max_steps),
                seed=args.seed,
                label=label_b,
                step_callback=_on_step if bool(args.tui) else None,
                episode_callback=_on_episode if bool(args.tui) else None,
            )
            tui.update(
                training_line="compare status=complete checkpoint_a={a} checkpoint_b={b}".format(
                    a=label_a,
                    b=label_b,
                ),
                action_line="action=idle reason=compare_complete",
            )
        finally:
            tui.close()

        report = summarize_dqn_comparison(report_a=report_a, report_b=report_b)
        print(format_comparison_table(report))
        _emit_json_output(comparison_report_to_json(report), output_path=args.json_out)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

"""Hybrid rollout and reporting helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
import time
from typing import Any

from src.env.runner_monitor import RunnerMonitorSession
from src.hybrid.coordinator import HybridCoordinator
from src.hybrid.env import HybridLiveEnv
from src.hybrid.rewards import (
    HybridRewardSuite,
    HybridThreatRewardBreakdown,
)
from src.hybrid.state_deltas import (
    enemy_cleared_delta,
    enemy_damage_delta,
    enemy_growth_delta,
    health_damage_taken,
    reason_indicates_fail_terminal,
)
from src.hybrid.tactical_model import siphon_spawn_cost_at_position
from src.hybrid.types import ObjectivePhase, ThreatOverride

def _format_monitor_actions(actions: object, *, limit: int = 8) -> str:
    if not isinstance(actions, (tuple, list)):
        return "-"
    normalized = tuple(str(item).strip() for item in actions if str(item).strip())
    if not normalized:
        return "-"
    return ",".join(normalized)


def _terminal_is_fail(reason: object) -> bool:
    return reason_indicates_fail_terminal(reason)


def _hook_event_is_victory_signal(event: dict[str, Any]) -> bool:
    return str(event.get("target_name") or "").strip() in {
        "normal_victory_flag_set",
        "points_victory_flag_set",
    }


def _serialize_position(position: object | None) -> dict[str, int] | None:
    if position is None:
        return None
    x_value = getattr(position, "x", None)
    y_value = getattr(position, "y", None)
    if not isinstance(x_value, int) or not isinstance(y_value, int):
        return None
    return {"x": x_value, "y": y_value}


def _update_last_known_positions(
    *,
    state: object,
    last_player_position: dict[str, int] | None,
    last_exit_position: dict[str, int] | None,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    map_state = getattr(state, "map", None)
    if getattr(map_state, "status", None) != "ok":
        return (last_player_position, last_exit_position)
    player_position = _serialize_position(getattr(map_state, "player_position", None))
    exit_position = _serialize_position(getattr(map_state, "exit_position", None))
    return (
        player_position if player_position is not None else last_player_position,
        exit_position if exit_position is not None else last_exit_position,
    )


def _zero_threat_reward_breakdown() -> HybridThreatRewardBreakdown:
    return HybridThreatRewardBreakdown(
        survival=0.0,
        damage_taken_penalty=0.0,
        fail_penalty=0.0,
        route_rejoin_bonus=0.0,
        invalid_override_penalty=0.0,
        enemy_damaged=0.0,
        enemy_cleared=0.0,
        spawn_debt_penalty=0.0,
        total=0.0,
    )


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
    threat_rewarded_steps: int
    route_rejoin_events: int
    phase_lock_overrides: int
    stall_releases: int
    requested_phase_override_steps: int
    requested_phase_counts: dict[str, int]
    executed_phase_counts: dict[str, int]
    requested_executed_phase_pair_counts: dict[str, int]
    meta_override_updates_skipped: int
    invalid_action_reason_counts: dict[str, int]
    action_ack_reason_counts: dict[str, int]
    invalid_action_action_counts: dict[str, int]
    action_ack_timeout_action_counts: dict[str, int]
    start_screen_detected: bool
    victory_detected: bool
    victory_inferred_from_start_screen: bool
    death_to_start_screen_detected: bool
    start_screen_unknown_detected: bool
    final_action: str | None
    last_known_player_position: dict[str, int] | None
    last_known_exit_position: dict[str, int] | None

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
    if normalized in {"state:victory", "state:sector_exit_to_start_screen"}:
        return "non_death_terminal"
    if normalized in {"state:start_screen", "state:start_screen_unknown"}:
        return "unexpected_start_screen"
    if _terminal_is_fail(normalized):
        return "fail_or_death"
    return "terminal_other"


def _terminal_reason_key(reason: str | None) -> str:
    normalized = str(reason or "").strip()
    return normalized if normalized else "none"


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _build_training_summary(results: tuple[HybridEpisodeSummary, ...]) -> dict[str, Any]:
    if not results:
        return {
            "episodes": 0,
            "done_episodes": 0,
            "non_death_terminal_episodes": 0,
            "unexpected_start_screen_episodes": 0,
            "hit_step_limit_episodes": 0,
            "done_rate": 0.0,
            "non_death_terminal_rate": 0.0,
            "unexpected_start_screen_rate": 0.0,
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
            "avg_threat_rewarded_steps": 0.0,
            "avg_route_rejoin_events": 0.0,
            "avg_phase_lock_overrides": 0.0,
            "avg_stall_releases": 0.0,
            "avg_requested_phase_override_steps": 0.0,
            "avg_meta_override_updates_skipped": 0.0,
            "terminal_reason_counts": {},
            "terminal_classification_counts": {},
            "requested_phase_counts": {},
            "executed_phase_counts": {},
            "requested_executed_phase_pair_counts": {},
            "invalid_action_reason_counts": {},
            "action_ack_reason_counts": {},
            "invalid_action_action_counts": {},
            "action_ack_timeout_action_counts": {},
        }

    episode_count = len(results)
    terminal_reason_counts = Counter(_terminal_reason_key(item.terminal_reason) for item in results)
    terminal_classification_counts = Counter(item.terminal_classification for item in results)
    requested_phase_counts: Counter[str] = Counter()
    executed_phase_counts: Counter[str] = Counter()
    requested_executed_phase_pair_counts: Counter[str] = Counter()
    invalid_action_reason_counts: Counter[str] = Counter()
    action_ack_reason_counts: Counter[str] = Counter()
    invalid_action_action_counts: Counter[str] = Counter()
    action_ack_timeout_action_counts: Counter[str] = Counter()
    for item in results:
        requested_phase_counts.update(item.requested_phase_counts)
        executed_phase_counts.update(item.executed_phase_counts)
        requested_executed_phase_pair_counts.update(item.requested_executed_phase_pair_counts)
        invalid_action_reason_counts.update(item.invalid_action_reason_counts)
        action_ack_reason_counts.update(item.action_ack_reason_counts)
        invalid_action_action_counts.update(item.invalid_action_action_counts)
        action_ack_timeout_action_counts.update(item.action_ack_timeout_action_counts)
    done_count = sum(1 for item in results if item.done)
    non_death_terminal_count = terminal_classification_counts.get("non_death_terminal", 0)
    unexpected_start_screen_count = terminal_classification_counts.get("unexpected_start_screen", 0)
    hit_step_limit_count = sum(1 for item in results if item.hit_step_limit)

    return {
        "episodes": episode_count,
        "done_episodes": done_count,
        "non_death_terminal_episodes": non_death_terminal_count,
        "unexpected_start_screen_episodes": unexpected_start_screen_count,
        "hit_step_limit_episodes": hit_step_limit_count,
        "done_rate": done_count / episode_count,
        "non_death_terminal_rate": non_death_terminal_count / episode_count,
        "unexpected_start_screen_rate": unexpected_start_screen_count / episode_count,
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
        "avg_threat_rewarded_steps": mean(item.threat_rewarded_steps for item in results),
        "avg_route_rejoin_events": mean(item.route_rejoin_events for item in results),
        "avg_phase_lock_overrides": mean(item.phase_lock_overrides for item in results),
        "avg_stall_releases": mean(item.stall_releases for item in results),
        "avg_requested_phase_override_steps": mean(item.requested_phase_override_steps for item in results),
        "avg_meta_override_updates_skipped": mean(item.meta_override_updates_skipped for item in results),
        "terminal_reason_counts": _sorted_counter_dict(terminal_reason_counts),
        "terminal_classification_counts": _sorted_counter_dict(terminal_classification_counts),
        "requested_phase_counts": _sorted_counter_dict(requested_phase_counts),
        "executed_phase_counts": _sorted_counter_dict(executed_phase_counts),
        "requested_executed_phase_pair_counts": _sorted_counter_dict(requested_executed_phase_pair_counts),
        "invalid_action_reason_counts": _sorted_counter_dict(invalid_action_reason_counts),
        "action_ack_reason_counts": _sorted_counter_dict(action_ack_reason_counts),
        "invalid_action_action_counts": _sorted_counter_dict(invalid_action_action_counts),
        "action_ack_timeout_action_counts": _sorted_counter_dict(action_ack_timeout_action_counts),
    }


def _print_results(results: tuple[HybridEpisodeSummary, ...]) -> None:
    print(
        "episode_id\tsteps\tdone\ttotal_reward\tmeta_reward\tthreat_reward\t"
        "invalid_actions\tpremature_exit\troute_len\treplans\tphase_switches\t"
        "threat_active_steps\tthreat_rewarded_steps\troute_rejoin_events\t"
        "phase_lock_overrides\tstall_releases\thit_step_limit\t"
        "terminal_classification\tterminal_reason"
    )
    for result in results:
        print(
            "{episode}\t{steps}\t{done}\t{total:.3f}\t{meta:.3f}\t{threat:.3f}\t"
            "{invalid}\t{premature}\t{route_len}\t{replans}\t{phase_switches}\t"
            "{threat_steps}\t{threat_rewarded_steps}\t{route_rejoin_events}\t"
            "{phase_lock_overrides}\t{stall_releases}\t{hit_step_limit}\t"
            "{terminal_classification}\t{reason}".format(
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
                threat_rewarded_steps=result.threat_rewarded_steps,
                route_rejoin_events=result.route_rejoin_events,
                phase_lock_overrides=result.phase_lock_overrides,
                stall_releases=result.stall_releases,
                hit_step_limit=result.hit_step_limit,
                terminal_classification=result.terminal_classification,
                reason=result.terminal_reason or "",
            )
        )
    summary = _build_training_summary(results)
    print(
        "\nsummary episodes={episodes} avg_steps={avg_steps:.2f} avg_total_reward={avg_total:.3f} "
        "done_rate={done_rate:.2%} non_death_terminal_rate={non_death_rate:.2%} "
        "unexpected_start_screen_rate={unexpected_start_rate:.2%} "
        "hit_step_limit_rate={hit_limit_rate:.2%} avg_invalid_actions={avg_invalid:.2f} "
        "avg_premature_exit={avg_premature:.2f} avg_phase_switches={avg_phase_switches:.2f} "
        "avg_threat_active_steps={avg_threat_steps:.2f} avg_threat_rewarded_steps={avg_rewarded_steps:.2f} "
        "avg_route_rejoin_events={avg_rejoin_events:.2f}".format(
            episodes=summary["episodes"],
            avg_steps=summary["avg_steps"],
            avg_total=summary["avg_total_reward"],
            done_rate=summary["done_rate"],
            non_death_rate=summary["non_death_terminal_rate"],
            unexpected_start_rate=summary["unexpected_start_screen_rate"],
            hit_limit_rate=summary["hit_step_limit_rate"],
            avg_invalid=summary["avg_invalid_actions"],
            avg_premature=summary["avg_premature_exit_attempts"],
            avg_phase_switches=summary["avg_phase_switches"],
            avg_threat_steps=summary["avg_threat_active_steps"],
            avg_rewarded_steps=summary["avg_threat_rewarded_steps"],
            avg_rejoin_events=summary["avg_route_rejoin_events"],
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
            "threat_rewarded_steps": item.threat_rewarded_steps,
            "route_rejoin_events": item.route_rejoin_events,
            "phase_lock_overrides": item.phase_lock_overrides,
            "stall_releases": item.stall_releases,
            "requested_phase_override_steps": item.requested_phase_override_steps,
            "requested_phase_counts": item.requested_phase_counts,
            "executed_phase_counts": item.executed_phase_counts,
            "requested_executed_phase_pair_counts": item.requested_executed_phase_pair_counts,
            "meta_override_updates_skipped": item.meta_override_updates_skipped,
            "invalid_action_reason_counts": item.invalid_action_reason_counts,
            "action_ack_reason_counts": item.action_ack_reason_counts,
            "invalid_action_action_counts": item.invalid_action_action_counts,
            "action_ack_timeout_action_counts": item.action_ack_timeout_action_counts,
            "start_screen_detected": item.start_screen_detected,
            "victory_detected": item.victory_detected,
            "victory_inferred_from_start_screen": item.victory_inferred_from_start_screen,
            "death_to_start_screen_detected": item.death_to_start_screen_detected,
            "start_screen_unknown_detected": item.start_screen_unknown_detected,
            "final_action": item.final_action,
            "last_known_player_position": item.last_known_player_position,
            "last_known_exit_position": item.last_known_exit_position,
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
    monitor_session: RunnerMonitorSession,
    monitor_enabled: bool,
    print_reward_breakdown: bool,
    meta_phase_override_credit_mode: str = "skip_overridden",
    victory_monitor_session: Any | None = None,
) -> tuple[HybridEpisodeSummary, ...]:
    results: list[HybridEpisodeSummary] = []

    for _ in range(episodes):
        state = env.reset()
        if victory_monitor_session is not None:
            victory_monitor_session.start_episode()
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
        threat_rewarded_steps = 0
        route_rejoin_events = 0
        updates_applied = 0
        last_phase: ObjectivePhase | None = None
        last_target_signature: tuple[str, int, int] | None = None
        last_stagnation_signature: tuple[str, int, int] | None = None
        objective_stagnation_steps = 0
        requested_phase_override_steps = 0
        meta_override_updates_skipped = 0
        requested_phase_counts: Counter[str] = Counter()
        executed_phase_counts: Counter[str] = Counter()
        requested_executed_phase_pair_counts: Counter[str] = Counter()
        invalid_action_reason_counts: Counter[str] = Counter()
        action_ack_reason_counts: Counter[str] = Counter()
        invalid_action_action_counts: Counter[str] = Counter()
        action_ack_timeout_action_counts: Counter[str] = Counter()
        last_threat_active = False
        last_threat_override = ThreatOverride.ROUTE_DEFAULT
        start_screen_detected = False
        victory_detected = False
        victory_inferred_from_start_screen = False
        death_to_start_screen_detected = False
        start_screen_unknown_detected = False
        final_action: str | None = None
        last_known_player_position: dict[str, int] | None = None
        last_known_exit_position: dict[str, int] | None = None
        hook_victory_signal_seen = False

        while steps < max_steps and not done:
            last_known_player_position, last_known_exit_position = _update_last_known_positions(
                state=state,
                last_player_position=last_known_player_position,
                last_exit_position=last_known_exit_position,
            )
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
            requested_phase_name = trace.requested_phase.value
            executed_phase_name = trace.decision.objective.phase.value
            requested_phase_counts[requested_phase_name] += 1
            executed_phase_counts[executed_phase_name] += 1
            requested_executed_phase_pair_counts[f"{requested_phase_name}->{executed_phase_name}"] += 1
            if trace.requested_phase != trace.decision.objective.phase:
                requested_phase_override_steps += 1
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
            else:
                target_signature = None

            threat_monitor_epsilon = (
                coordinator.threat_controller.epsilon
                if use_threat
                else None
            )
            if monitor_enabled:
                monitor_session.wait_for_step_gate(
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

            final_action = trace.decision.action
            next_state, _env_reward, done, info = env.step(trace.decision.action)
            hook_events = (
                victory_monitor_session.consume_new_events()
                if victory_monitor_session is not None
                else ()
            )
            for hook_event in hook_events:
                if _hook_event_is_victory_signal(hook_event):
                    hook_victory_signal_seen = True
            if (
                done
                and str(info.get("terminal_reason") or "").strip().lower() == "state:start_screen_unknown"
            ):
                if victory_monitor_session is not None and not hook_victory_signal_seen:
                    for _ in range(3):
                        time.sleep(0.05)
                        hook_events = victory_monitor_session.consume_new_events()
                        if not hook_events:
                            continue
                        for hook_event in hook_events:
                            if _hook_event_is_victory_signal(hook_event):
                                hook_victory_signal_seen = True
                        if hook_victory_signal_seen:
                            break
                if hook_victory_signal_seen:
                    info = {
                        **info,
                        "terminal_reason": "state:victory",
                        "victory_detected": True,
                        "victory_inferred_from_start_screen": True,
                        "start_screen_unknown_detected": False,
                    }
            last_known_player_position, last_known_exit_position = _update_last_known_positions(
                state=next_state,
                last_player_position=last_known_player_position,
                last_exit_position=last_known_exit_position,
            )
            terminal_reason = str(info.get("terminal_reason") or terminal_reason or "").strip() or None
            if trace.decision.used_fallback or info.get("invalid_action_reason") is not None:
                invalid_actions += 1
            invalid_action_reason = str(info.get("invalid_action_reason") or "").strip()
            if invalid_action_reason:
                invalid_action_reason_counts[invalid_action_reason] += 1
                invalid_action_action_counts[trace.decision.action] += 1
            action_ack_reason = str(info.get("action_ack_reason") or "").strip()
            if action_ack_reason:
                action_ack_reason_counts[action_ack_reason] += 1
                if action_ack_reason == "action_ack_timeout":
                    action_ack_timeout_action_counts[trace.decision.action] += 1
            if bool(info.get("premature_exit_attempt", False)):
                premature_exit_attempts += 1
            start_screen_detected = start_screen_detected or bool(info.get("start_screen_detected", False))
            victory_detected = victory_detected or bool(info.get("victory_detected", False))
            victory_inferred_from_start_screen = victory_inferred_from_start_screen or bool(
                info.get("victory_inferred_from_start_screen", False)
            )
            death_to_start_screen_detected = death_to_start_screen_detected or bool(
                info.get("death_to_start_screen_detected", False)
            )
            start_screen_unknown_detected = start_screen_unknown_detected or bool(
                info.get("start_screen_unknown_detected", False)
            )
            coordinator.observe_step_result(trace=trace, info=info, next_state=next_state)
            objective_distance_after = coordinator.objective_distance(
                state=next_state,
                target=target,
            )
            if target_signature is None or trace.objective_distance_before is None or objective_distance_after is None:
                objective_stagnation_steps = 0
                last_stagnation_signature = target_signature
            elif last_stagnation_signature != target_signature:
                objective_stagnation_steps = 0
                last_stagnation_signature = target_signature
            elif objective_distance_after < trace.objective_distance_before:
                objective_stagnation_steps = 0
            else:
                objective_stagnation_steps += 1
            hit_step_limit_imminent = bool((steps + 1) >= max_steps and not done)

            meta_breakdown = reward_suite.compute_meta_reward(
                previous_state=state,
                current_state=next_state,
                objective_phase=trace.decision.objective.phase,
                done=done,
                info={
                    **info,
                    "objective_target": target,
                    "objective_distance_before": trace.objective_distance_before,
                    "objective_distance_after": objective_distance_after,
                    "objective_stagnation_steps": objective_stagnation_steps,
                    "hit_step_limit": hit_step_limit_imminent,
                },
            )
            invalid_override = bool(
                trace.decision.used_fallback
                and trace.decision.threat_override != ThreatOverride.ROUTE_DEFAULT
            )
            route_rejoin_event = bool(
                use_threat
                and not trace.threat_active
                and trace.decision.threat_override == ThreatOverride.ROUTE_DEFAULT
                and (last_threat_active or last_threat_override != ThreatOverride.ROUTE_DEFAULT)
            )
            enemy_damage_dealt = enemy_damage_delta(previous_state=state, current_state=next_state)
            enemy_cleared = enemy_cleared_delta(previous_state=state, current_state=next_state)
            enemy_growth = enemy_growth_delta(previous_state=state, current_state=next_state)
            siphon_spawn_cost = (
                float(
                    siphon_spawn_cost_at_position(
                        state=state,
                        position=state.map.player_position if getattr(state.map, "status", None) == "ok" else None,
                    )
                )
                if bool(info.get("action_effective", False))
                and trace.decision.action in {"space", "z"}
                else 0.0
            )
            threat_signal_relevant = bool(
                use_threat
                and (
                    trace.threat_active
                    or route_rejoin_event
                    or invalid_override
                    or enemy_damage_dealt > 0.0
                    or enemy_cleared > 0.0
                    or enemy_growth > 0.0
                    or siphon_spawn_cost > 0.0
                    or health_damage_taken(previous_state=state, current_state=next_state) > 0.0
                    or (done and _terminal_is_fail(terminal_reason))
                )
            )
            threat_breakdown = (
                reward_suite.compute_threat_reward(
                    previous_state=state,
                    current_state=next_state,
                    done=done,
                    threat_override=trace.decision.threat_override,
                    info={
                        **info,
                        "action": trace.decision.action,
                        "enemy_damage_dealt": enemy_damage_dealt,
                        "enemy_cleared": enemy_cleared,
                        "enemy_growth": enemy_growth,
                        "siphon_spawn_cost": siphon_spawn_cost,
                        "invalid_override": invalid_override,
                        "route_rejoin_event": route_rejoin_event,
                    },
                )
                if threat_signal_relevant
                else _zero_threat_reward_breakdown()
            )
            step_reward = float(meta_breakdown.total + threat_breakdown.total)
            total_reward += step_reward
            meta_reward_total += float(meta_breakdown.total)
            threat_reward_total += float(threat_breakdown.total)
            if threat_signal_relevant:
                threat_rewarded_steps += 1
            if route_rejoin_event:
                route_rejoin_events += 1

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
                phase_overridden = trace.requested_phase != trace.decision.objective.phase
                if phase_overridden and meta_phase_override_credit_mode == "skip_overridden":
                    meta_override_updates_skipped += 1
                else:
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
            if train_threat and threat_signal_relevant:
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
                monitor_session.consume_manual_step_flag()
                monitor_session.update(
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
            last_threat_active = trace.threat_active
            last_threat_override = trace.decision.threat_override
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
                threat_rewarded_steps=threat_rewarded_steps,
                route_rejoin_events=route_rejoin_events,
                phase_lock_overrides=coordinator.phase_lock_overrides,
                stall_releases=coordinator.stall_releases,
                requested_phase_override_steps=requested_phase_override_steps,
                requested_phase_counts=dict(sorted(requested_phase_counts.items())),
                executed_phase_counts=dict(sorted(executed_phase_counts.items())),
                requested_executed_phase_pair_counts=dict(
                    sorted(requested_executed_phase_pair_counts.items())
                ),
                meta_override_updates_skipped=meta_override_updates_skipped,
                invalid_action_reason_counts=dict(sorted(invalid_action_reason_counts.items())),
                action_ack_reason_counts=dict(sorted(action_ack_reason_counts.items())),
                invalid_action_action_counts=dict(sorted(invalid_action_action_counts.items())),
                action_ack_timeout_action_counts=dict(sorted(action_ack_timeout_action_counts.items())),
                start_screen_detected=start_screen_detected,
                victory_detected=victory_detected,
                victory_inferred_from_start_screen=victory_inferred_from_start_screen,
                death_to_start_screen_detected=death_to_start_screen_detected,
                start_screen_unknown_detected=start_screen_unknown_detected,
                final_action=final_action,
                last_known_player_position=last_known_player_position,
                last_known_exit_position=last_known_exit_position,
            )
        )
    return tuple(results)

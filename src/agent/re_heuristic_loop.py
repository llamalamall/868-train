"""Automated analyze->synthesize->validate loop for re-heuristic growth."""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.agent.re_heuristic import (
    ReHeuristicAgent,
    ReHeuristicConfig,
    RuleDefinition,
    infer_fail_rate,
    save_rule_pack,
)
from src.state.schema import EnemyState, FieldState, GameStateSnapshot, GridPosition, MapState
from src.training.train import EpisodeRolloutResult, run_agent_policy

_INDEX_ROW_RE = re.compile(r"^\|\s*`(?P<name>fn_[0-9a-fA-F]+)`\s*\|\s*(?P<line>\d+)\s*\|\s*(?P<span>\d+)\s*\|$")
_PRIORITY_FN_HINTS = ("fn_1400428e0", "fn_140046250", "fn_140039ab0", "fn_14004d180")
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "map_cell_semantics": ("siphon", "cell_type", "wall", "exit", "overlay", "threat", "special"),
    "entity_behavior": ("virus", "daemon", "glitch", "cryptog", "state", "motion", "entity"),
    "prog_reward": ("prog", "credits", "energy", "score", "points", "0x1bb0", "0x1bb8"),
}
_CATEGORY_ACTIONS: dict[str, tuple[str, ...]] = {
    "map_cell_semantics": ("space", "move_right", "move_up", "wait"),
    "entity_behavior": ("move_right", "move_up", "move_left", "move_down", "wait"),
    "prog_reward": ("prog_slot_1", "prog_slot_2", "space", "move_right", "wait"),
}
_CATEGORY_CONDITIONS: dict[str, dict[str, Any]] = {
    "map_cell_semantics": {"require_siphons": True, "require_exit_visible": True},
    "entity_behavior": {"min_enemy_count": 1},
    "prog_reward": {"require_low_health": True},
}


@dataclass(frozen=True)
class FunctionRow:
    name: str
    line: int
    span: int


@dataclass(frozen=True)
class MinedCandidate:
    candidate_id: str
    signature: str
    category: str
    source_function: str
    evidence_refs: tuple[str, ...]
    evidence_hits: tuple[str, ...]
    novelty: float
    score: float
    suggested_reason: str
    action_preferences: tuple[str, ...]
    conditions: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyKpi:
    avg_reward: float
    fail_rate: float
    episodes: int
    terminal_reasons: Mapping[str, int]


@dataclass(frozen=True)
class KpiGateResult:
    passed: bool
    reward_delta: float
    fail_rate_delta: float
    improved_reward: bool
    improved_fail_rate: bool
    major_reward_regression: bool
    major_fail_regression: bool


@dataclass(frozen=True)
class LiveValidationResult:
    executed: bool
    passed: bool
    fail_rate: float | None
    return_code: int
    reason: str
    command: tuple[str, ...]
    stdout_tail: tuple[str, ...] = ()


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str


@dataclass(frozen=True)
class LoopConfig:
    readable_dir: Path = Path("artifacts/readable")
    output_root: Path = Path("artifacts/re_heuristic")
    max_iterations: int = 200
    min_candidate_score: float = 0.55
    top_candidates: int = 6
    offline_episodes: int = 40
    offline_seed: int = 7
    fail_rate_regression_tolerance: float = 0.05
    reward_regression_tolerance: float = 0.10
    live_validation: str = "required"
    live_command: tuple[str, ...] = (
        "python",
        "-m",
        "src.cli",
        "run-re-heuristic",
        "--episodes",
        "2",
        "--max-steps",
        "80",
        "--no-tui",
    )
    live_timeout_seconds: float = 240.0

    @property
    def iterations_dir(self) -> Path:
        return self.output_root / "iterations"

    @property
    def accepted_pack_path(self) -> Path:
        return self.output_root / "accepted_rule_pack.json"

    @property
    def loop_state_path(self) -> Path:
        return self.output_root / "loop_state.json"


@dataclass
class _SyntheticBenchmarkEnv:
    action_space: tuple[str, ...] = ("move_left", "move_right", "wait", "space", "confirm", "prog_slot_1")

    def __post_init__(self) -> None:
        self.current_episode_id: str | None = None
        self._episode_index = 0
        self._x = 0
        self._steps = 0

    def reset(self) -> GameStateSnapshot:
        self._episode_index += 1
        self.current_episode_id = f"episode-{self._episode_index:05d}"
        self._x = 0
        self._steps = 0
        return self._snapshot(failed=False)

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, Any]]:
        if action == "move_right":
            self._x = min(2, self._x + 1)
            reward = 1.0
        elif action == "move_left":
            self._x = max(0, self._x - 1)
            reward = -0.2
        elif action == "wait":
            reward = -0.35
        elif action == "space":
            reward = 0.2
        elif action.startswith("prog_slot_"):
            reward = 0.05
        else:
            reward = -0.1
        self._steps += 1
        goal = self._x == 2
        timeout = self._steps >= 4 and not goal
        done = goal or timeout
        reason = "goal" if goal else ("timeout_fail" if timeout else None)
        if goal:
            reward += 2.5
        return (self._snapshot(failed=timeout), reward, done, {"terminal_reason": reason})

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        del state
        return self.action_space

    def close(self) -> None:
        return

    def _snapshot(self, *, failed: bool) -> GameStateSnapshot:
        return GameStateSnapshot(
            timestamp_utc="2026-03-07T00:00:00+00:00",
            health=FieldState(value=2, status="ok"),
            energy=FieldState(value=6, status="ok"),
            currency=FieldState(value=0, status="ok"),
            fail_state=FieldState(value=failed, status="ok"),
            map=MapState(
                status="ok",
                width=3,
                height=3,
                player_position=GridPosition(self._x, 0),
                exit_position=GridPosition(2, 0),
                siphons=(GridPosition(1, 0),),
                enemies=(EnemyState(slot=1, type_id=2, position=GridPosition(1, 1), hp=1, state=0, in_bounds=True),),
            ),
        )


def parse_function_index(index_path: Path) -> tuple[FunctionRow, ...]:
    if not index_path.exists():
        return ()
    rows: list[FunctionRow] = []
    for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _INDEX_ROW_RE.match(line.strip())
        if match:
            rows.append(FunctionRow(name=match.group("name"), line=int(match.group("line")), span=int(match.group("span"))))
    return tuple(rows)


def mine_candidates(*, readable_dir: Path, existing_signatures: set[str], min_score: float = 0.55) -> tuple[MinedCandidate, ...]:
    rows = parse_function_index(readable_dir / "index.md")
    names = [row.name for row in sorted(rows, key=lambda item: item.span, reverse=True)[:40]]
    for hint in _PRIORITY_FN_HINTS:
        if hint not in names:
            names.insert(0, hint)

    found: dict[str, MinedCandidate] = {}
    fn_dir = readable_dir / "functions"
    for name in names:
        fn_path = fn_dir / f"{name}.c"
        if not fn_path.exists():
            continue
        text = fn_path.read_text(encoding="utf-8", errors="ignore").lower()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            hits = tuple(sorted(token for token in keywords if token in text))
            if not hits:
                continue
            signature = f"{category}:{name}:{','.join(hits[:3])}"
            novelty = 0.0 if signature in existing_signatures else 1.0
            evidence = min(1.0, len(hits) / max(2, len(keywords) / 2))
            score = round(0.6 * evidence + 0.4 * novelty, 4)
            if score < min_score:
                continue
            candidate = MinedCandidate(
                candidate_id=f"cand-{sha1(signature.encode('utf-8')).hexdigest()[:12]}",
                signature=signature,
                category=category,
                source_function=name,
                evidence_refs=(str(fn_path),),
                evidence_hits=hits,
                novelty=novelty,
                score=score,
                suggested_reason=f"re_mined_{category}",
                action_preferences=_CATEGORY_ACTIONS[category],
                conditions=_CATEGORY_CONDITIONS[category],
            )
            previous = found.get(signature)
            if previous is None or candidate.score > previous.score:
                found[signature] = candidate
    return tuple(sorted(found.values(), key=lambda item: item.score, reverse=True))


def stage_candidate_rules(*, candidates: Sequence[MinedCandidate], existing_rule_ids: set[str], top_k: int = 6) -> tuple[RuleDefinition, ...]:
    staged: list[RuleDefinition] = []
    for idx, candidate in enumerate(candidates):
        if len(staged) >= top_k:
            break
        short_hash = sha1(candidate.signature.encode("utf-8")).hexdigest()[:8]
        rule_id = f"re_mined.{candidate.category}.{short_hash}"
        if rule_id in existing_rule_ids:
            continue
        staged.append(
            RuleDefinition(
                rule_id=rule_id,
                priority=200 - idx,
                reason=candidate.suggested_reason,
                evidence_refs=(*candidate.evidence_refs, f"signature:{candidate.signature}"),
                action_preferences=candidate.action_preferences,
                conditions=candidate.conditions,
            )
        )
    return tuple(staged)


def generate_rule_scenarios(rules: Sequence[RuleDefinition]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "rule_id": rule.rule_id,
            "conditions": dict(rule.conditions),
            "action_preferences": list(rule.action_preferences),
        }
        for rule in rules
    )


def evaluate_behavior_fires(*, rule_pack_path: Path, scenarios: Sequence[Mapping[str, Any]]) -> tuple[int, tuple[str, ...]]:
    agent = ReHeuristicAgent(config=ReHeuristicConfig(enable_mined_rules=True, rule_pack_path=str(rule_pack_path)))
    fired: list[str] = []
    for idx, scenario in enumerate(scenarios):
        state = _build_state_for_conditions(conditions=scenario.get("conditions", {}))
        prefs = scenario.get("action_preferences", [])
        if not isinstance(prefs, list):
            prefs = []
        action_space = tuple(dict.fromkeys(("move_left", "move_right", "move_up", "move_down", "wait", "space", "confirm", "prog_slot_1", "prog_slot_2", *tuple(str(item) for item in prefs))))
        _ = agent.select_action(state=state, action_space=action_space, rng=random.Random(idx + 17))
        rule_id = agent.last_decision_rule_id or ""
        if rule_id.startswith("re_mined."):
            fired.append(rule_id)
    return (len(fired), tuple(fired))


def run_offline_benchmark(*, previous_rule_pack: Path | None, candidate_rule_pack: Path | None, episodes: int, seed: int) -> tuple[PolicyKpi, PolicyKpi]:
    return (_run_policy_with_pack(rule_pack=previous_rule_pack, episodes=episodes, seed=seed), _run_policy_with_pack(rule_pack=candidate_rule_pack, episodes=episodes, seed=seed))


def evaluate_kpi_gate(*, previous: PolicyKpi, current: PolicyKpi, fail_rate_regression_tolerance: float, reward_regression_tolerance: float) -> KpiGateResult:
    reward_delta = float(current.avg_reward - previous.avg_reward)
    fail_delta = float(current.fail_rate - previous.fail_rate)
    improved_reward = reward_delta > 0.0
    improved_fail = fail_delta < 0.0
    reward_drop = max(abs(previous.avg_reward) * reward_regression_tolerance, 0.05)
    major_reward_regression = current.avg_reward < previous.avg_reward - reward_drop
    if previous.fail_rate > 0:
        major_fail_regression = current.fail_rate > previous.fail_rate * (1.0 + fail_rate_regression_tolerance)
    else:
        major_fail_regression = current.fail_rate > fail_rate_regression_tolerance
    passed = (improved_reward or improved_fail) and not major_reward_regression and not major_fail_regression
    return KpiGateResult(
        passed=passed,
        reward_delta=reward_delta,
        fail_rate_delta=fail_delta,
        improved_reward=improved_reward,
        improved_fail_rate=improved_fail,
        major_reward_regression=major_reward_regression,
        major_fail_regression=major_fail_regression,
    )


def run_live_validation(*, mode: str, command: Sequence[str], timeout_seconds: float, fail_rate_regression_tolerance: float, previous_fail_rate: float | None) -> LiveValidationResult:
    normalized = mode.strip().lower()
    if normalized == "disabled":
        return LiveValidationResult(executed=False, passed=True, fail_rate=None, return_code=0, reason="live_validation_disabled", command=tuple(command))
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True, timeout=max(timeout_seconds, 1.0), check=False)
    except Exception as exc:  # noqa: BLE001
        return LiveValidationResult(executed=True, passed=normalized == "optional", fail_rate=None, return_code=1, reason=f"live_validation_execution_error:{exc}", command=tuple(command))
    tail = tuple(completed.stdout.splitlines()[-12:])
    if completed.returncode != 0:
        return LiveValidationResult(executed=True, passed=normalized == "optional", fail_rate=None, return_code=int(completed.returncode), reason="live_validation_nonzero_exit", command=tuple(command), stdout_tail=tail)
    reasons = _parse_terminal_reasons_from_runner_output(completed.stdout)
    fail_rate = infer_fail_rate(reasons)
    if previous_fail_rate is None:
        passed = True
    else:
        passed = fail_rate <= previous_fail_rate * (1.0 + fail_rate_regression_tolerance)
    return LiveValidationResult(executed=True, passed=passed or normalized == "optional", fail_rate=fail_rate, return_code=int(completed.returncode), reason="live_validation_ok" if passed else "live_validation_fail_rate_regression", command=tuple(command), stdout_tail=tail)


def should_stop_loop(*, accepted_rule_count: int, target_rule_count: int, consecutive_no_meaningful: int, iteration_index: int, max_iterations: int) -> StopDecision:
    if accepted_rule_count >= target_rule_count:
        return StopDecision(True, "target_rule_count_reached")
    if consecutive_no_meaningful >= 2:
        return StopDecision(True, "no_meaningful_rules_for_two_iterations")
    if iteration_index >= max_iterations:
        return StopDecision(True, "max_iterations_reached")
    return StopDecision(False, "continue")


def run_loop(config: LoopConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.iterations_dir.mkdir(parents=True, exist_ok=True)
    baseline_rule_count = ReHeuristicAgent.baseline_rule_count()
    target_rule_count = baseline_rule_count * 10
    accepted_rules = _load_rule_pack_definitions(config.accepted_pack_path)
    accepted_rule_count = baseline_rule_count + len(accepted_rules)
    existing_rule_ids = {rule.rule_id for rule in accepted_rules}
    previous_live_fail_rate: float | None = None
    consecutive_no_meaningful = 0
    summaries: list[dict[str, Any]] = []

    for iteration in range(1, config.max_iterations + 1):
        stop = should_stop_loop(accepted_rule_count=accepted_rule_count, target_rule_count=target_rule_count, consecutive_no_meaningful=consecutive_no_meaningful, iteration_index=iteration - 1, max_iterations=config.max_iterations)
        if stop.should_stop:
            break
        iter_dir = config.iterations_dir / f"iter-{iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        candidates = mine_candidates(readable_dir=config.readable_dir, existing_signatures={f"rule:{rule.rule_id}" for rule in accepted_rules}, min_score=config.min_candidate_score)
        staged_rules = stage_candidate_rules(candidates=candidates, existing_rule_ids=existing_rule_ids, top_k=config.top_candidates)
        staged_pack = iter_dir / "staged_rule_pack.json"
        save_rule_pack(path=staged_pack, rules=staged_rules)
        scenarios = generate_rule_scenarios(staged_rules)
        (iter_dir / "generated_scenarios.json").write_text(json.dumps(list(scenarios), indent=2, sort_keys=True), encoding="utf-8")
        behavior_count, fired_rule_ids = evaluate_behavior_fires(rule_pack_path=staged_pack, scenarios=scenarios)

        combined_rules = _merge_rules(base_rules=accepted_rules, extra_rules=staged_rules)
        combined_pack = iter_dir / "combined_rule_pack.json"
        save_rule_pack(path=combined_pack, rules=combined_rules)
        previous_pack = config.accepted_pack_path if accepted_rules else None
        previous_kpi, current_kpi = run_offline_benchmark(previous_rule_pack=previous_pack, candidate_rule_pack=combined_pack, episodes=config.offline_episodes, seed=config.offline_seed)
        gate = evaluate_kpi_gate(previous=previous_kpi, current=current_kpi, fail_rate_regression_tolerance=config.fail_rate_regression_tolerance, reward_regression_tolerance=config.reward_regression_tolerance)
        live = run_live_validation(mode=config.live_validation, command=config.live_command, timeout_seconds=config.live_timeout_seconds, fail_rate_regression_tolerance=config.fail_rate_regression_tolerance, previous_fail_rate=previous_live_fail_rate)

        meaningful = behavior_count > 0 and gate.passed and live.passed and bool(staged_rules)
        if meaningful:
            accepted_rules = list(combined_rules)
            save_rule_pack(path=config.accepted_pack_path, rules=accepted_rules)
            existing_rule_ids = {rule.rule_id for rule in accepted_rules}
            accepted_rule_count = baseline_rule_count + len(accepted_rules)
            consecutive_no_meaningful = 0
            if live.fail_rate is not None:
                previous_live_fail_rate = live.fail_rate
        else:
            consecutive_no_meaningful += 1

        summary = {
            "iteration": iteration,
            "candidate_count": len(candidates),
            "staged_rule_count": len(staged_rules),
            "behavior_fire_count": behavior_count,
            "fired_rule_ids": list(fired_rule_ids),
            "offline_previous_kpi": asdict(previous_kpi),
            "offline_current_kpi": asdict(current_kpi),
            "kpi_gate": asdict(gate),
            "live_validation": asdict(live),
            "meaningful": meaningful,
            "accepted_rule_count": accepted_rule_count,
            "target_rule_count": target_rule_count,
            "consecutive_no_meaningful": consecutive_no_meaningful,
        }
        (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (iter_dir / "candidates.json").write_text(json.dumps([asdict(item) for item in candidates], indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(summary)
        stop = should_stop_loop(accepted_rule_count=accepted_rule_count, target_rule_count=target_rule_count, consecutive_no_meaningful=consecutive_no_meaningful, iteration_index=iteration, max_iterations=config.max_iterations)
        if stop.should_stop:
            break

    stop = should_stop_loop(accepted_rule_count=accepted_rule_count, target_rule_count=target_rule_count, consecutive_no_meaningful=consecutive_no_meaningful, iteration_index=len(summaries), max_iterations=config.max_iterations)
    result = {
        "baseline_rule_count": baseline_rule_count,
        "target_rule_count": target_rule_count,
        "accepted_rule_count": accepted_rule_count,
        "iterations_executed": len(summaries),
        "consecutive_no_meaningful": consecutive_no_meaningful,
        "stop_reason": stop.reason,
        "accepted_rule_pack_path": str(config.accepted_pack_path),
        "live_validation_mode": config.live_validation,
        "iterations": summaries,
    }
    config.loop_state_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run iterative re-heuristic RE rule-mining loop.")
    parser.add_argument("--readable-dir", default="artifacts/readable")
    parser.add_argument("--output-root", default="artifacts/re_heuristic")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--min-candidate-score", type=float, default=0.55)
    parser.add_argument("--top-candidates", type=int, default=6)
    parser.add_argument("--offline-episodes", type=int, default=40)
    parser.add_argument("--offline-seed", type=int, default=7)
    parser.add_argument("--fail-rate-regression-tolerance", type=float, default=0.05)
    parser.add_argument("--reward-regression-tolerance", type=float, default=0.10)
    parser.add_argument("--live-validation", choices=("required", "optional", "disabled"), default="required")
    parser.add_argument("--live-command", default="python -m src.cli run-re-heuristic --episodes 2 --max-steps 80 --no-tui")
    parser.add_argument("--live-timeout", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    parsed_live_command = tuple(shlex.split(str(args.live_command), posix=False))
    config = LoopConfig(
        readable_dir=Path(str(args.readable_dir)),
        output_root=Path(str(args.output_root)),
        max_iterations=max(int(args.max_iterations), 1),
        min_candidate_score=float(args.min_candidate_score),
        top_candidates=max(int(args.top_candidates), 1),
        offline_episodes=max(int(args.offline_episodes), 1),
        offline_seed=int(args.offline_seed),
        fail_rate_regression_tolerance=max(float(args.fail_rate_regression_tolerance), 0.0),
        reward_regression_tolerance=max(float(args.reward_regression_tolerance), 0.0),
        live_validation=str(args.live_validation),
        live_command=parsed_live_command if parsed_live_command else LoopConfig().live_command,
        live_timeout_seconds=max(float(args.live_timeout), 1.0),
    )
    result = run_loop(config)
    print(f"re_heuristic_loop stop_reason={result['stop_reason']} iterations={result['iterations_executed']} accepted_rules={result['accepted_rule_count']}/{result['target_rule_count']}")
    print(f"state_file={config.loop_state_path}")


def _run_policy_with_pack(*, rule_pack: Path | None, episodes: int, seed: int) -> PolicyKpi:
    env = _SyntheticBenchmarkEnv()
    try:
        agent = ReHeuristicAgent(config=ReHeuristicConfig(enable_mined_rules=rule_pack is not None, rule_pack_path=str(rule_pack) if rule_pack is not None else None))
        results = run_agent_policy(env=env, agent=agent, episodes=episodes, max_steps_per_episode=8, seed=seed)
    finally:
        env.close()
    return _results_to_kpi(results)


def _results_to_kpi(results: Sequence[EpisodeRolloutResult]) -> PolicyKpi:
    terminal_counts: dict[str, int] = {}
    reasons: list[str | None] = []
    for result in results:
        reasons.append(result.terminal_reason)
        if result.terminal_reason:
            terminal_counts[result.terminal_reason] = terminal_counts.get(result.terminal_reason, 0) + 1
    return PolicyKpi(avg_reward=float(mean(result.total_reward for result in results) if results else 0.0), fail_rate=infer_fail_rate(reasons), episodes=len(results), terminal_reasons=terminal_counts)


def _merge_rules(*, base_rules: Sequence[RuleDefinition], extra_rules: Sequence[RuleDefinition]) -> tuple[RuleDefinition, ...]:
    merged: list[RuleDefinition] = []
    seen: set[str] = set()
    for rule in (*base_rules, *extra_rules):
        if rule.rule_id in seen:
            continue
        seen.add(rule.rule_id)
        merged.append(rule)
    return tuple(merged)


def _build_state_for_conditions(*, conditions: Mapping[str, Any]) -> GameStateSnapshot:
    require_siphons = bool(conditions.get("require_siphons", False))
    require_exit_visible = bool(conditions.get("require_exit_visible", True))
    require_low_health = bool(conditions.get("require_low_health", False))
    min_enemy_count = int(conditions.get("min_enemy_count", 0) or 0)
    require_adjacent_enemy = bool(conditions.get("require_adjacent_enemy", False))
    player = GridPosition(0, 0)
    enemy_count = max(min_enemy_count, 1 if require_adjacent_enemy else 0)
    enemies = tuple(
        EnemyState(
            slot=idx + 1,
            type_id=2,
            position=GridPosition(1, 0) if idx == 0 and require_adjacent_enemy else GridPosition(min(idx + 1, 2), 1),
            hp=1,
            state=0,
            in_bounds=True,
        )
        for idx in range(enemy_count)
    )
    return GameStateSnapshot(
        timestamp_utc="2026-03-07T00:00:00+00:00",
        health=FieldState(value=2 if require_low_health else 9, status="ok"),
        energy=FieldState(value=8, status="ok"),
        currency=FieldState(value=0, status="ok"),
        fail_state=FieldState(value=False, status="ok"),
        map=MapState(
            status="ok",
            width=3,
            height=3,
            player_position=player,
            exit_position=GridPosition(2, 0) if require_exit_visible else None,
            siphons=(GridPosition(1, 0),) if require_siphons else (),
            enemies=enemies,
        ),
    )


def _load_rule_pack_definitions(path: Path) -> list[RuleDefinition]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        return []
    parsed: list[RuleDefinition] = []
    seen: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            continue
        try:
            rule = RuleDefinition.from_mapping(raw_rule)
        except (TypeError, ValueError):
            continue
        if rule.rule_id in seen:
            continue
        seen.add(rule.rule_id)
        parsed.append(rule)
    return parsed


def _parse_terminal_reasons_from_runner_output(output: str) -> tuple[str | None, ...]:
    reasons: list[str | None] = []
    for line in output.splitlines():
        if "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        if not parts[0].strip().lower().startswith("episode"):
            continue
        reason = parts[4].strip()
        reasons.append(reason or None)
    return tuple(reasons)


def _is_failure_reason(reason: str | None) -> bool:
    text = (reason or "").strip().lower()
    return any(token in text for token in ("fail", "loss", "dead", "start_screen", "timeout"))


__all__ = [
    "KpiGateResult",
    "LiveValidationResult",
    "LoopConfig",
    "MinedCandidate",
    "PolicyKpi",
    "StopDecision",
    "build_parser",
    "evaluate_behavior_fires",
    "evaluate_kpi_gate",
    "generate_rule_scenarios",
    "main",
    "mine_candidates",
    "parse_function_index",
    "run_live_validation",
    "run_loop",
    "run_offline_benchmark",
    "should_stop_loop",
    "stage_candidate_rules",
]


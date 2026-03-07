"""Re-heuristic agent with a rule-registry shell over baseline heuristics."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.agent.baseline_heuristic import HeuristicBaselineAgent, HeuristicBaselineConfig
from src.state.schema import EnemyState, GameStateSnapshot, GridPosition

LOGGER = logging.getLogger(__name__)

RulePredicate = Callable[["RuleContext"], bool]
RuleActionSelector = Callable[["RuleContext"], str | None]

_FAIL_REASON_TOKENS = ("fail", "loss", "dead", "start_screen")


@dataclass(frozen=True)
class ReHeuristicConfig:
    """Config for re-heuristic policy behavior and external rule loading."""

    low_health_threshold: int = 3
    enemy_prediction_horizon_steps: int = 2
    verbose_action_logging: bool = False
    enable_mined_rules: bool = True
    rule_pack_path: str | None = None

    def to_baseline_config(self) -> HeuristicBaselineConfig:
        """Translate re-heuristic knobs into baseline heuristic knobs."""
        return HeuristicBaselineConfig(
            low_health_threshold=int(self.low_health_threshold),
            enemy_prediction_horizon_steps=max(int(self.enemy_prediction_horizon_steps), 0),
            verbose_action_logging=bool(self.verbose_action_logging),
        )


@dataclass(frozen=True)
class RuleContext:
    """One decision context passed to rule predicates/selectors."""

    state: GameStateSnapshot
    action_space: tuple[str, ...]
    rng: random.Random
    baseline_action: str
    baseline_reason: str


@dataclass(frozen=True)
class RuleSpec:
    """Compiled rule in the re-heuristic registry."""

    rule_id: str
    priority: int
    reason: str
    evidence_refs: tuple[str, ...]
    predicate: RulePredicate
    action_selector: RuleActionSelector


@dataclass(frozen=True)
class RuleDefinition:
    """Serializable rule definition used by loop-generated rule packs."""

    rule_id: str
    priority: int
    reason: str
    evidence_refs: tuple[str, ...] = ()
    action_preferences: tuple[str, ...] = ()
    conditions: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RuleDefinition":
        rule_id = str(raw.get("rule_id", "")).strip()
        if not rule_id:
            raise ValueError("rule_id is required for mined rules.")
        priority = int(raw.get("priority", 0))
        reason = str(raw.get("reason", rule_id)).strip() or rule_id
        evidence_refs = tuple(str(item) for item in raw.get("evidence_refs", ()))
        action_preferences = tuple(str(item) for item in raw.get("action_preferences", ()))
        raw_conditions = raw.get("conditions", {})
        conditions = raw_conditions if isinstance(raw_conditions, Mapping) else {}
        return cls(
            rule_id=rule_id,
            priority=priority,
            reason=reason,
            evidence_refs=evidence_refs,
            action_preferences=action_preferences,
            conditions=conditions,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "priority": self.priority,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "action_preferences": list(self.action_preferences),
            "conditions": dict(self.conditions),
        }


def baseline_rule_reasons() -> tuple[str, ...]:
    """Known baseline reason labels used to seed iteration zero rule registry."""
    return (
        "attack_adjacent_virus_priority",
        "survival_safe_move_over_attack",
        "attack_enemy_in_line_of_sight",
        "use_prog_delay_emergency",
        "use_prog_anti_v_emergency",
        "use_prog_show_recon",
        "use_prog_debug_recon",
        "use_prog_step_unblock",
        "use_prog_delay_when_all_moves_dangerous",
        "use_prog_anti_v_when_all_moves_dangerous",
        "use_prog_step_when_all_moves_dangerous",
        "use_prog_show_when_all_moves_dangerous",
        "use_prog_debug_when_all_moves_dangerous",
        "use_prog_generic_when_all_moves_dangerous",
        "fallback_random_when_all_moves_dangerous",
        "harvest_resources",
        "harvest_progs",
        "harvest_points",
        "move_to_resources_target",
        "move_to_progs_target",
        "move_to_points_target",
        "low_health_wait",
        "collect_siphon",
        "move_toward_exit",
        "fallback_confirm",
        "fallback_random",
    )


@dataclass
class ReHeuristicAgent:
    """Rule-engine wrapper around baseline behavior with optional mined rules."""

    config: ReHeuristicConfig = ReHeuristicConfig()
    _baseline_agent: HeuristicBaselineAgent = field(init=False, repr=False)
    _rule_registry: tuple[RuleSpec, ...] = field(init=False, repr=False)
    _last_decision_reason: str | None = field(default=None, init=False, repr=False)
    _last_decision_rule_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._baseline_agent = HeuristicBaselineAgent(config=self.config.to_baseline_config())
        self._rule_registry = self._build_registry()

    @property
    def last_decision_reason(self) -> str | None:
        return self._last_decision_reason

    @property
    def last_decision_rule_id(self) -> str | None:
        return self._last_decision_rule_id

    @property
    def rule_count(self) -> int:
        return len(self._rule_registry)

    @classmethod
    def baseline_rule_count(cls) -> int:
        return len(baseline_rule_reasons())

    def registry_metadata(self) -> tuple[dict[str, Any], ...]:
        """Expose ordered rule metadata for loop auditing/tests."""
        return tuple(
            {
                "rule_id": rule.rule_id,
                "priority": rule.priority,
                "reason": rule.reason,
                "evidence_refs": list(rule.evidence_refs),
            }
            for rule in self._rule_registry
        )

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Select one action using rule registry with baseline fallback parity."""
        actions = tuple(action_space)
        if not actions:
            raise ValueError("action_space must include at least one action.")

        baseline_action = self._baseline_agent.select_action(state=state, action_space=actions, rng=rng)
        baseline_reason = self._baseline_agent.last_decision_reason or "baseline_unknown_reason"
        context = RuleContext(
            state=state,
            action_space=actions,
            rng=rng,
            baseline_action=baseline_action,
            baseline_reason=baseline_reason,
        )

        for rule in self._rule_registry:
            if not rule.predicate(context):
                continue
            selected = rule.action_selector(context)
            if selected is None or selected not in actions:
                continue
            self._last_decision_reason = rule.reason
            self._last_decision_rule_id = rule.rule_id
            self._log_choice(state=state, action=selected, reason=rule.reason, rule_id=rule.rule_id)
            return selected

        self._last_decision_reason = baseline_reason
        self._last_decision_rule_id = "fallback.baseline_delegate"
        self._log_choice(
            state=state,
            action=baseline_action,
            reason=baseline_reason,
            rule_id="fallback.baseline_delegate",
        )
        return baseline_action

    def _build_registry(self) -> tuple[RuleSpec, ...]:
        compiled: list[RuleSpec] = []
        baseline_priority = 1000
        for offset, reason in enumerate(baseline_rule_reasons()):
            compiled.append(
                RuleSpec(
                    rule_id=f"baseline.{reason}",
                    priority=baseline_priority - offset,
                    reason=reason,
                    evidence_refs=("baseline_heuristic",),
                    predicate=_baseline_reason_predicate(reason),
                    action_selector=_baseline_action_selector,
                )
            )

        if self.config.enable_mined_rules:
            for definition in self._load_rule_pack(Path(self.config.rule_pack_path) if self.config.rule_pack_path else None):
                compiled.append(_compile_loaded_rule(definition))

        compiled.sort(key=lambda item: item.priority, reverse=True)
        return tuple(compiled)

    def _load_rule_pack(self, path: Path | None) -> tuple[RuleDefinition, ...]:
        if path is None:
            return ()
        if not path.exists():
            LOGGER.warning("Rule-pack path not found: %s", path)
            return ()

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed reading rule pack %s: %s", path, exc)
            return ()

        raw_rules = payload.get("rules", [])
        if not isinstance(raw_rules, list):
            return ()

        parsed: list[RuleDefinition] = []
        seen: set[str] = set()
        for raw_rule in raw_rules:
            if not isinstance(raw_rule, Mapping):
                continue
            try:
                definition = RuleDefinition.from_mapping(raw_rule)
            except (TypeError, ValueError):
                continue
            if definition.rule_id in seen:
                continue
            seen.add(definition.rule_id)
            parsed.append(definition)
        return tuple(parsed)

    def _log_choice(self, *, state: GameStateSnapshot, action: str, reason: str, rule_id: str) -> None:
        if not self.config.verbose_action_logging:
            return
        LOGGER.info(
            "re_heuristic_action choice=%s reason=%s rule=%s health=%s player=%s exit=%s",
            action,
            reason,
            rule_id,
            state.health.value if state.health.status == "ok" else None,
            state.map.player_position,
            state.map.exit_position,
        )


def save_rule_pack(*, path: str | Path, rules: Sequence[RuleDefinition]) -> None:
    """Persist loop-generated rules in the format consumed by ReHeuristicAgent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rules": [rule.to_mapping() for rule in rules]}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_reason_predicate(reason: str) -> RulePredicate:
    def _predicate(context: RuleContext) -> bool:
        return context.baseline_reason == reason and context.baseline_action in context.action_space

    return _predicate


def _baseline_action_selector(context: RuleContext) -> str | None:
    return context.baseline_action


def _compile_loaded_rule(definition: RuleDefinition) -> RuleSpec:
    action_preferences = tuple(action for action in definition.action_preferences if action)
    predicate = _compiled_conditions_predicate(definition.conditions)

    def _selector(context: RuleContext) -> str | None:
        for action in action_preferences:
            if action in context.action_space:
                return action
        return context.baseline_action

    return RuleSpec(
        rule_id=definition.rule_id,
        priority=int(definition.priority),
        reason=definition.reason,
        evidence_refs=definition.evidence_refs,
        predicate=predicate,
        action_selector=_selector,
    )


def _compiled_conditions_predicate(conditions: Mapping[str, Any]) -> RulePredicate:
    require_siphons = _coerce_optional_bool(conditions.get("require_siphons"))
    require_exit_visible = _coerce_optional_bool(conditions.get("require_exit_visible"))
    require_adjacent_enemy = _coerce_optional_bool(conditions.get("require_adjacent_enemy"))
    require_low_health = _coerce_optional_bool(conditions.get("require_low_health"))
    require_threat_cells = _coerce_optional_bool(conditions.get("require_threat_cells"))
    require_special_state = _coerce_optional_bool(conditions.get("require_special_state"))
    require_exit_overlay = _coerce_optional_bool(conditions.get("require_exit_overlay"))
    min_enemy_count = _coerce_optional_int(conditions.get("min_enemy_count"))
    max_health = _coerce_optional_int(conditions.get("max_health"))

    def _predicate(context: RuleContext) -> bool:
        state = context.state
        if require_siphons is not None:
            has_siphons = state.map.status == "ok" and bool(state.map.siphons)
            if has_siphons != require_siphons:
                return False

        if require_exit_visible is not None:
            has_exit = state.map.status == "ok" and state.map.exit_position is not None
            if has_exit != require_exit_visible:
                return False

        if require_adjacent_enemy is not None:
            adjacent = _has_adjacent_enemy(state=state)
            if adjacent != require_adjacent_enemy:
                return False

        if require_low_health is not None:
            low_health = _is_low_health(state=state, max_health=max_health)
            if low_health != require_low_health:
                return False

        if min_enemy_count is not None:
            enemy_count = len(state.map.enemies) if state.map.status == "ok" else 0
            if enemy_count < min_enemy_count:
                return False

        if require_threat_cells is not None:
            has_threat = state.map.status == "ok" and any(cell.threat > 0 for cell in state.map.cells)
            if has_threat != require_threat_cells:
                return False

        if require_special_state is not None:
            has_special_state = (
                state.map.status == "ok" and any(cell.special_state > 0 for cell in state.map.cells)
            )
            if has_special_state != require_special_state:
                return False

        if require_exit_overlay is not None:
            has_exit_overlay = (
                state.map.status == "ok" and any(cell.has_exit_overlay for cell in state.map.cells)
            )
            if has_exit_overlay != require_exit_overlay:
                return False

        if max_health is not None and state.health.status == "ok":
            value = state.health.value
            if isinstance(value, bool):
                return False
            if isinstance(value, int | float):
                if int(value) > max_health:
                    return False
        return True

    return _predicate


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _is_low_health(*, state: GameStateSnapshot, max_health: int | None) -> bool:
    if state.health.status != "ok":
        return False
    value = state.health.value
    if value is None or isinstance(value, bool):
        return False
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return False
    threshold = 3 if max_health is None else max_health
    return numeric <= threshold


def _has_adjacent_enemy(*, state: GameStateSnapshot) -> bool:
    if state.map.status != "ok" or state.map.player_position is None:
        return False
    player = state.map.player_position
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        if _enemy_can_attack_position(enemy_type=enemy.type_id, enemy_position=enemy.position, player_position=player):
            return True
    return False


def _enemy_can_attack_position(
    *,
    enemy_type: int,
    enemy_position: GridPosition,
    player_position: GridPosition,
) -> bool:
    if enemy_type == 2:  # virus, diagonal-inclusive attack shape
        return max(abs(enemy_position.x - player_position.x), abs(enemy_position.y - player_position.y)) <= 1
    return abs(enemy_position.x - player_position.x) + abs(enemy_position.y - player_position.y) <= 1


def infer_fail_rate(results: Sequence[str | None]) -> float:
    """Infer fail rate from terminal reason labels."""
    if not results:
        return 0.0
    failures = 0
    for reason in results:
        text = (reason or "").strip().lower()
        if any(token in text for token in _FAIL_REASON_TOKENS):
            failures += 1
    return failures / len(results)


def visible_enemy_count(state: GameStateSnapshot) -> int:
    """Utility used by loop/scenario generation to count visible hostiles."""
    if state.map.status != "ok":
        return 0
    return sum(1 for enemy in state.map.enemies if isinstance(enemy, EnemyState) and enemy.in_bounds)


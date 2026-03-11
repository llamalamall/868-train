"""Simple heuristic baseline policy."""

from __future__ import annotations

import heapq
import logging
import random
from dataclasses import dataclass, field
from typing import Literal, Sequence

from src.state.schema import EnemyState, GameStateSnapshot, GridPosition, MapCellState

LOGGER = logging.getLogger(__name__)

_MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}
_PROG_ACTION_BY_SLOT_INDEX: tuple[str, ...] = tuple(f"prog_slot_{slot}" for slot in range(1, 11))
_PROG_ID_SHOW = 2
_PROG_ID_DELAY = 7
_PROG_ID_STEP = 8
_PROG_ID_ANTI_V = 9
_PROG_ID_DEBUG = 10
_PROG_ID_SCORE = 17
_PROG_ID_HACK = 18
_ENEMY_TYPE_VIRUS = 2
_ENEMY_TYPE_CLASSIC = 3
_ENEMY_TYPE_GLITCH = 4
_ENEMY_TYPE_CRYPTOG = 5
_ENEMY_TYPE_DAEMON = 7


@dataclass(frozen=True)
class HeuristicBaselineConfig:
    """Tunable knobs for rule-based action selection."""

    low_health_threshold: int = 3
    verbose_action_logging: bool = False
    resource_goal_weight: float = 0.60
    prog_goal_weight: float = 0.30
    points_goal_weight: float = 0.10
    enable_prog_usage: bool = True
    prog_energy_floor: int = 4
    prog_retry_backoff_steps: int = 4
    show_recast_gap_steps: int = 6
    enemy_prediction_horizon_steps: int = 2


@dataclass(frozen=True)
class _HarvestPlan:
    category: Literal["resources", "progs", "points"]
    target_position: GridPosition


@dataclass(frozen=True)
class _WallCandidate:
    position: GridPosition
    prog_id: int | None
    points: int


@dataclass
class HeuristicBaselineAgent:
    """Rule-based policy that prioritizes survival and visible objectives."""

    config: HeuristicBaselineConfig = HeuristicBaselineConfig()
    _last_siphon_count: int | None = field(default=None, init=False, repr=False)
    _harvest_plan: _HarvestPlan | None = field(default=None, init=False, repr=False)
    _decision_step: int = field(default=0, init=False, repr=False)
    _last_prog_action: str | None = field(default=None, init=False, repr=False)
    _last_prog_energy: int | None = field(default=None, init=False, repr=False)
    _prog_backoff_steps: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_show_step: int | None = field(default=None, init=False, repr=False)
    _last_decision_reason: str | None = field(default=None, init=False, repr=False)

    @property
    def last_decision_reason(self) -> str | None:
        return self._last_decision_reason

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Choose an action from available actions using simple tactical rules."""
        actions = tuple(action_space)
        if not actions:
            raise ValueError("action_space must include at least one action.")

        reset_detected = self._update_harvest_plan_on_siphon_change(state=state, rng=rng)
        self._advance_prog_usage_state(state=state, reset_detected=reset_detected)
        enemy_action: str | None = None
        adjacent_virus_attack = False
        player_under_immediate_threat = False
        player_horizon_safe = True
        if state.map.status == "ok" and state.map.player_position is not None:
            adjacent_virus_action = self._select_adjacent_virus_attack(
                state=state,
                action_space=actions,
            )
            if adjacent_virus_action is not None:
                enemy_action = adjacent_virus_action
                adjacent_virus_attack = True
            else:
                enemy_action = self._select_enemy_sight_move(state=state, action_space=actions)

            enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
            player_under_immediate_threat = _position_takes_damage(
                position=state.map.player_position,
                enemies=enemies,
            )
            player_horizon_safe = self._position_is_safe_for_horizon(
                state=state,
                position=state.map.player_position,
            )

        actions = self._filter_exit_steps_while_siphons_remain(state=state, action_space=actions)
        if not actions:
            raise ValueError("No safe actions available after applying siphon-before-exit policy.")

        movement_actions = _navigable_movement_actions(state=state, action_space=actions)
        safe_movement_actions = self._safe_movement_actions(state=state, action_space=actions)
        if enemy_action is not None:
            if adjacent_virus_attack:
                return self._log_choice(
                    state=state,
                    action=enemy_action,
                    reason="attack_adjacent_virus_priority",
                    action_space=actions,
                )
            if (
                player_under_immediate_threat
                and safe_movement_actions
                and not player_horizon_safe
            ):
                survival_action = str(rng.choice(safe_movement_actions))
                if state.map.status == "ok" and state.map.player_position is not None:
                    siphon_survival_action = self._select_siphon_move(
                        state=state,
                        action_space=safe_movement_actions,
                    )
                    if siphon_survival_action is not None:
                        survival_action = siphon_survival_action
                    elif not state.map.siphons:
                        exit_survival_action = self._select_exit_move(
                            state=state,
                            action_space=safe_movement_actions,
                        )
                        if exit_survival_action is not None:
                            survival_action = exit_survival_action
                return self._log_choice(
                    state=state,
                    action=survival_action,
                    reason="survival_safe_move_over_attack",
                    action_space=actions,
                )
            return self._log_choice(
                state=state,
                action=enemy_action,
                reason="attack_enemy_in_line_of_sight",
                action_space=actions,
            )

        prog_action, prog_reason = self._select_prog_action(
            state=state,
            action_space=actions,
        )
        if prog_action is not None and prog_reason is not None:
            self._record_prog_action_attempt(action_name=prog_action, state=state)
            return self._log_choice(
                state=state,
                action=prog_action,
                reason=prog_reason,
                action_space=actions,
            )

        if movement_actions and not safe_movement_actions:
            escape_prog_action, escape_prog_reason = self._select_prog_escape_action(
                state=state,
                action_space=actions,
            )
            if escape_prog_action is not None and escape_prog_reason is not None:
                self._record_prog_action_attempt(action_name=escape_prog_action, state=state)
                return self._log_choice(
                    state=state,
                    action=escape_prog_action,
                    reason=escape_prog_reason,
                    action_space=actions,
                )
            return self._log_choice(
                state=state,
                action=str(rng.choice(movement_actions)),
                reason="fallback_random_when_all_moves_dangerous",
                action_space=actions,
            )

        safe_actions = _with_movement_actions_filtered(
            action_space=actions,
            allowed_movement_actions=safe_movement_actions if safe_movement_actions else movement_actions,
        )

        planned_action, planned_reason = self._execute_harvest_plan(state=state, action_space=safe_actions)
        if planned_action is not None and planned_reason is not None:
            return self._log_choice(
                state=state,
                action=planned_action,
                reason=planned_reason,
                action_space=safe_actions,
            )

        health_value = self._coerce_int(state.health.value) if state.health.status == "ok" else None
        if (
            health_value is not None
            and health_value <= self.config.low_health_threshold
            and "wait" in safe_actions
            and self._is_wait_safe(state=state)
        ):
            return self._log_choice(
                state=state,
                action="wait",
                reason="low_health_wait",
                action_space=safe_actions,
            )

        if state.map.status == "ok" and state.map.player_position is not None:
            siphon_action = self._select_siphon_move(state=state, action_space=safe_actions)
            if siphon_action is not None:
                return self._log_choice(
                    state=state,
                    action=siphon_action,
                    reason="collect_siphon",
                    action_space=safe_actions,
                )

            if not state.map.siphons:
                exit_action = self._select_exit_move(state=state, action_space=safe_actions)
                if exit_action is not None:
                    return self._log_choice(
                        state=state,
                        action=exit_action,
                        reason="move_toward_exit",
                        action_space=safe_actions,
                    )

        if "confirm" in safe_actions:
            return self._log_choice(
                state=state,
                action="confirm",
                reason="fallback_confirm",
                action_space=safe_actions,
            )

        return self._log_choice(
            state=state,
            action=str(rng.choice(safe_actions)),
            reason="fallback_random",
            action_space=safe_actions,
        )

    def _update_harvest_plan_on_siphon_change(
        self,
        *,
        state: GameStateSnapshot,
        rng: random.Random,
    ) -> bool:
        if state.map.status != "ok":
            self._last_siphon_count = None
            self._harvest_plan = None
            self._reset_prog_usage_state()
            return False

        current_siphon_count = len(state.map.siphons)
        if self._last_siphon_count is None:
            self._last_siphon_count = current_siphon_count
            return False

        reset_detected = False
        if current_siphon_count > self._last_siphon_count:
            # Likely reset/new episode.
            self._harvest_plan = None
            self._reset_prog_usage_state()
            reset_detected = True
        elif current_siphon_count < self._last_siphon_count:
            self._harvest_plan = None
            if current_siphon_count > 0:
                self._harvest_plan = self._choose_harvest_plan(state=state, rng=rng)

        self._last_siphon_count = current_siphon_count
        return reset_detected

    def _execute_harvest_plan(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        plan = self._harvest_plan
        if plan is None:
            return (None, None)

        if state.map.status != "ok" or state.map.player_position is None:
            self._harvest_plan = None
            return (None, None)
        if not state.map.siphons:
            self._harvest_plan = None
            return (None, None)

        player = state.map.player_position
        if player == plan.target_position:
            if "space" in action_space:
                self._harvest_plan = None
                return ("space", f"harvest_{plan.category}")
            self._harvest_plan = None
            return (None, None)

        route = _shortest_path_first_action(
            start=player,
            target=plan.target_position,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            allowed_first_actions=_movement_actions(action_space),
        )
        if route is None:
            self._harvest_plan = None
            return (None, None)

        return (route.action, f"move_to_{plan.category}_target")

    def _advance_prog_usage_state(
        self,
        *,
        state: GameStateSnapshot,
        reset_detected: bool,
    ) -> None:
        self._decision_step += 1
        for action_name in tuple(self._prog_backoff_steps):
            remaining = self._prog_backoff_steps[action_name] - 1
            if remaining <= 0:
                self._prog_backoff_steps.pop(action_name, None)
            else:
                self._prog_backoff_steps[action_name] = remaining

        if reset_detected:
            self._last_prog_action = None
            self._last_prog_energy = None
            return

        if self._last_prog_action is None:
            return

        current_energy = self._coerce_int(state.energy.value) if state.energy.status == "ok" else None
        energy_spent = (
            self._last_prog_energy is not None
            and current_energy is not None
            and current_energy < self._last_prog_energy
        )
        if not energy_spent and self.config.prog_retry_backoff_steps > 0:
            backoff_steps = max(int(self.config.prog_retry_backoff_steps), 0)
            existing = self._prog_backoff_steps.get(self._last_prog_action, 0)
            self._prog_backoff_steps[self._last_prog_action] = max(existing, backoff_steps)

        self._last_prog_action = None
        self._last_prog_energy = None

    def _reset_prog_usage_state(self) -> None:
        self._last_prog_action = None
        self._last_prog_energy = None
        self._prog_backoff_steps.clear()
        self._last_show_step = None

    def _record_prog_action_attempt(
        self,
        *,
        action_name: str,
        state: GameStateSnapshot,
    ) -> None:
        self._last_prog_action = action_name
        self._last_prog_energy = self._coerce_int(state.energy.value) if state.energy.status == "ok" else None

    def _select_prog_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        if not self.config.enable_prog_usage:
            return (None, None)
        if state.map.status != "ok":
            return (None, None)

        energy = self._coerce_int(state.energy.value) if state.energy.status == "ok" else None
        if energy is None or energy < self.config.prog_energy_floor:
            return (None, None)

        slot_prog_actions = self._slot_prog_actions(state=state, action_space=action_space)
        if not slot_prog_actions:
            return (None, None)

        has_enemy_pressure = self._has_enemy_pressure(state=state)
        low_health = self._is_low_health(state=state)

        if low_health and has_enemy_pressure:
            delay_action = self._find_prog_action_by_id(
                slot_prog_actions=slot_prog_actions,
                prog_id=_PROG_ID_DELAY,
            )
            if delay_action is not None:
                return (delay_action, "use_prog_delay_emergency")

            anti_v_action = self._find_prog_action_by_id(
                slot_prog_actions=slot_prog_actions,
                prog_id=_PROG_ID_ANTI_V,
            )
            if anti_v_action is not None and len(state.map.enemies) >= 2:
                return (anti_v_action, "use_prog_anti_v_emergency")

        if self._should_cast_show(state=state, has_enemy_pressure=has_enemy_pressure):
            show_action = self._find_prog_action_by_id(
                slot_prog_actions=slot_prog_actions,
                prog_id=_PROG_ID_SHOW,
            )
            if show_action is not None:
                self._last_show_step = self._decision_step
                return (show_action, "use_prog_show_recon")

        if self._should_cast_debug(state=state, has_enemy_pressure=has_enemy_pressure):
            debug_action = self._find_prog_action_by_id(
                slot_prog_actions=slot_prog_actions,
                prog_id=_PROG_ID_DEBUG,
            )
            if debug_action is not None:
                return (debug_action, "use_prog_debug_recon")

        if self._should_cast_step(state=state, has_enemy_pressure=has_enemy_pressure):
            step_action = self._find_prog_action_by_id(
                slot_prog_actions=slot_prog_actions,
                prog_id=_PROG_ID_STEP,
            )
            if step_action is not None:
                return (step_action, "use_prog_step_unblock")

        return (None, None)

    def _select_prog_escape_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        if not self.config.enable_prog_usage:
            return (None, None)
        if state.map.status != "ok":
            return (None, None)
        energy = self._coerce_int(state.energy.value) if state.energy.status == "ok" else None
        if energy is None or energy < self.config.prog_energy_floor:
            return (None, None)

        slot_prog_actions = self._slot_prog_actions(state=state, action_space=action_space)
        if not slot_prog_actions:
            return (None, None)

        for prog_id, reason in (
            (_PROG_ID_DELAY, "use_prog_delay_when_all_moves_dangerous"),
            (_PROG_ID_ANTI_V, "use_prog_anti_v_when_all_moves_dangerous"),
            (_PROG_ID_STEP, "use_prog_step_when_all_moves_dangerous"),
            (_PROG_ID_SHOW, "use_prog_show_when_all_moves_dangerous"),
            (_PROG_ID_DEBUG, "use_prog_debug_when_all_moves_dangerous"),
        ):
            action = self._find_prog_action_by_id(slot_prog_actions=slot_prog_actions, prog_id=prog_id)
            if action is not None:
                return (action, reason)

        for prog_id, actions in slot_prog_actions.items():
            if prog_id in {_PROG_ID_HACK, _PROG_ID_SCORE}:
                continue
            if actions:
                return (actions[0], "use_prog_generic_when_all_moves_dangerous")
        return (None, None)

    def _safe_movement_actions(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str, ...]:
        movement_actions = _movement_actions(action_space)
        if not movement_actions:
            return ()
        movement_actions = _navigable_movement_actions(state=state, action_space=action_space)
        if not movement_actions:
            return ()

        safe_actions: list[str] = []
        player = state.map.player_position
        if player is None:
            return ()
        for action in movement_actions:
            delta = _MOVE_VECTORS[action]
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            if not self._position_is_safe_for_horizon(state=state, position=candidate):
                continue
            safe_actions.append(action)
        return tuple(safe_actions)

    def _is_wait_safe(self, *, state: GameStateSnapshot) -> bool:
        if state.map.status != "ok" or state.map.player_position is None:
            return True
        return self._position_is_safe_for_horizon(state=state, position=state.map.player_position)

    def _position_is_safe_for_horizon(
        self,
        *,
        state: GameStateSnapshot,
        position: GridPosition,
    ) -> bool:
        if state.map.status != "ok":
            return True
        horizon = max(int(self.config.enemy_prediction_horizon_steps), 1)
        enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
        if not enemies:
            return True
        walls = _wall_positions(state)
        width = state.map.width
        height = state.map.height
        memo: dict[
            tuple[int, int, int, tuple[tuple[int, int, int, int], ...]],
            bool,
        ] = {}

        def can_survive(
            *,
            player_position: GridPosition,
            enemy_positions: tuple[EnemyState, ...],
            remaining_turns: int,
        ) -> bool:
            enemy_key = tuple(
                (enemy.type_id, enemy.position.x, enemy.position.y, enemy.slot)
                for enemy in enemy_positions
            )
            key = (player_position.x, player_position.y, remaining_turns, enemy_key)
            cached = memo.get(key)
            if cached is not None:
                return cached

            if remaining_turns <= 0:
                memo[key] = True
                return True

            enemies_next, player_took_damage = _simulate_enemy_turn(
                enemies=enemy_positions,
                player_position=player_position,
                width=width,
                height=height,
                walls=walls,
            )
            if player_took_damage:
                memo[key] = False
                return False

            if remaining_turns == 1:
                memo[key] = True
                return True

            for next_player_position in _next_player_positions(
                position=player_position,
                width=width,
                height=height,
                walls=walls,
            ):
                if can_survive(
                    player_position=next_player_position,
                    enemy_positions=enemies_next,
                    remaining_turns=remaining_turns - 1,
                ):
                    memo[key] = True
                    return True

            memo[key] = False
            return False

        return can_survive(
            player_position=position,
            enemy_positions=enemies,
            remaining_turns=horizon,
        )

    def _slot_prog_actions(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> dict[int, tuple[str, ...]]:
        if state.inventory.status != "ok":
            return {}

        available_actions = set(action_space)
        by_prog_id: dict[int, list[str]] = {}
        for slot_index, prog_id in enumerate(state.inventory.raw_prog_ids[:10]):
            action_name = _PROG_ACTION_BY_SLOT_INDEX[slot_index]
            if action_name not in available_actions:
                continue
            if self._prog_action_is_suppressed(action_name):
                continue
            by_prog_id.setdefault(int(prog_id), []).append(action_name)
        return {prog_id: tuple(actions) for prog_id, actions in by_prog_id.items()}

    @staticmethod
    def _find_prog_action_by_id(
        *,
        slot_prog_actions: dict[int, tuple[str, ...]],
        prog_id: int,
    ) -> str | None:
        actions = slot_prog_actions.get(prog_id)
        if not actions:
            return None
        return actions[0]

    def _prog_action_is_suppressed(self, action_name: str) -> bool:
        return self._prog_backoff_steps.get(action_name, 0) > 0

    def _is_low_health(self, *, state: GameStateSnapshot) -> bool:
        health_value = self._coerce_int(state.health.value) if state.health.status == "ok" else None
        return health_value is not None and health_value <= self.config.low_health_threshold

    def _has_enemy_pressure(self, *, state: GameStateSnapshot) -> bool:
        if state.map.status != "ok" or state.map.player_position is None:
            return False
        if not state.map.enemies:
            return False

        player = state.map.player_position
        nearest_enemy = min(_manhattan(player, enemy.position) for enemy in state.map.enemies)
        return nearest_enemy <= 1 or (nearest_enemy <= 2 and len(state.map.enemies) >= 2) or len(state.map.enemies) >= 4

    def _should_cast_show(self, *, state: GameStateSnapshot, has_enemy_pressure: bool) -> bool:
        if has_enemy_pressure:
            return False
        if not state.map.siphons:
            return False
        if self._count_unknown_walls(state=state) < 2:
            return False
        if self._last_show_step is None:
            return True
        return (self._decision_step - self._last_show_step) >= self.config.show_recast_gap_steps

    def _should_cast_debug(self, *, state: GameStateSnapshot, has_enemy_pressure: bool) -> bool:
        if has_enemy_pressure:
            return False
        player = state.map.player_position
        if player is None:
            return False

        reachable_walls = 0
        nearby_unknown = 0
        for wall in _iter_wall_candidates(state):
            if wall.prog_id is None and wall.points <= 0 and _manhattan(player, wall.position) <= 2:
                nearby_unknown += 1
            if _best_adjacent_position(player=player, target_wall=wall.position, state=state) is not None:
                reachable_walls += 1

        return nearby_unknown >= 1 and reachable_walls >= 2

    def _should_cast_step(self, *, state: GameStateSnapshot, has_enemy_pressure: bool) -> bool:
        if has_enemy_pressure:
            return False
        player = state.map.player_position
        if player is None:
            return False
        if state.map.siphons:
            targets = tuple(state.map.siphons)
        elif state.map.exit_position is not None:
            targets = (state.map.exit_position,)
        else:
            return False

        walls = _wall_positions(state)
        for target in targets:
            route = _shortest_path_first_action(
                start=player,
                target=target,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
                allowed_first_actions=tuple(_MOVE_VECTORS.keys()),
            )
            if route is not None:
                return False
        return True

    @staticmethod
    def _count_unknown_walls(*, state: GameStateSnapshot) -> int:
        return sum(
            1
            for wall in _iter_wall_candidates(state)
            if wall.prog_id is None and wall.points <= 0
        )

    def _choose_harvest_plan(
        self,
        *,
        state: GameStateSnapshot,
        rng: random.Random,
    ) -> _HarvestPlan | None:
        plans: dict[str, _HarvestPlan] = {}

        resource_plan = self._build_resource_plan(state=state)
        if resource_plan is not None:
            plans["resources"] = resource_plan

        prog_plan = self._build_prog_plan(state=state)
        if prog_plan is not None:
            plans["progs"] = prog_plan

        points_plan = self._build_points_plan(state=state)
        if points_plan is not None:
            plans["points"] = points_plan

        if not plans:
            return None

        weighted_options: list[tuple[str, float]] = []
        for category, weight in (
            ("resources", self.config.resource_goal_weight),
            ("progs", self.config.prog_goal_weight),
            ("points", self.config.points_goal_weight),
        ):
            if category in plans:
                weighted_options.append((category, max(float(weight), 0.0)))

        if not weighted_options:
            return None

        total_weight = sum(weight for _, weight in weighted_options)
        if total_weight <= 0:
            for category in ("resources", "progs", "points"):
                plan = plans.get(category)
                if plan is not None:
                    return plan
            return None

        pick = rng.random() * total_weight
        cumulative = 0.0
        for category, weight in weighted_options:
            cumulative += weight
            if pick <= cumulative:
                return plans[category]
        return plans[weighted_options[-1][0]]

    def _build_resource_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        walls = _wall_positions(state)
        movement_actions = tuple(_MOVE_VECTORS.keys())
        cells_by_position = _cells_by_position(state)
        best_target: GridPosition | None = None
        best_score = 0
        best_distance: int | None = None

        for x in range(state.map.width):
            for y in range(state.map.height):
                position = GridPosition(x=x, y=y)
                if position in walls:
                    continue
                score = _resource_cluster_score(
                    position=position,
                    width=state.map.width,
                    height=state.map.height,
                    cells_by_position=cells_by_position,
                )
                if score <= 0:
                    continue

                if position == player:
                    distance = 0
                else:
                    route = _shortest_path_first_action(
                        start=player,
                        target=position,
                        width=state.map.width,
                        height=state.map.height,
                        walls=walls,
                        allowed_first_actions=movement_actions,
                    )
                    if route is None:
                        continue
                    distance = route.distance

                if (
                    score > best_score
                    or (
                        score == best_score
                        and (best_distance is None or distance < best_distance)
                    )
                ):
                    best_score = score
                    best_distance = distance
                    best_target = position

        if best_target is None:
            return None
        return _HarvestPlan(category="resources", target_position=best_target)

    def _build_prog_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        preferred_prog_rank = {10: 0, 5: 1, 9: 2, 11: 3, 8: 4}
        best: tuple[int, int, GridPosition] | None = None

        for wall in _iter_wall_candidates(state):
            if wall.prog_id is None:
                continue
            candidate = _best_adjacent_position(
                player=player,
                target_wall=wall.position,
                state=state,
            )
            if candidate is None:
                continue
            target_position, distance = candidate
            rank = preferred_prog_rank.get(wall.prog_id, len(preferred_prog_rank) + 1)
            score = (rank, distance)
            if best is None or score < (best[0], best[1]):
                best = (rank, distance, target_position)

        if best is None:
            return None
        return _HarvestPlan(category="progs", target_position=best[2])

    def _build_points_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        best: tuple[int, int, GridPosition] | None = None
        for wall in _iter_wall_candidates(state):
            if wall.points <= 0:
                continue
            candidate = _best_adjacent_position(
                player=player,
                target_wall=wall.position,
                state=state,
            )
            if candidate is None:
                continue
            target_position, distance = candidate
            # Max points, then shortest path.
            score = (wall.points, -distance)
            if best is None or score > (best[0], best[1]):
                best = (wall.points, -distance, target_position)

        if best is None:
            return None
        return _HarvestPlan(category="points", target_position=best[2])

    def _filter_exit_steps_while_siphons_remain(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str, ...]:
        if state.map.status != "ok" or not state.map.siphons:
            return action_space
        if state.map.player_position is None or state.map.exit_position is None:
            return action_space

        player = state.map.player_position
        exit_position = state.map.exit_position
        filtered = tuple(
            action
            for action in action_space
            if not _action_steps_to_position(action=action, start=player, target=exit_position)
        )
        return filtered

    def _select_exit_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        exit_position = state.map.exit_position
        if player is None or exit_position is None:
            return None
        route = _shortest_path_first_action(
            start=player,
            target=exit_position,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            allowed_first_actions=_movement_actions(action_space),
        )
        return route.action if route is not None else None

    def _select_path_to_nearest_target(
        self,
        *,
        start: GridPosition,
        targets: tuple[GridPosition, ...],
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        if not targets:
            return None

        walls = _wall_positions(state)
        movement_actions = _movement_actions(action_space)
        best_action: str | None = None
        best_distance: int | None = None
        for target in targets:
            route = _shortest_path_first_action(
                start=start,
                target=target,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
                allowed_first_actions=movement_actions,
            )
            if route is None:
                continue
            if best_distance is None or route.distance < best_distance:
                best_distance = route.distance
                best_action = route.action
        return best_action

    def _select_siphon_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        if player is None or not state.map.siphons:
            return None

        return self._select_path_to_nearest_target(
            start=player,
            targets=tuple(state.map.siphons),
            state=state,
            action_space=action_space,
        )

    def _select_enemy_sight_move(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        player = state.map.player_position
        if player is None:
            return None

        wall_positions = _wall_positions(state)
        movement_actions = _movement_actions(action_space)
        if not movement_actions:
            return None

        target_enemy: tuple[int, GridPosition] | None = None
        target_distance: int | None = None
        for enemy in state.map.enemies:
            enemy_position = enemy.position
            if enemy_position.x == player.x:
                if not _is_vertical_line_clear(player=player, target=enemy_position, walls=wall_positions):
                    continue
            elif enemy_position.y == player.y:
                if not _is_horizontal_line_clear(player=player, target=enemy_position, walls=wall_positions):
                    continue
            else:
                continue

            distance = _manhattan(player, enemy_position)
            if target_distance is None or distance < target_distance:
                target_distance = distance
                target_enemy = (enemy.type_id, enemy_position)

        if target_enemy is None:
            return None

        _target_type, target_position = target_enemy
        return _first_step_toward_axis_aligned_target(
            start=player,
            target=target_position,
            allowed_actions=movement_actions,
        )

    def _select_adjacent_virus_attack(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        player = state.map.player_position
        if player is None:
            return None

        movement_actions = _movement_actions(action_space)
        if not movement_actions:
            return None

        walls = _wall_positions(state)
        for enemy in state.map.enemies:
            if not enemy.in_bounds or enemy.type_id != _ENEMY_TYPE_VIRUS:
                continue
            if _manhattan(player, enemy.position) != 1:
                continue
            if not _has_line_of_sight(player=player, target=enemy.position, walls=walls):
                continue
            action = _first_step_toward_axis_aligned_target(
                start=player,
                target=enemy.position,
                allowed_actions=movement_actions,
            )
            if action is not None:
                return action
        return None

    @staticmethod
    def _coerce_int(value: object | None) -> int | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _log_choice(
        self,
        *,
        state: GameStateSnapshot,
        action: str,
        reason: str,
        action_space: tuple[str, ...],
    ) -> str:
        self._last_decision_reason = reason
        if self.config.verbose_action_logging:
            LOGGER.info(
                "heuristic_action choice=%s reason=%s health=%s player=%s exit=%s available=%s",
                action,
                reason,
                state.health.value if state.health.status == "ok" else None,
                state.map.player_position,
                state.map.exit_position,
                ",".join(action_space),
            )
        return action


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _chebyshev(a: GridPosition, b: GridPosition) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def _with_movement_actions_filtered(
    *,
    action_space: tuple[str, ...],
    allowed_movement_actions: tuple[str, ...],
) -> tuple[str, ...]:
    allowed = set(allowed_movement_actions)
    return tuple(
        action
        for action in action_space
        if action not in _MOVE_VECTORS or action in allowed
    )


def _first_step_toward_axis_aligned_target(
    *,
    start: GridPosition,
    target: GridPosition,
    allowed_actions: tuple[str, ...],
) -> str | None:
    if start.x == target.x:
        action = "move_up" if target.y > start.y else "move_down"
        return action if action in allowed_actions else None
    if start.y == target.y:
        action = "move_right" if target.x > start.x else "move_left"
        return action if action in allowed_actions else None
    return None


def _position_takes_damage(
    *,
    position: GridPosition,
    enemies: tuple[EnemyState, ...],
) -> bool:
    for enemy in enemies:
        if not enemy.in_bounds:
            continue
        if _enemy_can_attack_position(
            enemy_type=enemy.type_id,
            enemy_position=enemy.position,
            player_position=position,
        ):
            return True
    return False


def _enemy_can_attack_position(
    *,
    enemy_type: int,
    enemy_position: GridPosition,
    player_position: GridPosition,
) -> bool:
    if enemy_type == _ENEMY_TYPE_VIRUS:
        return _chebyshev(enemy_position, player_position) <= 1
    return _manhattan(enemy_position, player_position) <= 1


def _simulate_enemy_turn(
    *,
    enemies: tuple[EnemyState, ...],
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> tuple[tuple[EnemyState, ...], bool]:
    predicted: list[EnemyState] = []
    took_damage = False
    for enemy in enemies:
        if not enemy.in_bounds:
            predicted.append(enemy)
            continue

        current_position = enemy.position
        if _enemy_can_attack_position(
            enemy_type=enemy.type_id,
            enemy_position=current_position,
            player_position=player_position,
        ):
            took_damage = True
            predicted.append(enemy)
            continue

        if enemy.type_id == _ENEMY_TYPE_VIRUS:
            first_step = _predict_enemy_substep(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
                width=width,
                height=height,
                walls=walls,
            )
            current_position = first_step
            if _enemy_can_attack_position(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
            ):
                took_damage = True
            else:
                current_position = _predict_enemy_substep(
                    enemy_type=enemy.type_id,
                    enemy_position=current_position,
                    player_position=player_position,
                    width=width,
                    height=height,
                    walls=walls,
                )
        else:
            current_position = _predict_enemy_substep(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
                width=width,
                height=height,
                walls=walls,
            )

        predicted.append(
            EnemyState(
                slot=enemy.slot,
                type_id=enemy.type_id,
                position=current_position,
                hp=enemy.hp,
                state=enemy.state,
                in_bounds=enemy.in_bounds,
            )
        )

    return (tuple(predicted), took_damage)


def _predict_enemies_one_turn(
    *,
    enemies: tuple[EnemyState, ...],
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> tuple[EnemyState, ...]:
    predicted, _took_damage = _simulate_enemy_turn(
        enemies=enemies,
        player_position=player_position,
        width=width,
        height=height,
        walls=walls,
    )
    return predicted


def _predict_enemy_substep(
    *,
    enemy_type: int,
    enemy_position: GridPosition,
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> GridPosition:
    can_pass_walls = enemy_type == _ENEMY_TYPE_GLITCH
    candidates: list[GridPosition] = []
    for dx, dy in _MOVE_VECTORS.values():
        candidate = GridPosition(x=enemy_position.x + dx, y=enemy_position.y + dy)
        if not _is_in_bounds(candidate, width=width, height=height):
            continue
        if not can_pass_walls and candidate in walls:
            continue
        candidates.append(candidate)
    candidates.append(enemy_position)

    if not candidates:
        return enemy_position

    distances = {candidate: _manhattan(candidate, player_position) for candidate in candidates}
    current_distance = _manhattan(enemy_position, player_position)
    best_distance = min(distances.values())
    best_candidates = [candidate for candidate in candidates if distances[candidate] == best_distance]

    if enemy_type == _ENEMY_TYPE_GLITCH:
        reducing_candidates = [
            candidate
            for candidate in candidates
            if distances[candidate] < current_distance
        ]
        if reducing_candidates:
            wall_reducing_candidates = [candidate for candidate in reducing_candidates if candidate in walls]
            if wall_reducing_candidates:
                return wall_reducing_candidates[0]
            best_reducing_distance = min(distances[candidate] for candidate in reducing_candidates)
            best_reducing_candidates = [
                candidate
                for candidate in reducing_candidates
                if distances[candidate] == best_reducing_distance
            ]
            return best_reducing_candidates[0]
        return best_candidates[0]

    if enemy_type == _ENEMY_TYPE_CRYPTOG:
        hidden_best_candidates = [
            candidate
            for candidate in best_candidates
            if not _has_line_of_sight(player=player_position, target=candidate, walls=walls)
        ]
        if hidden_best_candidates:
            return hidden_best_candidates[0]
        return best_candidates[0]

    return best_candidates[0]


def _next_player_positions(
    *,
    position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> tuple[GridPosition, ...]:
    candidates: list[GridPosition] = [position]
    for dx, dy in _MOVE_VECTORS.values():
        candidate = GridPosition(x=position.x + dx, y=position.y + dy)
        if not _is_in_bounds(candidate, width=width, height=height):
            continue
        if candidate in walls:
            continue
        candidates.append(candidate)
    return tuple(candidates)


def _movement_actions(action_space: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(action for action in action_space if action in _MOVE_VECTORS)


def _navigable_movement_actions(
    *,
    state: GameStateSnapshot,
    action_space: tuple[str, ...],
) -> tuple[str, ...]:
    movement_actions = _movement_actions(action_space)
    if state.map.status != "ok" or state.map.player_position is None:
        return movement_actions

    player = state.map.player_position
    walls = _wall_positions(state)
    navigable: list[str] = []
    for action in movement_actions:
        dx, dy = _MOVE_VECTORS[action]
        candidate = GridPosition(x=player.x + dx, y=player.y + dy)
        if not _is_in_bounds(candidate, width=state.map.width, height=state.map.height):
            continue
        if candidate in walls:
            continue
        navigable.append(action)
    return tuple(navigable)


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    wall_positions = {cell.position for cell in state.map.cells if cell.is_wall}
    if wall_positions:
        return wall_positions
    return {wall.position for wall in state.map.walls}


@dataclass(frozen=True)
class _RouteStep:
    action: str
    distance: int


def _cells_by_position(state: GameStateSnapshot) -> dict[GridPosition, MapCellState]:
    return {cell.position: cell for cell in state.map.cells}


def _resource_cluster_score(
    *,
    position: GridPosition,
    width: int,
    height: int,
    cells_by_position: dict[GridPosition, MapCellState],
) -> int:
    total = 0
    neighbors = (
        position,
        GridPosition(position.x + 1, position.y),
        GridPosition(position.x - 1, position.y),
        GridPosition(position.x, position.y + 1),
        GridPosition(position.x, position.y - 1),
    )
    for neighbor in neighbors:
        if not _is_in_bounds(neighbor, width=width, height=height):
            continue
        cell = cells_by_position.get(neighbor)
        if cell is None:
            continue
        total += max(int(cell.credits), 0) + max(int(cell.energy), 0)
    return total


def _iter_wall_candidates(state: GameStateSnapshot) -> tuple[_WallCandidate, ...]:
    walls: list[_WallCandidate] = []
    seen: set[GridPosition] = set()

    for wall in state.map.walls:
        walls.append(
            _WallCandidate(
                position=wall.position,
                prog_id=wall.prog_id,
                points=int(wall.points),
            )
        )
        seen.add(wall.position)

    for cell in state.map.cells:
        if not cell.is_wall or cell.position in seen:
            continue
        walls.append(
            _WallCandidate(
                position=cell.position,
                prog_id=cell.prog_id,
                points=int(cell.points),
            )
        )
    return tuple(walls)


def _best_adjacent_position(
    *,
    player: GridPosition,
    target_wall: GridPosition,
    state: GameStateSnapshot,
) -> tuple[GridPosition, int] | None:
    walls = _wall_positions(state)
    best_target: GridPosition | None = None
    best_distance: int | None = None
    movement_actions = tuple(_MOVE_VECTORS.keys())

    for dx, dy in _MOVE_VECTORS.values():
        adjacent = GridPosition(target_wall.x + dx, target_wall.y + dy)
        if not _is_in_bounds(adjacent, width=state.map.width, height=state.map.height):
            continue
        if adjacent in walls:
            continue

        if adjacent == player:
            distance = 0
        else:
            route = _shortest_path_first_action(
                start=player,
                target=adjacent,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
                allowed_first_actions=movement_actions,
            )
            if route is None:
                continue
            distance = route.distance

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_target = adjacent

    if best_target is None or best_distance is None:
        return None
    return (best_target, best_distance)


def _shortest_path_first_action(
    *,
    start: GridPosition,
    target: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
    allowed_first_actions: tuple[str, ...],
) -> _RouteStep | None:
    if start == target:
        return None
    if not allowed_first_actions:
        return None
    if not _is_in_bounds(target, width=width, height=height):
        return None
    if start in walls or target in walls:
        return None

    open_heap: list[tuple[int, int, int, GridPosition]] = []
    push_counter = 0
    start_h = _manhattan(start, target)
    heapq.heappush(open_heap, (start_h, 0, push_counter, start))
    best_cost: dict[GridPosition, int] = {start: 0}
    parent: dict[GridPosition, tuple[GridPosition, str]] = {}

    while open_heap:
        _, current_cost, _, current = heapq.heappop(open_heap)
        known_cost = best_cost.get(current)
        if known_cost is None or current_cost != known_cost:
            continue
        if current == target:
            break
        if current == start:
            actions_to_try = allowed_first_actions
        else:
            actions_to_try = tuple(_MOVE_VECTORS.keys())
        for action in actions_to_try:
            dx, dy = _MOVE_VECTORS[action]
            candidate = GridPosition(x=current.x + dx, y=current.y + dy)
            if not _is_in_bounds(candidate, width=width, height=height):
                continue
            if candidate in walls:
                continue
            tentative_cost = current_cost + 1
            candidate_best = best_cost.get(candidate)
            if candidate_best is not None and tentative_cost >= candidate_best:
                continue
            best_cost[candidate] = tentative_cost
            parent[candidate] = (current, action)
            push_counter += 1
            priority = tentative_cost + _manhattan(candidate, target)
            heapq.heappush(open_heap, (priority, tentative_cost, push_counter, candidate))

    if target not in parent:
        return None

    cursor = target
    first_action: str | None = None
    distance = 0
    while cursor != start:
        previous, action = parent[cursor]
        first_action = action
        distance += 1
        cursor = previous
    if first_action is None:
        return None
    return _RouteStep(action=first_action, distance=distance)


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _action_steps_to_position(
    *,
    action: str,
    start: GridPosition,
    target: GridPosition,
) -> bool:
    delta = _MOVE_VECTORS.get(action)
    if delta is None:
        return False
    return GridPosition(x=start.x + delta[0], y=start.y + delta[1]) == target


def _is_horizontal_line_clear(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player.y != target.y:
        return False
    start = min(player.x, target.x) + 1
    end = max(player.x, target.x)
    for x in range(start, end):
        if GridPosition(x=x, y=player.y) in walls:
            return False
    return True


def _is_vertical_line_clear(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player.x != target.x:
        return False
    start = min(player.y, target.y) + 1
    end = max(player.y, target.y)
    for y in range(start, end):
        if GridPosition(x=player.x, y=y) in walls:
            return False
    return True


def _has_line_of_sight(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player == target:
        return True
    if player.x == target.x:
        return _is_vertical_line_clear(player=player, target=target, walls=walls)
    if player.y == target.y:
        return _is_horizontal_line_clear(player=player, target=target, walls=walls)
    return False

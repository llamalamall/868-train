"""Shared state-delta helpers for Hybrid rewarding and rollout metrics."""

from __future__ import annotations

from src.state.schema import GameStateSnapshot

_FAIL_TERMINAL_REASON_TOKENS: tuple[str, ...] = ("fail", "loss", "dead", "death", "player_health")


def health_damage_taken(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_health = getattr(previous_state, "health", None)
    current_health = getattr(current_state, "health", None)
    previous_value = getattr(previous_health, "value", None)
    current_value = getattr(current_health, "value", None)
    if getattr(previous_health, "status", None) != "ok" or getattr(current_health, "status", None) != "ok":
        return 0.0
    try:
        previous_numeric = float(previous_value)
        current_numeric = float(current_value)
    except (TypeError, ValueError):
        return 0.0
    return max(previous_numeric - current_numeric, 0.0)


def enemy_hp_by_slot(state: GameStateSnapshot) -> dict[int, int] | None:
    if state.map.status != "ok":
        return None
    hp_by_slot: dict[int, int] = {}
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        hp_by_slot[int(enemy.slot)] = max(int(enemy.hp), 0)
    return hp_by_slot


def enemy_damage_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_hp = enemy_hp_by_slot(previous_state)
    current_hp = enemy_hp_by_slot(current_state)
    if previous_hp is None or current_hp is None:
        return 0.0
    total_damage = 0
    for enemy_id, before_hp in previous_hp.items():
        after_hp = current_hp.get(enemy_id)
        if after_hp is None:
            continue
        if before_hp > after_hp:
            total_damage += before_hp - after_hp
    return float(total_damage)


def enemy_presence_by_slot(state: GameStateSnapshot) -> dict[int, int] | None:
    if state.map.status != "ok":
        return None
    presence: dict[int, int] = {}
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        presence[int(enemy.slot)] = presence.get(int(enemy.slot), 0) + 1
    return presence


def enemy_cleared_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_presence = enemy_presence_by_slot(previous_state)
    current_presence = enemy_presence_by_slot(current_state)
    if previous_presence is None or current_presence is None:
        return 0.0
    cleared = 0
    for enemy_id, previous_count in previous_presence.items():
        current_count = current_presence.get(enemy_id, 0)
        if previous_count > current_count:
            cleared += previous_count - current_count
    return float(cleared)


def enemy_growth_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_presence = enemy_presence_by_slot(previous_state)
    current_presence = enemy_presence_by_slot(current_state)
    if previous_presence is None or current_presence is None:
        return 0.0
    growth = 0
    for enemy_id, current_count in current_presence.items():
        previous_count = previous_presence.get(enemy_id, 0)
        if current_count > previous_count:
            growth += current_count - previous_count
    return float(growth)


def reason_indicates_fail_terminal(reason: object) -> bool:
    if not isinstance(reason, str):
        return False
    normalized = reason.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _FAIL_TERMINAL_REASON_TOKENS)

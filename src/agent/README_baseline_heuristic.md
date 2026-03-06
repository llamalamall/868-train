# HeuristicBaselineAgent

## File
- `src/agent/baseline_heuristic.py`

## Purpose
- Provides a rule-based non-ML baseline policy that reacts to decoded state.
- Designed to outperform pure random in simple, observable situations.

## Config
- `HeuristicBaselineConfig.low_health_threshold` (default `3`)
- `HeuristicBaselineConfig.avoid_enemy_distance` (default `1`)

## Decision Order
1. Validate `action_space` is non-empty.
2. If health is known and `health <= low_health_threshold`, choose `wait` when available.
3. If map/player data is available:
   - Try `_select_escape_move(...)` first.
   - If no escape move is found, try `_select_goal_move(...)`.
4. Fallback priority:
   - `confirm` (if available)
   - `wait` (if available)
   - random choice from available actions (tie/fallback)

## Escape Rule
- Computes nearest-enemy Manhattan distance from player.
- If enemy is within `avoid_enemy_distance`, picks a move that increases nearest-enemy distance the most.
- Only considers movement actions present in the current `action_space`.

## Goal Rule
- Uses player and exit positions from `state.map`.
- Chooses a movement action that strictly reduces Manhattan distance to exit.
- If no move improves distance, no goal move is selected.

## Failure Behavior
- Raises `ValueError` when `action_space` is empty.
- Handles missing/invalid state fields defensively by falling back to safe defaults.

# HeuristicBaselineAgent

## File
- `src/agent/baseline_heuristic.py`

## Purpose
- Provides a rule-based non-ML baseline policy that reacts to decoded state.
- Designed to outperform pure random in simple, observable situations.

## Config
- `HeuristicBaselineConfig.low_health_threshold` (default `3`)
- `HeuristicBaselineConfig.verbose_action_logging` (default `False`)

## Decision Order
1. Validate `action_space` is non-empty.
2. If health is known and `health <= low_health_threshold`, choose `wait` when available.
3. If map/player data is available:
   - If an enemy is in direct line-of-sight (same row/column with no wall between), move toward that enemy.
   - Otherwise, if siphons remain, move toward the nearest siphon.
   - Otherwise, move toward the exit.
4. Fallback priority:
   - `confirm` (if available)
   - `wait` (if available)
   - random choice from available actions (tie/fallback)

## Coordinate Convention
- Uses game coordinates where `(0,0)` is bottom-left.
- `move_up` means `y + 1`.
- `move_down` means `y - 1`.
- `move_right` means `x + 1`.
- `move_left` means `x - 1`.

## Enemy LOS Rule
- Scans enemies in the same row or column as the player.
- Requires a clear path (no wall cells between player and enemy).
- Chooses a move that reduces distance to the closest visible enemy.

## Siphon Rule
- If `map.siphons` contains entries, targets the nearest siphon first.
- This causes siphons to be collected before routing to the exit when possible.

## Goal Rule
- Uses player and exit positions from `state.map` once no siphon target remains.
- Chooses a movement action that strictly reduces Manhattan distance to exit.

## Failure Behavior
- Raises `ValueError` when `action_space` is empty.
- Handles missing/invalid state fields defensively by falling back to safe defaults.

## Verbose Action Logs
- When `verbose_action_logging=True`, each `select_action(...)` call logs the chosen action and decision reason.
- Log format includes: chosen action, reason, health, player position, exit position, and available actions.

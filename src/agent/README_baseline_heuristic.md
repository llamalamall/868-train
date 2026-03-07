# HeuristicBaselineAgent

## File
- `src/agent/baseline_heuristic.py`

## Purpose
- Provides a rule-based non-ML baseline policy that reacts to decoded state.
- Designed to outperform pure random in simple, observable situations.

## Config
- `HeuristicBaselineConfig.low_health_threshold` (default `3`)
- `HeuristicBaselineConfig.verbose_action_logging` (default `False`)
- `HeuristicBaselineConfig.resource_goal_weight` (default `0.60`)
- `HeuristicBaselineConfig.prog_goal_weight` (default `0.30`)
- `HeuristicBaselineConfig.points_goal_weight` (default `0.10`)
- `HeuristicBaselineConfig.enable_prog_usage` (default `True`)
- `HeuristicBaselineConfig.prog_energy_floor` (default `4`)
- `HeuristicBaselineConfig.prog_retry_backoff_steps` (default `4`)
- `HeuristicBaselineConfig.show_recast_gap_steps` (default `6`)

## Decision Order
1. Validate `action_space` is non-empty.
2. Evaluate guarded prog-slot actions (`prog_slot_1..prog_slot_10`) when inventory data is available:
   - emergency-first `.delay` / `.anti-v` under pressure,
   - periodic `.show` / `.debug` recon,
   - `.step` only when route to objective appears blocked.
3. Apply short per-slot backoff after ineffective prog attempts (for example, no energy spend observed).
4. If health is known and `health <= low_health_threshold`, choose `wait` when available.
5. If siphon count decreases (a siphon was collected), build a temporary harvest plan by weighted random choice:
   - `resources` (favored),
   - `progs`,
   - `points`.
6. If a harvest plan is active, move to its target and press `space` at target.
7. If map/player data is available:
   - If an enemy is in direct line-of-sight (same row/column with no wall between), move toward that enemy.
   - Otherwise, if siphons remain, move toward the nearest siphon.
   - Otherwise, move toward the exit.
8. Fallback priority:
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
- Chooses the first step on a shortest wall-aware path toward the closest visible enemy.

## Siphon Rule
- If `map.siphons` contains entries, targets the nearest siphon first.
- This causes siphons to be collected before routing to the exit when possible.
- Uses shortest-path search (BFS) over the map grid so detours around walls are handled.
- While siphons remain, the heuristic filters out any immediate move that would step onto the exit tile.

## Post-Siphon Harvest Rule
- Trigger: remaining siphon count decreases between consecutive states.
- Weighted random category selection favors:
  - resources (`credits + energy` cluster score),
  - then progs (favoring `.debug`, `.push`, `.anti-v`, `.d_bom`, `.step`),
  - then points (highest-point wall).
- Harvest behavior:
  - Resources: move to best resource cluster cell and press `space`.
  - Progs/Points: move to a cell adjacent to target wall and press `space`.

## Goal Rule
- Uses player and exit positions from `state.map` once no siphon target remains.
- Uses shortest-path search (BFS) toward the exit so wall detours are handled.

## Failure Behavior
- Raises `ValueError` when `action_space` is empty.
- Handles missing/invalid state fields defensively by falling back to safe defaults.

## Verbose Action Logs
- When `verbose_action_logging=True`, each `select_action(...)` call logs the chosen action and decision reason.
- Log format includes: chosen action, reason, health, player position, exit position, and available actions.

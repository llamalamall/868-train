# ReHeuristicAgent

## File
- `src/agent/re_heuristic.py`

## Purpose
- Provides a rule-registry wrapper around `HeuristicBaselineAgent`.
- Starts with baseline-equivalent seeded rules (`iteration 0`) and can load mined rule packs.
- Exposes stable `rule_id` metadata for loop acceptance/auditing.

## Config
- `ReHeuristicConfig.low_health_threshold` (default `3`)
- `ReHeuristicConfig.enemy_prediction_horizon_steps` (default `2`)
- `ReHeuristicConfig.verbose_action_logging` (default `False`)
- `ReHeuristicConfig.enable_mined_rules` (default `True`)
- `ReHeuristicConfig.rule_pack_path` (default `None`)

## Decision Model
1. Get baseline action/reason from `HeuristicBaselineAgent`.
2. Evaluate compiled rules in descending `priority`.
3. First matching rule returns action; falls back to baseline action when no rule matches.
4. Record:
   - `last_decision_reason`
   - `last_decision_rule_id`

## Rule Registry
- Rule fields:
  - `rule_id`
  - `priority`
  - `reason`
  - `predicate`
  - `action_selector`
  - `evidence_refs`
- Baseline seed rule count is available via `ReHeuristicAgent.baseline_rule_count()`.
- External rule packs are loaded from JSON (`rules` array of `RuleDefinition` objects).

## Loop Driver
- Loop implementation: `src/agent/re_heuristic_loop.py`
- Script wrapper: `artifacts/run_re_heuristic_loop.py`
- Pipeline:
  - mine candidates from `artifacts/readable`,
  - stage rules,
  - generate deterministic scenarios,
  - offline KPI gate,
  - live validation gate,
  - accept/reject and persist iteration artifacts.

## Failure Behavior
- Raises `ValueError` when `action_space` is empty.
- Missing/invalid rule packs are ignored with warning and baseline behavior is retained.
- Live-validation failures in loop mode produce explicit reject reasons and count as no-meaningful iterations.


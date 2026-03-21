# HYBRID_TASKS.md

Current architecture direction is Hybrid-first.

## Supported Runtime Surfaces

- `run-hybrid movement-test`
- `run-hybrid train-meta-no-enemies`
- `run-hybrid train-full-hierarchical`
- `run-hybrid eval-hybrid`
- `hybrid-gui`
- `run-random` and `run-heuristic` as smoke-test utilities

## Architectural Notes

- The standalone DQN runner, evaluator, and agent have been removed.
- `MetaControllerDQN` and `ThreatControllerDRQN` remain because they are part of the Hybrid algorithm itself.
- Shared runner concerns now belong in `src/env/runner_common.py`.
- Hybrid runner responsibilities are split across:
  - `src/hybrid/cli.py`
  - `src/hybrid/rollout.py`
  - `src/hybrid/session.py`
- Coordinator support code is split across:
  - `src/hybrid/objective_planning.py`
  - `src/hybrid/feature_encoding.py`
  - `src/hybrid/coordinator.py`

## Cleanup Rules

- Do not reintroduce standalone DQN entrypoints.
- Keep GUI naming and docs Hybrid-first.
- If Hybrid workflow flags change, update the CLI, GUI, tests, and README in the same change.

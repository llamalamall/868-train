# HYBRID_TASKS.md — Hybrid Agent Rollout (A* + Meta DQN + Threat DRQN)

## Summary
- Build a brand-new hybrid stack (new agent, new environment wrapper, new rewards) while leaving the existing DQN stack unchanged.
- Reuse the existing GUI by adding Hybrid workflow tabs.
- Roll out in three gates: movement-only no-enemy validation, meta-controller no-enemy training, full hierarchical enemy-enabled training.

## Implementation Changes
1. `Foundation`: Added `src/hybrid` subsystem with:
   - `HybridLiveEnv`
   - `AStarMovementController`
   - `MetaControllerDQN`
   - `ThreatControllerDRQN`
   - `HybridCoordinator`
   - `HybridRewardSuite`
   - `HybridCheckpointManager`
2. `Objective Model`: Explicit objective phases:
   - `collect_siphons`
   - `collect_resources_progs_points`
   - `exit_sector`
3. `Threat Control`: Threat override actions:
   - `route_default`
   - `evade`
   - `engage`
   - `wait`
   - `use_prog`
4. `Phase 1 (movement-test, no enemies)`: `run-hybrid movement-test`
5. `Phase 2 (train-meta-no-enemies)`: `run-hybrid train-meta-no-enemies`
6. `Phase 3 (train-full-hierarchical, enemies on)`: `run-hybrid train-full-hierarchical`
7. `CLI Surface`: Added `train-868 run-hybrid <subcommand>`.
8. `GUI Reuse`: Extended `src/gui/dqn_runner_gui.py` with Hybrid tabs and monitor integration.
9. `CLI Registration`: Registered `run-hybrid` in `src/cli.py`.
10. `Artifacts/Checkpoints`: Hybrid bundles now save under `artifacts/hybrid/<run-id>/` with:
    - `meta_controller.pt`
    - `threat_drqn.pt`
    - `hybrid_config.json`
    - `training_state.json`

## Public Interfaces and Types
- New CLI interface: `train-868 run-hybrid <subcommand> [flags]`.
- New env protocol: `HybridEpisodeEnv`.
- New decision/type contracts:
  - `ObjectivePhase`
  - `MetaObjectiveChoice`
  - `ThreatOverride`
  - `HybridDecision`
- New checkpoint schema version independent of legacy DQN checkpoints.

## Gate Workflow
1. **Gate A**: `movement-test` metrics (route length, replans, invalid actions, premature exits).
2. **Gate B**: `train-meta-no-enemies` objective progression efficiency uplift.
3. **Gate C**: `train-full-hierarchical` survival/map-clear uplift vs Gate B baseline.


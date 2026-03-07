# DQN Agent

## File
- `src/agent/dqn_agent.py`

## Implemented Scope
- Objective-aware compact state featurizer for `GameStateSnapshot` (`state_to_feature_vector`).
- Linear DQN-style Q approximator with:
  - replay buffer sampling,
  - target-network sync,
  - epsilon-greedy action selection with linear decay.
- Checkpoint save/load:
  - model parameters (online + target),
  - replay buffer contents,
  - training counters (`total_env_steps`, `optimization_steps`, `episodes_seen`, `last_loss`),
  - arbitrary metadata payload.

## Main Types
- `DQNConfig`: hyperparameters for replay, update cadence, epsilon schedule, and feature controls.
- `DQNAgent`: action selection, transition ingestion (`observe`), learning updates, and checkpoint I/O.
- `DQNUpdateResult`: per-step update status (`did_update`, `loss`, replay size, epsilon).

## Training Integration
- Learning rollout loop is in `src/training/train.py` as `run_dqn_training(...)`.
- The loop:
  - resets episodes via `EpisodeEnv`,
  - selects valid actions using `DQNAgent`,
  - records transitions and runs replay updates,
  - returns per-episode summaries including update count, terminal reason, and epsilon.

## Notes
- This first version intentionally keeps the function approximator simple for stability and deterministic tests.
- Default feature vector length is `22` and includes objective-phase and action-availability signals.
- The checkpoint format is JSON and versioned (`CHECKPOINT_VERSION`), with validation on load.

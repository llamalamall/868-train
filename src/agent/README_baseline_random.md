# RandomBaselineAgent

## File
- `src/agent/baseline_random.py`

## Purpose
- Provides a non-ML baseline policy that samples actions uniformly at random.
- Useful as a floor metric for comparing heuristic and learning-based agents.

## How It Works
- Exposes `RandomBaselineAgent.select_action(...)`.
- Accepts the current `state`, `action_space`, and a caller-provided `random.Random`.
- If `preferred_actions` is not set, samples from the full `action_space`.
- If `preferred_actions` is set, filters to actions that exist in the current `action_space` and samples from that subset.

## Determinism
- The agent is deterministic when the caller supplies a seeded `random.Random`.
- No internal RNG state is stored in the agent itself.

## Failure Behavior
- Raises `ValueError` when `action_space` is empty.
- Raises `ValueError` when `preferred_actions` is set but none are valid for the current `action_space`.

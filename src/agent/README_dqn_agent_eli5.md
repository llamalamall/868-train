# DQN Agent (Explain It Like I'm 5)

## What is this?
Think of the agent like a kid learning a game:
- It tries buttons.
- It gets rewards or penalties.
- Over time, it learns which buttons are better in each situation.

The settings below control **how fast it learns**, **how random it is**, and **how stable training is**.

## Most Important Parameters

### `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`
This is the "curious mode" control.
- `epsilon_start`: how random the agent is at the beginning.
- `epsilon_end`: how random it stays after training for a while.
- `epsilon_decay_steps`: how quickly it goes from very random to less random.

ELI5:
- High epsilon = "try random stuff."
- Low epsilon = "use what I already learned."

If training feels too random for too long, lower `epsilon_decay_steps` or lower `epsilon_start`.

Current defaults:
- `epsilon_start = 0.8`
- `epsilon_end = 0.05`
- `epsilon_decay_steps = 5000`

### `learning_rate`
How big each learning update is.

ELI5:
- Too high: the kid overreacts and forgets old lessons.
- Too low: the kid learns very slowly.

Current default:
- `learning_rate = 0.005`

### `gamma`
How much the agent cares about future rewards.

ELI5:
- Low gamma: "I only care about candy right now."
- High gamma: "I care about finishing the whole level."

Current default:
- `gamma = 0.99`

### `batch_size`
How many old experiences are used for one learning step.

ELI5:
- Bigger batch = steadier learning, but slower/heavier.
- Smaller batch = noisier learning, but lighter/faster.

Current default:
- `batch_size = 64`

### `min_replay_size`
How many experiences must be collected before learning starts.

ELI5:
- If this is too small, the kid learns from too little data.
- If too big, learning starts late.

Current default:
- `min_replay_size = 256`

### `replay_capacity`
How many past experiences the memory can hold.

ELI5:
- Bigger memory = remembers more old lessons.
- Smaller memory = forgets old lessons sooner.

Current default:
- `replay_capacity = 20000`

### `target_sync_interval`
How often the stable "teacher copy" is updated.

ELI5:
- Updating too often can make learning wobble.
- Updating less often usually makes learning steadier.

Current default:
- `target_sync_interval = 500`

## Quick Tuning Guide

If the agent is random for too long:
- Reduce `epsilon_decay_steps` (for example, 5000 -> 3000).
- Or reduce `epsilon_start`.

If learning is unstable (reward swings wildly):
- Lower `learning_rate`.
- Increase `batch_size` a bit.
- Increase `target_sync_interval`.

If learning starts too late:
- Lower `min_replay_size`.

If the agent overfits recent behavior:
- Increase `replay_capacity`.

## Where these live in code
- Main config: `src/agent/dqn_agent.py` (`DQNConfig`)
- CLI defaults and overrides: `src/env/dqn_policy_runner.py`

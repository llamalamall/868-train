# 868-train

Python scaffold for automating and training an agent to play 868-hack.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -e .[dev]
```

3. Run the app:

```powershell
python -m src.app
```

## Development Commands

```powershell
ruff check .
mypy src
pytest -q
```

## Master CLI

Use the unified CLI to run project tools:

```powershell
python -m src.cli --help
```

Or via console script (after `pip install -e .[dev]`):

```powershell
train-868 --help
```

Available subcommands:
- `bootstrap`: run startup checks (`src.app`)
- `fingerprint`: binary hash helper (`src.config.fingerprint`)
- `offset-smoke`: live offsets smoke test (`src.memory.offset_smoke_test`)
- `state-monitor`: interactive memory monitor (`src.memory.state_monitor_tui`)
- `run-random`: random baseline runner (`src.env.random_policy_runner`)
- `run-heuristic`: heuristic baseline runner (`src.env.heuristic_policy_runner`)
- `run-dqn`: DQN train/eval runner with checkpoint save/load (`src.env.dqn_policy_runner`)

For command-specific options, forward `--help` to the underlying tool:

```powershell
python -m src.cli run-heuristic -- --help
```

Examples:

```powershell
# Startup validation checks
python -m src.cli bootstrap

# Print binary SHA256
python -m src.cli fingerprint --print-sha256 "C:\path\to\868-hack.exe"

# Run random baseline episodes
python -m src.cli run-random --episodes 5 --max-steps 200

# Run heuristic baseline episodes
python -m src.cli run-heuristic --episodes 5 --max-steps 200 --movement-keys wasd

# Run heuristic with deeper enemy lookahead
python -m src.cli run-heuristic --episodes 5 --enemy-prediction-horizon-steps 4

# Step through actions manually (press Enter in TUI before each move)
python -m src.cli run-dqn --episodes 1 --step-through

# Train DQN and auto-save checkpoint (path auto-generated under artifacts/checkpoints)
python -m src.cli run-dqn --episodes 20 --max-steps 200

# Evaluate from a saved DQN checkpoint
python -m src.cli run-dqn --mode eval --checkpoint artifacts/checkpoints/dqn-latest.json --episodes 5
```

When `--step-through` is enabled, runners automatically use window-targeted input so actions still dispatch to the game while the TUI has focus.

## Control Smoke Test

Task 04 adds a deterministic action API in `src/controller/action_api.py` and low-level key
input retries in `src/controller/input_driver.py`.

- Action methods: `move_up`, `move_down`, `move_left`, `move_right`, `confirm`, `cancel`, `wait`
- Fixed smoke sequence helper: `run_smoke_test_sequence(action_api)`
- Tunables: `ActionConfig` + `ActionTimings` (key map, press duration, inter-action delay, retries)

## Binary Fingerprinting

The app validates the game binary hash at startup using `src/config/binary_fingerprint.json`.

1. Set `enabled` to `true`.
2. Set `binary_path` to your pinned executable.
3. Compute SHA256:

```powershell
python -m src.config.fingerprint --print-sha256 "C:\path\to\868-hack.exe"
```

4. Paste the value into `expected_sha256`.

The app exits on hash mismatch with an actionable error. To intentionally bypass the check,
set `enabled` to `false` explicitly.

## Offsets Registry

Offsets discovered from Ghidra are tracked in `src/config/offsets.json`.
Startup validates schema and fails fast when required fields are missing or malformed.

Each entry requires:
- `name`
- `data_type`
- `base` (`kind` + `value`)
- `pointer_chain`
- `confidence` (`low`, `medium`, `high`)
- `notes`

Use `notes/ghidra_findings.md` as the worksheet before updating offsets.

## Memory Monitor (Interactive TUI)

Use the interactive monitor to watch all configured offset fields live:

```powershell
python -m src.memory.state_monitor_tui
```

Common options:

```powershell
# Monitor selected fields only
python -m src.memory.state_monitor_tui --fields player_energy,player_credits

# Faster polling + re-resolve pointer chains every tick
python -m src.memory.state_monitor_tui --interval 0.25 --resolve-each-poll
```

TUI controls:
- `q`: quit
- `z`: pause/resume polling
- `r`: refresh immediately
- `p`: toggle pointer mode (`cached-addresses` vs `resolve-each-poll`)
- `enter`: advance one pending runner step when `--step-through` is enabled
- `up/down/left/right`, `1` through `0`, `escape`, `space`: pass controls to the game window

## Telemetry Logging (Task 10)

Structured telemetry writes JSONL event streams under a namespaced run directory in `logs/`.

- Core writer: `src/telemetry/logger.py` (`JsonlTelemetryLogger`)
- Replay + summaries: `src/telemetry/metrics.py`

Typical events:
- `episode_start`
- `step` (action, pre/post state, reward, done, episode_id, step_index, timestamp)
- `terminal`

Example usage:

```python
from src.telemetry.logger import JsonlTelemetryLogger, TelemetryLoggerConfig

logger = JsonlTelemetryLogger(TelemetryLoggerConfig(run_name="train"))
episode_id = logger.start_episode()
logger.log_step(
    episode_id=episode_id,
    action="move_up",
    pre_state={"health": 10},
    post_state={"health": 9},
    reward=-1.0,
    done=False,
)
logger.log_terminal(episode_id=episode_id, reason="fail_state")
logger.close()
```

## Baseline Runners (Task 13)

Two baseline policy runners are available:
- Random baseline: `python -m src.cli run-random ...`
- Heuristic baseline: `python -m src.cli run-heuristic ...`

Both support reward shaping flags (`--reward-*`) and print per-episode + summary metrics for quick comparison.
All policy runners (`run-random`, `run-heuristic`, `run-dqn`) now launch the live state-monitor TUI by
default in a separate window; disable with `--no-tui` and tune polling with `--tui-interval`.

## Project Structure

- `src/controller`: input/window control layer
- `src/memory`: process attach + memory reading layer
- `src/state`: normalized game-state extraction
- `src/env`: environment wrapper (`reset`, `step`)
- `src/agent`: baseline and learning agents
- `src/training`: training/evaluation/reward pipeline
- `src/telemetry`: logs and metrics
- `src/config`: static config schemas and loaders
- `logs/`: runtime telemetry output
- `notes/`: reverse-engineering notes

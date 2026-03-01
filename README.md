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

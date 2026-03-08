# 868-train

Python scaffold for automating and training an agent to play 868-hack.

## Runbook

This runbook is the Task 18 operational workflow for repeated experiments.
It is organized by the exact phases you execute in practice:

1. Setup
2. Run (bootstrap + smoke checks)
3. Train
4. Evaluate
5. Troubleshoot

## Setup

### 1) Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

### 2) Confirm CLI surface

```powershell
python -m src.cli --help
```

### 3) Optional dev checks

```powershell
ruff check .
mypy src
pytest -q
```

## Binary Fingerprinting

Bootstrap validates the configured game binary hash before runtime starts.

1. Open `src/config/binary_fingerprint.json`.
2. Set:
- `enabled: true`
- `binary_path: "<absolute path to your 868 executable>"`
3. Compute SHA256:

```powershell
python -m src.cli fingerprint -- --print-sha256 "C:\path\to\868-HACK.exe"
```

4. Paste that hash into `expected_sha256`.

If the game binary changes intentionally, recalculate and update `expected_sha256`.

## Offsets Workflow (Ghidra -> Runtime)

Use `notes/ghidra_findings.md` as the working worksheet for discovery and validation.

1. Document candidate fields and pointer chains in `notes/ghidra_findings.md`.
2. Update `src/config/offsets.json`.
3. Validate config schema via bootstrap:

```powershell
python -m src.cli bootstrap
```

4. Validate live reads with the smoke test:

```powershell
python -m src.cli offset-smoke -- --fields player_health,player_energy,player_currency --iterations 20 --interval 0.5
```

5. Observe ongoing state stability in the monitor:

```powershell
python -m src.cli state-monitor -- --interval 0.5
```

## Run (Bootstrap + Smoke)

Run this sequence before each training session.

### 1) Startup validation

```powershell
python -m src.cli bootstrap
```

### 2) Offset read smoke

```powershell
python -m src.cli offset-smoke -- --iterations 20 --interval 0.5
```

### 3) Policy smoke with short episode budgets

```powershell
python -m src.cli run-random --episodes 2 --max-steps 50
python -m src.cli run-heuristic --episodes 2 --max-steps 50
```

### 4) Optional manual action verification

```powershell
python -m src.cli run-dqn --episodes 1 --max-steps 20 --step-through
```

## Train

`run-dqn` supports both train and eval mode.

### Quick training run (auto checkpoint path)

```powershell
python -m src.cli run-dqn --mode train --episodes 20 --max-steps 200
```

When `--checkpoint` is omitted in train mode, checkpoints are written under:
- `artifacts/checkpoints/dqn-YYYYMMDD-HHMMSS.json`
- periodic files (if enabled): `artifacts/checkpoints/dqn-YYYYMMDD-HHMMSS.ep00001.json`

### Recommended reproducible training run (explicit path + seed)

```powershell
python -m src.cli run-dqn --mode train --episodes 50 --max-steps 300 --seed 42 --checkpoint artifacts/checkpoints/dqn-20260308-01-baseline.json --checkpoint-every 10
```

### Useful train flags

- `--no-tui` disables live monitor window.
- `--movement-keys wasd` changes key mapping.
- `--reward-*` flags tune reward shaping.
- `--no-launch-exe` prevents auto-launch and requires game already running.

## Evaluate

### Evaluate one checkpoint

```powershell
python -m src.cli evaluate -- run --checkpoint artifacts/checkpoints/dqn-20260308-01-baseline.json --episodes 10 --max-steps 200 --seed 42 --json-out artifacts/eval/20260308-01/run-baseline.json
```

### Compare two checkpoints under identical settings

```powershell
python -m src.cli evaluate -- compare --checkpoint-a artifacts/checkpoints/dqn-20260308-01-baseline.json --checkpoint-b artifacts/checkpoints/dqn-20260308-02-candidate.json --episodes 10 --max-steps 200 --seed 42 --json-out artifacts/eval/20260308-02/compare-baseline-vs-candidate.json
```

`evaluate compare` launches the monitor TUI by default; use `--no-tui` if you do not want it.

## End-to-End First Run

Use this exact sequence to go from fresh setup to a scored checkpoint.

```powershell
pip install -e .[dev]
python -m src.cli fingerprint -- --print-sha256 "C:\path\to\868-HACK.exe"
# update src/config/binary_fingerprint.json
python -m src.cli bootstrap
python -m src.cli offset-smoke -- --iterations 20 --interval 0.5
python -m src.cli run-dqn --mode train --episodes 20 --max-steps 200 --seed 42 --checkpoint artifacts/checkpoints/dqn-20260308-01-baseline.json
python -m src.cli evaluate -- run --checkpoint artifacts/checkpoints/dqn-20260308-01-baseline.json --episodes 10 --max-steps 200 --seed 42 --json-out artifacts/eval/20260308-01/run-baseline.json
```

## Troubleshoot

Use this triage flow when a run fails.

1. `bootstrap` fails immediately:
- Run: `python -m src.cli bootstrap`
- If message contains `Executable fingerprint mismatch`: recompute hash and update `src/config/binary_fingerprint.json`.
- If message contains offsets schema errors: fix `src/config/offsets.json` (missing keys, bad types, invalid confidence/base values).

2. Process attach fails (`Process not found`, attach retries exhausted):
- Confirm executable name/path: `--exe 868-HACK.exe` (or explicit path).
- If game should not auto-launch, use `--no-launch-exe` only when the process is already running.
- Re-run a narrow smoke command first: `python -m src.cli offset-smoke -- --iterations 5`.

3. Window attach/focus/input issues:
- Try window-targeted input explicitly: add `--window-input`.
- If focus steals or fails repeatedly, run with `--no-focus-window --window-input`.
- Validate controls with short runs: `python -m src.cli run-random --episodes 1 --max-steps 20`.

4. Memory read instability (`null_pointer`, `read_failed`, `short_read`, desync):
- Run `state-monitor` and `offset-smoke` to identify which fields are unstable.
- Enable per-poll pointer resolution while debugging:
  `python -m src.cli state-monitor -- --resolve-each-poll`
  `python -m src.cli offset-smoke -- --resolve-each-loop`
- Revisit pointer chain entries in `notes/ghidra_findings.md` and `src/config/offsets.json`.

5. Step/reset timeout failures:
- Increase watchdogs: `--step-timeout`, `--reset-timeout`.
- Keep retries enabled (default) and verify reset sequence (`--reset-sequence confirm` or empty).
- Run a short, deterministic check:
  `python -m src.cli run-dqn --episodes 2 --max-steps 50 --seed 42`.

6. Evaluation fails due to checkpoint/action mismatch:
- Ensure checkpoint exists and is from this project.
- Run with `python -m src.cli run-dqn --mode eval --checkpoint <path> --episodes 1` first.
- Then run full KPI evaluation.

## Experiment Checklist

Use this checklist for every experiment iteration.

### Pre-run

- [ ] Fingerprint config points to intended binary and hash.
- [ ] Offsets in `src/config/offsets.json` validated with `bootstrap`.
- [ ] Offset smoke/state monitor show stable key fields.
- [ ] Planned hyperparameters and reward weights recorded.

### During run

- [ ] Train command saved in notes or shell history.
- [ ] Seed, episodes, max steps, and reward flags captured.
- [ ] Checkpoint save path confirmed.
- [ ] Any runtime anomalies noted with timestamps.

### Post-run

- [ ] Evaluate single checkpoint with fixed seed.
- [ ] Compare against previous baseline checkpoint.
- [ ] Save JSON outputs for both run and compare.
- [ ] Decide: promote, reject, or schedule follow-up experiment.

## Artifact Naming and Versioning Conventions

Use stable, sortable names to keep experiment history auditable.

### Run ID format

- `YYYYMMDD-NN-<tag>`
- Example: `20260308-01-baseline`, `20260308-02-reward-tune`

### Checkpoints

- `artifacts/checkpoints/dqn-<run-id>.json` (final)
- `artifacts/checkpoints/dqn-<run-id>.ep00010.json` (periodic)

### Evaluation outputs

- Single checkpoint KPI: `artifacts/eval/<run-id>/run-<label>.json`
- Comparison KPI: `artifacts/eval/<run-id>/compare-<label-a>-vs-<label-b>.json`

### Notes and logs

- Keep experiment notes under `notes/` keyed by run ID.
- Keep telemetry under `logs/` and preserve the run directory per experiment.

## Master CLI Reference

Available subcommands:
- `bootstrap`: startup checks (`src.app`)
- `fingerprint`: binary hash helper (`src.config.fingerprint`)
- `offset-smoke`: live offsets smoke test (`src.memory.offset_smoke_test`)
- `state-monitor`: interactive memory monitor (`src.memory.state_monitor_tui`)
- `run-random`: random baseline runner (`src.env.random_policy_runner`)
- `run-heuristic`: heuristic baseline runner (`src.env.heuristic_policy_runner`)
- `run-dqn`: DQN train/eval runner with checkpoint save/load (`src.env.dqn_policy_runner`)
- `dqn-gui`: GUI launcher for DQN run/eval/compare settings (`src.gui.dqn_runner_gui`)
- `evaluate`: fixed-seed DQN KPI harness and checkpoint comparison (`src.training.evaluate`)

To forward native command help through the master CLI:

```powershell
python -m src.cli run-dqn -- --help
python -m src.cli evaluate -- compare --help
```

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

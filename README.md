# 868-train

Hybrid-first training and tooling scaffold for automating 868-hack.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Optional checks:

```powershell
ruff check src tests
pytest -q
```

## Main Workflows

Bootstrap and monitor tools:

```powershell
python -m src.cli bootstrap
python -m src.cli fingerprint -- --print-sha256 "C:\path\to\868-HACK.exe"
python -m src.cli offset-smoke -- --iterations 20 --interval 0.5
python -m src.cli victory-monitor -- --pid <pid>
```

Baseline smoke runners:

```powershell
python -m src.cli run-random --episodes 2 --max-steps 50
python -m src.cli run-heuristic --episodes 2 --max-steps 50
```

Hybrid workflows:

```powershell
python -m src.cli run-hybrid movement-test --episodes 5 --max-steps 250
python -m src.cli run-hybrid train-meta-no-enemies --episodes 120 --max-steps 350
python -m src.cli run-hybrid train-full-hierarchical --warmstart-checkpoint artifacts/hybrid/hybrid-meta-best --episodes 200 --max-steps 450
python -m src.cli run-hybrid eval-hybrid --checkpoint artifacts/hybrid/full/<gate-c-run> --episodes 20
```

Hybrid GUI:

```powershell
python -m src.cli hybrid-gui
```

## Hybrid Checkpoints

Hybrid training runs now split by default:

- meta runs: `artifacts/hybrid/meta/<run-id>/`
- full runs: `artifacts/hybrid/full/<run-id>/`
- best meta pointer: `artifacts/hybrid/hybrid-meta-best`

Each bundle contains:

- `meta_controller.pt`
- `threat_drqn.pt`
- `hybrid_config.json`
- `training_state.json`
- optional victory-monitor outputs when enabled

`train-full-hierarchical` requires either `--resume-checkpoint` or `--warmstart-checkpoint`.

## Master CLI

Available commands:

- `bootstrap`
- `fingerprint`
- `offset-smoke`
- `victory-monitor`
- `run-random`
- `run-heuristic`
- `run-hybrid`
- `hybrid-gui`

To forward native command help through the master CLI:

```powershell
python -m src.cli run-hybrid -- --help
```

## Troubleshooting

If bootstrap fails:

- recheck `src/config/binary_fingerprint.json`
- revalidate `src/config/offsets.json`

If attach or input fails:

- confirm `--exe`
- try `--window-input`
- run a short baseline smoke command first

If Hybrid training fails:

- re-run `movement-test`
- confirm warmstart/resume bundle completeness
- inspect the run directory and victory-monitor log output when enabled

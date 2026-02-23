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

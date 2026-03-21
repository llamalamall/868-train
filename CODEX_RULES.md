# Codex Development Rules for 868-Train

## 1) Primary Objective

- Build and maintain the project as a Hybrid-first training stack.
- Prioritize reliability, observability, and maintainability over feature breadth.
- Treat one pinned binary version as the only supported runtime target.

## 2) Architecture Boundaries

- `src/controller`: input and window control only.
- `src/memory`: process attach, read, pointer resolution, and debugger-style monitoring only.
- `src/state`: decode and normalize memory into typed snapshots.
- `src/env`: runtime environment wrappers and shared runner helpers.
- `src/hybrid`: Hybrid decision-making, rollout flow, checkpointing, and reward logic.
- `src/training`: baseline rollout helpers and reward shaping only.
- `src/telemetry`: structured episode and step diagnostics.

## 3) Hybrid-First Surface Rules

- Do not add standalone DQN entrypoints, checkpoint formats, or docs back into the repo.
- Legacy algorithm names are only allowed when they describe the Hybrid internals themselves, such as `MetaControllerDQN` and `ThreatControllerDRQN`.
- Public runnable surfaces must stay aligned with the current Hybrid-first CLI:
  - `run-hybrid`
  - `hybrid-gui`
  - `run-random`
  - `run-heuristic`
  - monitoring/bootstrap tools

## 4) Shared Helper Rules

- Duplicated runner helpers must live in one shared module.
- Do not copy save-restore helpers, action-map builders, game-tick argument validators, or monitor-formatting helpers across runners.
- Prefer pure helper modules for reusable Hybrid calculations such as objective planning, feature encoding, and state deltas.

## 5) Change Coupling Rules

- Any CLI surface change must update `src/cli.py`, `README.md`, GUI exposure, and tests in the same change.
- Any Hybrid workflow flag change must update parser tests and GUI argument exposure in the same change.
- Any checkpoint schema change must update load/save tests in the same change.

## 6) Size and Refactor Thresholds

- Do not add new behavior to runner or GUI files already above roughly 600 lines without extracting helpers first.
- Do not add new behavior to classes already above roughly 300 lines without extracting helpers first.
- If the same pure helper block appears in more than one module, extract it before extending it.

## 7) Testing Requirements

- Every non-trivial new function needs tests.
- Use deterministic fake environments and fake backends where possible.
- Cover success and failure paths.
- Keep tests independent of the live game process unless the command is explicitly a manual runtime tool.

## 8) Documentation Hygiene

- Runtime docs must describe only supported surfaces.
- Temporary rollout notes must be updated or retired when architecture direction changes.
- Do not leave stale file names, commands, or examples in repo docs after a cleanup.

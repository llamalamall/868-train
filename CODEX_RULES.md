# Codex Development Rules for 868-Train

## 1) Primary Objective
- Build the bot in the order defined by [TASKS.md](/c:/Users/John White/868-train/TASKS.md).
- Prioritize reliability and observability over feature breadth.
- Treat one pinned binary version as the only supported runtime target.

## 2) Task-Order Discipline
- Do not skip ahead across milestone gates.
- Complete tasks in sequence unless a dependency issue forces a small prerequisite change.
- For each task: implement, run local checks, verify acceptance criteria, then move on.
- Use task-scoped commits: `feat(task-XX): <short description>`.

## 3) Architecture Boundaries (Keep Layers Clean)
- `src/controller`: input and window control only.
- `src/memory`: process attach/read/pointer resolution only.
- `src/state`: decode and normalize memory into typed snapshots.
- `src/env`: orchestrate control + state into `reset()`/`step()`.
- `src/training` and `src/agent`: policies, rewarding, loops, eval.
- `src/telemetry`: structured episode/step logging and KPIs.
- Never mix OS calls, policy logic, and state decoding in one module.

## 4) Existing Code Contracts to Preserve
- Use typed dataclasses for state/config/result payloads.
- Keep explicit machine-readable errors (`code`, message, metadata).
- Prefer result objects (`ReadResult`, `PointerChainResult`) over uncaught exceptions in core loops.
- Validate inputs early (`retries`, intervals, offsets, address ranges).
- Keep Windows-specific implementation behind protocol-style backends.

## 5) State/Offsets Rules
- `offsets.json` entries must be schema-valid and include confidence + notes.
- Normalize confidence to `low|medium|high` and keep notes actionable.
- Preserve extractor field naming expectations (`player_health`, `player_energy`, `player_credits`, `fail_state`, `run_active`, `collected_progs`) unless extractor is updated in the same change.
- For unknown inventory/prog IDs: preserve them in output; never drop silently.

## 6) Testing Requirements
- Every new non-trivial function needs tests.
- Follow existing test style: fake backends, deterministic inputs, no live game dependency.
- Cover both success and failure paths (attach failures, read failures, invalid config, null pointers, retries exhausted).
- Maintain deterministic behavior by injecting `sleep`, `time`, and timestamp hooks where needed.

## 7) Logging and Diagnostics
- Log key transitions with structured context (action, pid/hwnd, address, attempt, reason).
- Fail with actionable errors (what failed, where, and likely recovery action).
- Runtime helper tools (smoke test/TUI) may degrade gracefully, but core contracts should remain strict.

## 8) Platform and Safety Rules
- Windows-first is acceptable, but keep OS-specific code isolated behind interfaces.
- Avoid hard crashes for expected runtime instability (window lost, short reads, null pointers); return structured failure and allow caller recovery.
- Any recovery/retry loop must have bounded retries and explicit timeout semantics.

## 9) Anti-Regression Rules
- Do not break startup guards in [src/app.py](/c:/Users/John White/868-train/src/app.py): fingerprint and offsets validation remain mandatory.
- Avoid duplicating low-level memory decode logic; reuse shared primitives when extending tools.
- When introducing new task-era modules (`env`, `training`, `telemetry`, `agent`), define clear interfaces first, then integrate incrementally with tests.

## 10) Definition of Done for Future Tasks
- Acceptance criteria from `TASKS.md` are met and demonstrated.
- Unit tests added/updated and passing locally.
- Public interfaces are typed and documented via concise docstrings.
- Failure modes are explicit and observable in logs or structured results.

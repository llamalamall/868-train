# 868-hack Bot Plan (Codex Task Board)

## Project Objective
Build a Python application that can:
1. Control the 868-hack game client reliably.
2. Read game state from memory (from one pinned binary version).
3. Detect fail states and track key resources (health, energy, currency, collected progs).
4. Learn a stronger policy by running episodes, observing state transitions, and optimizing reward.

## Constraints
- Target one game binary version only.
- Reverse engineering done in Ghidra.
- Prefer reproducible, automated workflows and structured logs.
- Every task below is designed as a discrete chunk for GPT-5.3-Codex.

## Suggested Repo Layout
```text
868-train/
  TASKS.md
  README.md
  pyproject.toml
  src/
    app.py
    config/
      offsets.json
      binary_fingerprint.json
    controller/
      input_driver.py
      window_attach.py
      action_api.py
    memory/
      process_attach.py
      reader.py
      pointer_chain.py
      validators.py
    state/
      extractor.py
      schema.py
      fail_detector.py
    env/
      game_env.py
      reset_manager.py
    agent/
      baseline_random.py
      baseline_heuristic.py
      dqn_agent.py
    training/
      train.py
      evaluate.py
      rewards.py
    telemetry/
      logger.py
      metrics.py
  logs/
  notes/
    ghidra_findings.md
```

## Execution Order

### Task 01 - Bootstrap Project Skeleton
**Goal**
Create a minimal Python project with folders/modules for control, memory, state, env, and training.

**Deliverables**
- `pyproject.toml`
- `README.md`
- `src/` package structure
- `logs/` and `notes/` directories

**Acceptance Criteria**
- `python -m src.app` runs and prints a startup banner.
- Basic lint/type/test commands are documented.

**Codex Prompt**
```text
Set up a Python project in this repo for a game bot. Create:
- pyproject.toml (with dependencies placeholders),
- README.md,
- src package with modules for controller, memory, state, env, agent, training, telemetry,
- a runnable src/app.py entrypoint.
Use clear docstrings and minimal stubs so each module imports cleanly.
Then summarize what was created and any commands to run it.
```

---

### Task 02 - Pin Binary Version + Fingerprint Check
**Goal**
Prevent running against the wrong executable.

**Deliverables**
- `src/config/binary_fingerprint.json`
- Utility that computes and validates `sha256` of target binary

**Acceptance Criteria**
- Bot exits with clear error if hash mismatch.
- Hash check can be disabled only via explicit config flag.

**Codex Prompt**
```text
Add binary fingerprint validation:
- Create config schema with target executable path and expected sha256.
- Implement hash computation and startup validation.
- Fail fast with actionable error messages when mismatch occurs.
- Add docs showing how to update fingerprint intentionally.
Include unit tests for success/failure paths.
```

---

### Task 03 - Basic Window/Process Attach
**Goal**
Attach to the running game process and window safely.

**Deliverables**
- Process discovery by executable name/PID
- Window handle lookup and focus routine
- Retry/backoff on attach failures

**Acceptance Criteria**
- Script can find and focus the game window repeatedly.
- Clear logs when process/window not found.

**Codex Prompt**
```text
Implement process/window attach in src/controller and src/memory:
- Find process by name or PID.
- Find game window handle and bring to foreground.
- Add robust logging and retries.
- Expose a small API: attach_process(), attach_window(), focus_window().
Keep Windows support first (ctypes/pywin32), and isolate OS-specific code.
```

---

### Task 04 - Deterministic Input Action API
**Goal**
Establish reliable game control primitives.

**Deliverables**
- Action API: up/down/left/right/confirm/cancel/wait
- Configurable keybind map and timings
- Optional keypress verification hooks

**Acceptance Criteria**
- A scripted action sequence executes consistently across repeated runs.
- Input timing can be tuned via config, not hard-coded.

**Codex Prompt**
```text
Build a deterministic action API for keyboard control:
- Implement actions: move_up, move_down, move_left, move_right, confirm, cancel, wait.
- Support configurable key mapping and press/release timing.
- Add a simple scripted smoke test function that runs a fixed action sequence.
Log each action with timestamp and include retry logic for dropped input.
```

---

### Task 05 - Ghidra Recon Workflow + Offset Registry
**Goal**
Create a repeatable workflow for reverse engineering findings and storing offsets.

**Deliverables**
- `notes/ghidra_findings.md` template
- `src/config/offsets.json` schema for pointer chains/types

**Acceptance Criteria**
- Offset entries include field name, type, pointer chain/base, confidence, and notes.
- Loader validates required keys at startup.

**Codex Prompt**
```text
Create an offset registry system:
- Add src/config/offsets.json format for memory fields (fail_state, health, energy, currency, progs, etc.).
- Implement loader + schema validation.
- Create notes/ghidra_findings.md template documenting how each offset was discovered.
Include examples and confidence levels for each offset entry.
```

---

### Task 06 - Memory Read Primitives
**Goal**
Read primitive values and follow pointer chains safely.

**Deliverables**
- Process memory reader for int/float/bool/byte-array
- Pointer-chain resolver
- Address/value validators

**Acceptance Criteria**
- Reader returns stable values across repeated polling.
- Invalid chains fail gracefully without crashing the bot.

**Codex Prompt**
```text
Implement memory reading primitives:
- Add a reader that can read int32/int64/float/bool/bytes from process memory.
- Add pointer chain resolution from offsets config.
- Add guards for null pointers, invalid addresses, and out-of-range values.
- Return typed results plus structured errors.
Include tests with mocked memory backends.
```

---

### Task 07 - Fail-State Detection
**Goal**
Detect when an episode/run is lost.

**Deliverables**
- `state/fail_detector.py`
- Primary detector from memory, optional fallback hook

**Acceptance Criteria**
- Fail state flips to true within one polling interval when loss occurs.
- Terminal reason is logged.

**Codex Prompt**
```text
Implement fail-state detection:
- Build a detector that reads fail condition from memory offsets.
- Add polling loop helper with configurable interval.
- Emit structured event when terminal failure is detected.
- Include an interface for optional fallback detectors (e.g., pixel/UI) but keep memory detector primary.
```

---

### Task 08 - Core State Extraction (Health/Energy/Currency)
**Goal**
Build a normalized state snapshot per step.

**Deliverables**
- `state/schema.py` with typed state model
- `state/extractor.py` that reads all configured fields

**Acceptance Criteria**
- Snapshot contains health, energy, currency, fail flag, and timestamp.
- Missing fields are represented with explicit null/error metadata.

**Codex Prompt**
```text
Create a normalized game state extractor:
- Define a typed state schema with health, energy, currency, fail_state, and timestamp.
- Read values from offsets registry and memory reader.
- Attach per-field status metadata (ok/missing/invalid).
- Provide a single extract_state() function used by the rest of the app.
```

---

### Task 09 - Prog/Inventory Tracking
**Goal**
Decode and track collected progs and related metadata.

**Deliverables**
- Inventory/progs section in state schema
- Decoder for prog IDs/counts/flags based on offsets

**Acceptance Criteria**
- Collecting or losing a prog is reflected in consecutive snapshots.
- Unknown IDs are preserved (not dropped) and logged.

**Codex Prompt**
```text
Extend state extraction for collected progs:
- Add inventory/progs structures to state schema.
- Decode prog identifiers/counts/flags from memory.
- Preserve unknown prog IDs and log them for later mapping.
- Add tests for empty, partial, and populated inventory cases.
```

---

### Task 10 - Telemetry and Episode Logging
**Goal**
Capture high-quality data for training and debugging.

**Deliverables**
- Structured JSONL logger for actions, states, rewards, and terminal events
- Session/episode IDs and step counters

**Acceptance Criteria**
- One complete episode can be replayed from logs.
- Log files rotate or are namespaced per run.

**Codex Prompt**
```text
Implement telemetry logging:
- Log every step with action, pre/post state, reward, done flag, episode_id, step_index, timestamp.
- Write JSONL files under logs/ with per-run directory naming.
- Add helper utilities to load and summarize episodes.
- Ensure logging failures do not crash control loop.
```

---

### Task 11 - Gym-Like Environment Wrapper
**Goal**
Create a standard interface for learning loops.

**Deliverables**
- `env/game_env.py` with `reset()` and `step(action)`
- Internal synchronization between input timing and state polling

**Acceptance Criteria**
- Random policy can run N episodes unattended.
- `step()` returns `(state, reward, done, info)` consistently.

**Codex Prompt**
```text
Build a gym-like environment wrapper around the live game:
- Implement reset() and step(action)->(state, reward, done, info).
- Integrate action API, state extractor, and fail detector.
- Add timeout/watchdog protections to avoid hangs.
- Provide a random-policy runner script to validate end-to-end loop.
```

---

### Task 12 - Reward Function v1
**Goal**
Define a practical initial reward signal.

**Deliverables**
- `training/rewards.py` with configurable weights
- Reward components for survival, health change, currency change, fail penalty

**Acceptance Criteria**
- Reward output is deterministic given two consecutive states and done flag.
- Component breakdown is logged for debugging.

**Codex Prompt**
```text
Implement reward shaping v1:
- Reward positive survival/progress,
- Penalize health loss,
- Reward useful gains (e.g., currency),
- Large penalty on fail/terminal loss.
Make weights configurable via config and log per-component reward contributions.
```

---

### Task 13 - Baseline Agents (Random + Heuristic)
**Goal**
Establish non-ML baselines before deep RL.

**Deliverables**
- `agent/baseline_random.py`
- `agent/baseline_heuristic.py` (simple rule-based policy)

**Acceptance Criteria**
- Baselines run through full episodes using `game_env`.
- Metrics show baseline performance compared to pure random.

**Codex Prompt**
```text
Add baseline agents:
- Random action agent.
- Simple heuristic agent that reacts to state (e.g., conserve health/energy, avoid bad outcomes if detectable).
Integrate both with training/eval runner and produce comparable metrics.
```

---

### Task 14 - Learning Agent v1 (DQN or PPO)
**Goal**
Train a first ML policy on extracted state features.

**Deliverables**
- Initial RL agent implementation (choose DQN first unless state/action demands PPO)
- Checkpoint save/load

**Acceptance Criteria**
- Agent can train for multiple episodes without crashing.
- Checkpoint resumes training correctly.

**Codex Prompt**
```text
Implement a first learning agent for this environment:
- Prefer DQN on compact vector state; justify if PPO is better.
- Add replay buffer (if DQN), target updates, epsilon schedule.
- Save/load checkpoints and training metadata.
- Keep architecture simple and stable before optimization.
Provide a short rationale for chosen algorithm.
```

---

### Task 15 - Evaluation Harness + KPIs
**Goal**
Measure improvement rigorously.

**Deliverables**
- `training/evaluate.py` with fixed-seed eval runs
- KPI report: fail rate, avg episode length, avg health delta, avg currency gain

**Acceptance Criteria**
- Repeated evaluations are reproducible under same seed/config.
- Results can compare two checkpoints directly.

**Codex Prompt**
```text
Create an evaluation harness:
- Run fixed-seed episodes without exploration noise.
- Compute KPIs: fail rate, mean episode length, mean health delta, mean currency gain.
- Output machine-readable summary (JSON) plus human-readable table.
- Add checkpoint-vs-checkpoint comparison command.
```

---

### Task 16 - Reliability Hardening
**Goal**
Keep long training runs stable.

**Deliverables**
- Auto-recovery for lost process/window focus
- Memory read desync handling and reattach
- Timeouts/watchdogs around reset/step loops

**Acceptance Criteria**
- Overnight run can recover from transient attach/read/input failures.
- Fatal errors include actionable diagnostics.

**Codex Prompt**
```text
Harden runtime reliability:
- Add watchdogs and bounded retries around attach, input, state polling, and reset.
- Implement auto-reattach when process/window handles become invalid.
- Detect stale/invalid memory reads and trigger controlled recovery.
- Improve diagnostics with clear error categories and context.
```

---

### Task 17 - Regression Tests for Core Contracts
**Goal**
Prevent breakage as offsets and logic evolve.

**Deliverables**
- Tests for: config validation, pointer-chain resolution, fail detection, reward calculation, env step contract

**Acceptance Criteria**
- CI/local test suite catches contract regressions quickly.
- Mock-based tests run without launching the game.

**Codex Prompt**
```text
Add regression tests for core contracts:
- offsets config validation,
- pointer chain + memory reader behavior,
- fail-state detector behavior,
- reward determinism,
- game_env step/reset contract.
Use mocks/fakes so tests run offline without the actual game process.
```

---

### Task 18 - Training Runbook + Iteration Loop
**Goal**
Document operational workflow for repeated experiments.

**Deliverables**
- `README.md` sections for setup, run, train, evaluate, troubleshoot
- Experiment checklist and naming/versioning conventions

**Acceptance Criteria**
- A new run can be started and evaluated with documented commands only.
- Runbook includes failure triage flow.

**Codex Prompt**
```text
Write a practical runbook in README.md:
- setup/install,
- binary fingerprinting,
- offsets workflow from Ghidra notes,
- running control/state smoke tests,
- training and evaluation commands,
- troubleshooting common failures.
Include an experiment checklist and artifact naming conventions.
```

## Milestone Gates
- **Gate A (Control):** Tasks 01-04 complete.
- **Gate B (Observability):** Tasks 05-10 complete.
- **Gate C (Learning Loop):** Tasks 11-15 complete.
- **Gate D (Productionizing):** Tasks 16-18 complete.

## Recommended Execution Rhythm
1. Complete one task.
2. Run its acceptance checks.
3. Commit with a task-scoped message.
4. Move to next task only after logs/tests are green.

## Suggested Commit Message Pattern
`feat(task-XX): <short description>`

Examples:
- `feat(task-04): deterministic keyboard action API with retries`
- `feat(task-08): normalized state extractor for health energy currency`


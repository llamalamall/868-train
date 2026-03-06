# Ghidra Findings (Task 05 Template)

## Binary Context
- Binary file:
- Binary SHA256:
- Game version/build:
- Ghidra project name:
- Analysis date:

## Workflow Checklist
- Confirm binary fingerprint in `src/config/binary_fingerprint.json`.
- Locate candidate structures for run/fail state and player stats.
- Trace each candidate to a stable base (`module` or `absolute`) and pointer chain.
- Validate each candidate in live runtime by forcing state changes.
- Record confidence and failure modes before updating `src/config/offsets.json`.

## Offset Entry Template

Use one section per field you intend to add/update in `src/config/offsets.json`.

### Field: `<name>`
- `data_type`:
- `base.kind`: `module` or `absolute`
- `base.value`:
- `pointer_chain`:
- `read_offset`:
- `confidence`: `low` / `medium` / `high`
- `notes`:

### Discovery Notes
- Why this address chain is believed correct:
- Cross-check method used in runtime:
- Known edge cases / uncertainty:

## Suggested Initial Fields
- `fail_state`
- `player_health`
- `player_energy`
- `player_currency`
- `collected_progs`

## Session: 2026-03-05 Static Pass (Readable Decompiled Source)

### Field: `run_active`
- `data_type`: `bool`
- `base.kind`: `module`
- `base.value`: `868-HACK.exe`
- `pointer_chain`: `["0x115808"]`
- `read_offset`: `0x1B94`
- `confidence`: `medium`
- `notes`: Flag is serialized/deserialized and toggled in runtime control paths; derived fail-state is likely `!run_active`.

### Discovery Notes
- Why this address chain is believed correct:
  - Save/load reads and writes include `(arg_1 + 0x1B94)` as a boolean field.
  - Runtime logic branches on `*(char*)(state + 0x1B94)` for active update behavior.
- Cross-check method used in runtime:
  - `python -m src.memory.offset_smoke_test` attached successfully, but pointer root was null in current state; runtime in-level validation still pending.
- Known edge cases / uncertainty:
  - Semantics are likely "run active" rather than "terminal fail"; keep as `run_active` and derive `fail_state`.

### Field: `collected_progs`
- `data_type`: `array<int32>`
- `base.kind`: `module`
- `base.value`: `868-HACK.exe`
- `pointer_chain`: `["0x115808"]`
- `read_offset`: `0x1BB0`
- `confidence`: `medium`
- `notes`: Appears to be `std::vector<int>` layout with begin/end/capacity at `+0x1BB0/+0x1BB8/+0x1BC0`.

### Discovery Notes
- Why this address chain is believed correct:
  - Push-back style growth logic compares `+0x1BB8` and `+0x1BC0`, and uses helper calls with base `+0x1BB0`.
  - Serialization writes vector length via `(*(i64_t*)(+0x1BB8) - *(i64_t*)(+0x1BB0)) >> 2`.
- ID-to-name mapping method:
  - Prog names are in module `.rdata` table at `[868-HACK.exe+0x6A5B8]`.
  - Table rows are `6` pointers each; first pointer is prog short name (e.g. `.wait`, `.undo`, `.show`).
  - Collected-prog list values are IDs that index this table.
- Cross-check method used in runtime:
  - Pending in-level runtime validation while root pointer is non-null.
- Known edge cases / uncertainty:
  - Container content semantics must be confirmed (inventory vs temporary/runtime subset).

### Field: `player_health`
- `data_type`: `int32`
- `base.kind`: `module`
- `base.value`: `868-HACK.exe`
- `pointer_chain`: `["0x115808"]`
- `read_offset`: `0x18` (placeholder)
- `confidence`: `low`
- `notes`: Static pass did not find strong evidence for a health scalar at `+0x18`; likely needs remapping or replacement by other survivability signals.

## Session: 2026-03-06 Map/Entity Layout (Readable Decompiled Source)

### Candidate Block: `map_cells` (`6x6`, stride `0x38`) at `state + 0x11B8`
- Evidence:
  - Render path iterates `6` by `6` cells and indexes using `index = x*6 + y`.
  - `artifacts/readable/functions/fn_1400428e0.c` (`while ((int)(ivar_12 + 1U) < 6)` and `while (localv_88 < 6)`).
- Observed per-cell offsets (all `int32`, relative to cell base):
  - `+0x00` cell type (`0` normal, `1` prog-wall, `2` points-wall, `3` exit seen)
  - `+0x08` tile variant
  - `+0x0C` credits
  - `+0x10` energy
  - `+0x14` prog id (`-1` means none)
  - `+0x18` wall/cell state
  - `+0x1C` threat/penalty accumulator input
  - `+0x20` points
  - `+0x24` siphon marker
  - `+0x28` special-state flag
  - `+0x2C` exit-overlay flag
  - `+0x30` lock/hidden-style flag
  - `+0x34` marker flag

### Reward Semantics Confirmation
- Siphon/collect path updates global resources from cell fields:
  - `+0x0C` -> credits (`+0x1B5C`)
  - `+0x10` -> energy (`+0x1B60`)
  - `+0x20` -> score (`+0x19DC`)
  - `+0x14` -> collected-prog handling when non-negative
- Reference: `artifacts/readable/functions/fn_140046250.c`.

### Derived Locations
- Siphons:
  - `cell + 0x24 > 0` (`state + 0x11DC` per cell).
  - Hover path uses `fn_140039ab0(..., "data siphon", ...)` under this condition.
- Exit:
  - `cell_type == 3` (`cell + 0x00`).
- Walls:
  - `cell_type == 1`: prog-wall tooltip path.
  - `cell_type == 2`: points-wall tooltip path.
  - Reference: `artifacts/readable/functions/fn_1400428e0.c`.

### Candidate Block: `entities` (`64`, stride `0x44`) at `state + 0x0C`
- Evidence:
  - Lookup scans up to `0x40` slots and returns `state + 0x0C + slot*0x44`.
  - Reference: `artifacts/readable/functions/fn_14004d180.c`.
- Useful per-entity offsets:
  - `+0x00` active flag (byte)
  - `+0x08` type id (`int32`)
  - `+0x0C` hp/state-like value (`int32`)
  - `+0x18` state/cooldown-like value (`int32`)
  - `+0x34` x coordinate (`int32`)
  - `+0x38` y coordinate (`int32`)

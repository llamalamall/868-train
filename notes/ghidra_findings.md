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

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

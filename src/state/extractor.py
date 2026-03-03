"""Normalized game-state extraction from configured memory offsets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.config.offsets import OffsetEntry, OffsetRegistry
from src.memory.pointer_chain import ModuleBaseResolver, resolve_offset_entry_address
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult
from src.state.schema import FieldState, GameStateSnapshot

_HEALTH_FIELD_CANDIDATES = ("player_health", "health")
_ENERGY_FIELD_CANDIDATES = ("player_energy", "energy")
_CURRENCY_FIELD_CANDIDATES = ("player_credits", "player_currency", "currency")
_FAIL_FIELD_CANDIDATES = ("fail_state",)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _missing_field(error_code: str, *, source_field: str | None = None, error: str | None = None) -> FieldState:
    return FieldState(
        value=None,
        status="missing",
        error_code=error_code,
        error=error,
        source_field=source_field,
    )


def _invalid_field(
    error_code: str,
    *,
    address: int | None = None,
    source_field: str | None = None,
    error: str | None = None,
) -> FieldState:
    return FieldState(
        value=None,
        status="invalid",
        error_code=error_code,
        error=error,
        address=address,
        source_field=source_field,
    )


def _ok_field(value: Any, *, address: int | None, source_field: str) -> FieldState:
    return FieldState(
        value=value,
        status="ok",
        address=address,
        source_field=source_field,
    )


def _entry_by_name(registry: OffsetRegistry) -> dict[str, OffsetEntry]:
    return {entry.name: entry for entry in registry.entries}


def _select_entry(
    entries_by_name: dict[str, OffsetEntry],
    candidates: tuple[str, ...],
) -> OffsetEntry | None:
    for name in candidates:
        entry = entries_by_name.get(name)
        if entry is not None:
            return entry
    return None


def _read_uint32(reader: ProcessMemoryReader, address: int) -> ReadResult[int]:
    raw = reader.read_bytes(address, 4)
    if not raw.is_ok:
        return ReadResult.fail(
            raw.error or ReadFailure(code="read_failed", message="Read failed.", address=address, size=4)
        )
    return ReadResult.ok(int.from_bytes(raw.value or b"\x00\x00\x00\x00", "little", signed=False))


def _decode_value(reader: ProcessMemoryReader, entry: OffsetEntry, address: int) -> ReadResult[Any]:
    normalized = entry.data_type.strip().lower()
    if normalized == "int32":
        return reader.read_int32(address)
    if normalized == "int64":
        return reader.read_int64(address)
    if normalized == "uint32":
        return _read_uint32(reader, address)
    if normalized == "uint64":
        return reader.read_uint64(address)
    if normalized == "float":
        return reader.read_float32(address)
    if normalized == "bool":
        return reader.read_bool(address)
    return ReadResult.fail(
        ReadFailure(
            code="unsupported_data_type",
            message=f"Unsupported data_type '{entry.data_type}'.",
            address=address,
        )
    )


def _extract_field(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> FieldState:
    if entry is None:
        return _missing_field("field_not_configured")

    resolved = resolve_offset_entry_address(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if not resolved.is_ok or resolved.value is None:
        error_code = resolved.error.code if resolved.error is not None else "resolve_failed"
        error_message = resolved.error.message if resolved.error is not None else "Address resolution failed."
        return _invalid_field(
            error_code,
            source_field=entry.name,
            error=error_message,
        )

    address = resolved.value
    decoded = _decode_value(reader, entry, address)
    if not decoded.is_ok:
        code = decoded.error.code if decoded.error is not None else "read_failed"
        message = decoded.error.message if decoded.error is not None else "Read failed."
        return _invalid_field(
            code,
            address=address,
            source_field=entry.name,
            error=message,
        )

    return _ok_field(decoded.value, address=address, source_field=entry.name)


def _derive_fail_state(
    explicit_fail_state: FieldState,
    health: FieldState,
    terminal_health_value: int,
) -> FieldState:
    if explicit_fail_state.status == "ok":
        return explicit_fail_state
    if health.status != "ok":
        return _missing_field(
            "fail_state_unavailable",
            source_field=health.source_field,
            error="Health field unavailable; cannot derive fail_state.",
        )
    try:
        numeric_health = int(float(health.value))
    except (TypeError, ValueError):
        return _invalid_field(
            "health_not_numeric",
            source_field=health.source_field,
            address=health.address,
            error="Health value is non-numeric; cannot derive fail_state.",
        )

    return _ok_field(
        numeric_health == terminal_health_value,
        address=health.address,
        source_field=f"derived:{health.source_field}=={terminal_health_value}",
    )


def extract_state(
    *,
    reader: ProcessMemoryReader,
    registry: OffsetRegistry,
    module_base_resolver: ModuleBaseResolver | None = None,
    terminal_health_value: int = -1,
    timestamp_fn: Callable[[], str] = _now_iso_utc,
) -> GameStateSnapshot:
    """Extract a normalized state snapshot with per-field status metadata."""
    entries = _entry_by_name(registry)
    health_entry = _select_entry(entries, _HEALTH_FIELD_CANDIDATES)
    energy_entry = _select_entry(entries, _ENERGY_FIELD_CANDIDATES)
    currency_entry = _select_entry(entries, _CURRENCY_FIELD_CANDIDATES)
    fail_entry = _select_entry(entries, _FAIL_FIELD_CANDIDATES)

    health = _extract_field(reader=reader, entry=health_entry, module_base_resolver=module_base_resolver)
    energy = _extract_field(reader=reader, entry=energy_entry, module_base_resolver=module_base_resolver)
    currency = _extract_field(reader=reader, entry=currency_entry, module_base_resolver=module_base_resolver)
    explicit_fail_state = _extract_field(
        reader=reader,
        entry=fail_entry,
        module_base_resolver=module_base_resolver,
    )
    fail_state = _derive_fail_state(explicit_fail_state, health, terminal_health_value)

    known_names = {
        entry.name for entry in (health_entry, energy_entry, currency_entry, fail_entry) if entry is not None
    }
    extra_fields: dict[str, FieldState] = {}
    for entry in registry.entries:
        if entry.name in known_names:
            continue
        extra_fields[entry.name] = _extract_field(
            reader=reader,
            entry=entry,
            module_base_resolver=module_base_resolver,
        )

    return GameStateSnapshot(
        timestamp_utc=timestamp_fn(),
        health=health,
        energy=energy,
        currency=currency,
        fail_state=fail_state,
        extra_fields=extra_fields,
    )

"""Normalized game-state extraction from configured memory offsets."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from src.config.offsets import OffsetEntry, OffsetRegistry
from src.memory.pointer_chain import ModuleBaseResolver, resolve_offset_entry_address
from src.memory.reader import ProcessMemoryReader, ReadFailure, ReadResult
from src.state.prog_catalog import PROG_NAME_BY_ID
from src.state.schema import FieldState, GameStateSnapshot, InventoryState, ProgInventoryItem

LOGGER = logging.getLogger(__name__)

_HEALTH_FIELD_CANDIDATES = ("player_health", "health")
_ENERGY_FIELD_CANDIDATES = ("player_energy", "energy")
_CURRENCY_FIELD_CANDIDATES = ("player_credits", "player_currency", "currency")
_COLLECTED_PROGS_FIELD_CANDIDATES = ("collected_progs",)
_MAX_VECTOR_INT32_ITEMS = 4096

_LOGGED_UNKNOWN_PROG_IDS: set[int] = set()


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


def _missing_inventory(
    error_code: str,
    *,
    source_field: str | None = None,
    error: str | None = None,
) -> InventoryState:
    return InventoryState(
        status="missing",
        error_code=error_code,
        error=error,
        source_field=source_field,
    )


def _invalid_inventory(
    error_code: str,
    *,
    source_field: str | None = None,
    address: int | None = None,
    error: str | None = None,
    raw_prog_ids: tuple[int, ...] = (),
) -> InventoryState:
    return InventoryState(
        status="invalid",
        error_code=error_code,
        error=error,
        source_field=source_field,
        address=address,
        raw_prog_ids=raw_prog_ids,
    )


def _ok_inventory(
    *,
    source_field: str,
    address: int | None,
    raw_prog_ids: tuple[int, ...],
    collected_progs: tuple[ProgInventoryItem, ...],
    unknown_prog_ids: tuple[int, ...],
) -> InventoryState:
    return InventoryState(
        status="ok",
        source_field=source_field,
        address=address,
        raw_prog_ids=raw_prog_ids,
        collected_progs=collected_progs,
        unknown_prog_ids=unknown_prog_ids,
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


def _read_vector_int32(reader: ProcessMemoryReader, address: int) -> ReadResult[tuple[int, ...]]:
    begin_result = reader.read_pointer(address)
    if not begin_result.is_ok:
        return ReadResult.fail(
            begin_result.error
            or ReadFailure(
                code="vector_begin_read_failed",
                message="Failed reading vector begin pointer.",
                address=address,
                size=reader.pointer_size,
            )
        )
    end_address = address + reader.pointer_size
    end_result = reader.read_pointer(end_address)
    if not end_result.is_ok:
        return ReadResult.fail(
            end_result.error
            or ReadFailure(
                code="vector_end_read_failed",
                message="Failed reading vector end pointer.",
                address=end_address,
                size=reader.pointer_size,
            )
        )

    begin = int(begin_result.value or 0)
    end = int(end_result.value or 0)
    if begin == 0 and end == 0:
        return ReadResult.ok(())
    if begin == 0 or end == 0:
        return ReadResult.fail(
            ReadFailure(
                code="vector_null_boundary",
                message=f"Invalid vector boundary begin=0x{begin:X} end=0x{end:X}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )
    if end < begin:
        return ReadResult.fail(
            ReadFailure(
                code="vector_end_before_begin",
                message=f"Invalid vector range begin=0x{begin:X} end=0x{end:X}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )
    span = end - begin
    if span % 4 != 0:
        return ReadResult.fail(
            ReadFailure(
                code="vector_unaligned_span",
                message=f"Vector span {span} is not aligned to int32.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )

    count = span // 4
    if count > _MAX_VECTOR_INT32_ITEMS:
        return ReadResult.fail(
            ReadFailure(
                code="vector_too_large",
                message=f"Vector item count {count} exceeds limit {_MAX_VECTOR_INT32_ITEMS}.",
                address=address,
                size=reader.pointer_size * 2,
            )
        )

    values: list[int] = []
    for index in range(count):
        item_address = begin + index * 4
        item_result = reader.read_int32(item_address)
        if not item_result.is_ok:
            return ReadResult.fail(
                item_result.error
                or ReadFailure(
                    code="vector_item_read_failed",
                    message=f"Failed reading vector item at index {index}.",
                    address=item_address,
                    size=4,
                )
            )
        values.append(int(item_result.value or 0))
    return ReadResult.ok(tuple(values))


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
    if normalized == "array<int32>":
        return _read_vector_int32(reader, address)
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


def _log_unknown_prog_ids_once(unknown_prog_ids: tuple[int, ...]) -> None:
    for prog_id in unknown_prog_ids:
        if prog_id in _LOGGED_UNKNOWN_PROG_IDS:
            continue
        _LOGGED_UNKNOWN_PROG_IDS.add(prog_id)
        LOGGER.warning(
            "Unknown collected prog id detected: %s. Preserving id in snapshot for later mapping.",
            prog_id,
        )


def _build_prog_inventory(raw_prog_ids: tuple[int, ...]) -> tuple[tuple[ProgInventoryItem, ...], tuple[int, ...]]:
    counts = Counter(raw_prog_ids)
    seen: set[int] = set()
    ordered_ids: list[int] = []
    for prog_id in raw_prog_ids:
        if prog_id in seen:
            continue
        seen.add(prog_id)
        ordered_ids.append(prog_id)

    unknown: list[int] = []
    items: list[ProgInventoryItem] = []
    for prog_id in ordered_ids:
        name = PROG_NAME_BY_ID.get(prog_id)
        known = name is not None
        flags = () if known else ("unknown_id",)
        if not known:
            unknown.append(prog_id)
        items.append(
            ProgInventoryItem(
                prog_id=prog_id,
                count=counts[prog_id],
                name=name,
                known=known,
                flags=flags,
            )
        )
    unknown_ids = tuple(unknown)
    _log_unknown_prog_ids_once(unknown_ids)
    return (tuple(items), unknown_ids)


def _extract_inventory(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry | None,
    module_base_resolver: ModuleBaseResolver | None,
) -> InventoryState:
    if entry is None:
        return _missing_inventory("field_not_configured")

    resolved = resolve_offset_entry_address(
        reader=reader,
        entry=entry,
        module_base_resolver=module_base_resolver,
    )
    if not resolved.is_ok or resolved.value is None:
        error_code = resolved.error.code if resolved.error is not None else "resolve_failed"
        error_message = resolved.error.message if resolved.error is not None else "Address resolution failed."
        return _invalid_inventory(
            error_code,
            source_field=entry.name,
            error=error_message,
        )

    address = resolved.value
    decoded = _decode_value(reader, entry, address)
    if not decoded.is_ok:
        code = decoded.error.code if decoded.error is not None else "read_failed"
        message = decoded.error.message if decoded.error is not None else "Read failed."
        return _invalid_inventory(
            code,
            source_field=entry.name,
            address=address,
            error=message,
        )

    raw_prog_ids = tuple(int(value) for value in (decoded.value or ()))
    items, unknown_prog_ids = _build_prog_inventory(raw_prog_ids)
    return _ok_inventory(
        source_field=entry.name,
        address=address,
        raw_prog_ids=raw_prog_ids,
        collected_progs=items,
        unknown_prog_ids=unknown_prog_ids,
    )


def _derive_fail_state(
    health: FieldState,
    terminal_health_value: int,
) -> FieldState:
    if health.status != "ok":
        return _missing_field(
            "fail_state_unavailable",
            source_field=health.source_field,
            error="Health is unavailable; cannot derive fail_state.",
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
    collected_progs_entry = _select_entry(entries, _COLLECTED_PROGS_FIELD_CANDIDATES)

    health = _extract_field(reader=reader, entry=health_entry, module_base_resolver=module_base_resolver)
    energy = _extract_field(reader=reader, entry=energy_entry, module_base_resolver=module_base_resolver)
    currency = _extract_field(reader=reader, entry=currency_entry, module_base_resolver=module_base_resolver)
    inventory = _extract_inventory(
        reader=reader,
        entry=collected_progs_entry,
        module_base_resolver=module_base_resolver,
    )
    fail_state = _derive_fail_state(
        health,
        terminal_health_value,
    )

    known_names = {
        entry.name
        for entry in (
            health_entry,
            energy_entry,
            currency_entry,
            collected_progs_entry,
        )
        if entry is not None
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
        inventory=inventory,
        extra_fields=extra_fields,
    )

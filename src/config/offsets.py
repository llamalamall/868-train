"""Offset registry loading and schema validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_HEX_OFFSET_REGEX = re.compile(r"^0x[0-9a-fA-F]+$")
_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_BASE_KINDS = {"module", "absolute"}


class OffsetRegistryValidationError(RuntimeError):
    """Raised when offsets registry configuration is malformed."""


@dataclass(frozen=True)
class OffsetBase:
    """Base address source for an offset entry."""

    kind: Literal["module", "absolute"]
    value: str


@dataclass(frozen=True)
class OffsetEntry:
    """Single memory field entry with pointer-chain traversal metadata."""

    name: str
    data_type: str
    base: OffsetBase
    pointer_chain: tuple[int, ...]
    confidence: Literal["low", "medium", "high"]
    notes: str
    read_offset: int = 0


@dataclass(frozen=True)
class OffsetRegistry:
    """Top-level offsets registry."""

    version: int
    entries: tuple[OffsetEntry, ...]


def _default_config_path() -> Path:
    return Path(__file__).with_name("offsets.json")


def _error(message: str) -> OffsetRegistryValidationError:
    return OffsetRegistryValidationError(message)


def _require_key(data: dict[str, Any], key: str, expected_type: type, where: str) -> Any:
    if key not in data:
        raise _error(f"Missing required key '{key}' in {where}.")
    value = data[key]
    if not isinstance(value, expected_type):
        raise _error(
            f"Key '{key}' in {where} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _parse_offset_value(value: Any, where: str) -> int:
    if isinstance(value, int):
        if value < 0:
            raise _error(f"{where} must be >= 0.")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not _HEX_OFFSET_REGEX.fullmatch(stripped):
            raise _error(f"{where} must be a non-negative int or hex string like 0x1A2B.")
        return int(stripped, 16)
    raise _error(f"{where} must be a non-negative int or hex string.")


def _parse_base(raw: dict[str, Any], where: str) -> OffsetBase:
    kind = _require_key(raw, "kind", str, where).strip().lower()
    value = _require_key(raw, "value", str, where).strip()
    if kind not in _BASE_KINDS:
        raise _error(
            f"Base kind '{kind}' in {where} is invalid. Supported kinds: {sorted(_BASE_KINDS)}."
        )
    if not value:
        raise _error(f"Base value in {where} cannot be empty.")
    if kind == "absolute" and not _HEX_OFFSET_REGEX.fullmatch(value):
        raise _error(f"Absolute base value in {where} must be a hex string like 0x140000000.")
    return OffsetBase(kind=kind, value=value)


def _parse_entry(raw: dict[str, Any], index: int) -> OffsetEntry:
    where = f"entry[{index}]"
    name = _require_key(raw, "name", str, where).strip()
    data_type = _require_key(raw, "data_type", str, where).strip()
    base = _parse_base(_require_key(raw, "base", dict, where), where=f"{where}.base")
    raw_pointer_chain = _require_key(raw, "pointer_chain", list, where)
    confidence = _require_key(raw, "confidence", str, where).strip().lower()
    notes = _require_key(raw, "notes", str, where).strip()
    read_offset_raw = raw.get("read_offset", 0)

    if not name:
        raise _error(f"{where}.name cannot be empty.")
    if not data_type:
        raise _error(f"{where}.data_type cannot be empty.")
    if confidence not in _CONFIDENCE_LEVELS:
        raise _error(
            f"{where}.confidence '{confidence}' is invalid. "
            f"Supported values: {sorted(_CONFIDENCE_LEVELS)}."
        )
    if not notes:
        raise _error(f"{where}.notes cannot be empty.")

    pointer_chain: list[int] = []
    for pointer_index, pointer_value in enumerate(raw_pointer_chain):
        pointer_chain.append(
            _parse_offset_value(
                pointer_value,
                where=f"{where}.pointer_chain[{pointer_index}]",
            )
        )

    read_offset = _parse_offset_value(read_offset_raw, where=f"{where}.read_offset")
    return OffsetEntry(
        name=name,
        data_type=data_type,
        base=base,
        pointer_chain=tuple(pointer_chain),
        confidence=confidence,
        notes=notes,
        read_offset=read_offset,
    )


def load_offset_registry(config_path: Path | None = None) -> OffsetRegistry:
    """Load and validate the offsets registry configuration."""
    path = config_path or _default_config_path()
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except FileNotFoundError as error:
        raise _error(f"Offsets config not found: {path}.") from error
    except json.JSONDecodeError as error:
        raise _error(f"Offsets config is not valid JSON ({path}): {error}") from error

    if not isinstance(raw, dict):
        raise _error(f"Offsets config must be a JSON object: {path}.")

    version = _require_key(raw, "version", int, "root")
    raw_entries = _require_key(raw, "entries", list, "root")
    if version < 1:
        raise _error("Offsets config version must be >= 1.")
    if not raw_entries:
        raise _error("Offsets config must define at least one entry.")

    entries: list[OffsetEntry] = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise _error(f"entry[{index}] must be a JSON object.")
        entry = _parse_entry(raw_entry, index)
        if entry.name in seen_names:
            raise _error(f"Duplicate entry name '{entry.name}' in offsets config.")
        seen_names.add(entry.name)
        entries.append(entry)

    return OffsetRegistry(version=version, entries=tuple(entries))


def ensure_offsets_registry_valid(config_path: Path | None = None) -> None:
    """Validate offsets config and raise on malformed content."""
    load_offset_registry(config_path=config_path)

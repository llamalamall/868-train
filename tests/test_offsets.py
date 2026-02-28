"""Tests for offsets registry schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.offsets import (
    OffsetRegistryValidationError,
    ensure_offsets_registry_valid,
    load_offset_registry,
)


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="ascii")


def _valid_entry(name: str) -> dict:
    return {
        "name": name,
        "data_type": "int32",
        "base": {"kind": "module", "value": "868-HACK.exe"},
        "pointer_chain": ["0x10", "0x20", 48],
        "read_offset": "0x4",
        "confidence": "medium",
        "notes": "Validated through live value changes.",
    }


def test_load_offset_registry_parses_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "offsets.json"
    _write_config(
        config_path,
        {
            "version": 1,
            "entries": [_valid_entry("fail_state"), _valid_entry("player_health")],
        },
    )

    registry = load_offset_registry(config_path)
    assert registry.version == 1
    assert len(registry.entries) == 2
    assert registry.entries[0].name == "fail_state"
    assert registry.entries[0].pointer_chain == (16, 32, 48)
    assert registry.entries[0].read_offset == 4


def test_ensure_offsets_registry_valid_accepts_default_config_file() -> None:
    ensure_offsets_registry_valid()


def test_load_offset_registry_rejects_missing_required_key(tmp_path: Path) -> None:
    entry = _valid_entry("player_energy")
    del entry["notes"]
    config_path = tmp_path / "offsets.json"
    _write_config(config_path, {"version": 1, "entries": [entry]})

    with pytest.raises(OffsetRegistryValidationError, match="Missing required key 'notes'"):
        load_offset_registry(config_path)


def test_load_offset_registry_rejects_invalid_confidence(tmp_path: Path) -> None:
    entry = _valid_entry("player_currency")
    entry["confidence"] = "certain"
    config_path = tmp_path / "offsets.json"
    _write_config(config_path, {"version": 1, "entries": [entry]})

    with pytest.raises(OffsetRegistryValidationError, match="confidence"):
        load_offset_registry(config_path)


def test_load_offset_registry_rejects_invalid_base_kind(tmp_path: Path) -> None:
    entry = _valid_entry("collected_progs")
    entry["base"] = {"kind": "region", "value": "868-HACK.exe"}
    config_path = tmp_path / "offsets.json"
    _write_config(config_path, {"version": 1, "entries": [entry]})

    with pytest.raises(OffsetRegistryValidationError, match="Base kind"):
        load_offset_registry(config_path)


def test_load_offset_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    config_path = tmp_path / "offsets.json"
    _write_config(
        config_path,
        {
            "version": 1,
            "entries": [_valid_entry("fail_state"), _valid_entry("fail_state")],
        },
    )

    with pytest.raises(OffsetRegistryValidationError, match="Duplicate entry name"):
        load_offset_registry(config_path)


def test_load_offset_registry_rejects_bad_pointer_chain_value(tmp_path: Path) -> None:
    entry = _valid_entry("player_health")
    entry["pointer_chain"] = ["bad-offset"]
    config_path = tmp_path / "offsets.json"
    _write_config(config_path, {"version": 1, "entries": [entry]})

    with pytest.raises(OffsetRegistryValidationError, match="pointer_chain"):
        load_offset_registry(config_path)

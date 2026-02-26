"""Tests for binary fingerprint loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.config.fingerprint import (
    FingerprintValidationError,
    compute_sha256,
    ensure_binary_fingerprint_valid,
    load_binary_fingerprint_config,
)


def _write_config(path: Path, enabled: bool, binary_path: Path, expected_sha256: str) -> None:
    payload = {
        "enabled": enabled,
        "binary_path": str(binary_path),
        "expected_sha256": expected_sha256,
    }
    path.write_text(json.dumps(payload), encoding="ascii")


def test_compute_sha256_matches_known_value(tmp_path: Path) -> None:
    binary = tmp_path / "game.exe"
    binary.write_bytes(b"868-hack")
    expected = hashlib.sha256(b"868-hack").hexdigest()
    assert compute_sha256(binary) == expected


def test_validation_passes_when_hash_matches(tmp_path: Path) -> None:
    binary = tmp_path / "game.exe"
    binary.write_bytes(b"same-bytes")
    config = tmp_path / "binary_fingerprint.json"
    _write_config(config, True, binary, compute_sha256(binary))
    ensure_binary_fingerprint_valid(config)


def test_validation_fails_when_hash_mismatch(tmp_path: Path) -> None:
    binary = tmp_path / "game.exe"
    binary.write_bytes(b"expected-one")
    config = tmp_path / "binary_fingerprint.json"
    _write_config(config, True, binary, "f" * 64)

    with pytest.raises(FingerprintValidationError, match="fingerprint mismatch"):
        ensure_binary_fingerprint_valid(config)


def test_validation_skips_when_disabled(tmp_path: Path) -> None:
    config = tmp_path / "binary_fingerprint.json"
    _write_config(config, False, tmp_path / "missing.exe", "0" * 64)
    ensure_binary_fingerprint_valid(config)


def test_loader_rejects_bad_sha_format_when_enabled(tmp_path: Path) -> None:
    binary = tmp_path / "game.exe"
    binary.write_bytes(b"anything")
    config = tmp_path / "binary_fingerprint.json"
    _write_config(config, True, binary, "not-a-sha")

    with pytest.raises(FingerprintValidationError, match="expected_sha256"):
        load_binary_fingerprint_config(config)

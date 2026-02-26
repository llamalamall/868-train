"""Binary fingerprint configuration and validation utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256_REGEX = re.compile(r"^[a-fA-F0-9]{64}$")


class FingerprintValidationError(RuntimeError):
    """Raised when binary fingerprint configuration or validation fails."""


@dataclass(frozen=True)
class BinaryFingerprintConfig:
    """Runtime configuration for executable hash validation."""

    enabled: bool
    binary_path: Path
    expected_sha256: str


def _default_config_path() -> Path:
    return Path(__file__).with_name("binary_fingerprint.json")


def _require_key(data: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in data:
        raise FingerprintValidationError(
            f"Missing required key '{key}' in binary fingerprint config."
        )
    value = data[key]
    if not isinstance(value, expected_type):
        raise FingerprintValidationError(
            f"Config key '{key}' must be of type {expected_type.__name__}."
        )
    return value


def load_binary_fingerprint_config(config_path: Path | None = None) -> BinaryFingerprintConfig:
    """Load and validate binary fingerprint configuration from JSON."""
    path = config_path or _default_config_path()
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except FileNotFoundError as error:
        raise FingerprintValidationError(
            f"Fingerprint config not found: {path}. Create this file before running."
        ) from error
    except json.JSONDecodeError as error:
        raise FingerprintValidationError(
            f"Fingerprint config is not valid JSON ({path}): {error}"
        ) from error

    if not isinstance(raw, dict):
        raise FingerprintValidationError(f"Fingerprint config must be a JSON object: {path}")

    enabled = _require_key(raw, "enabled", bool)
    binary_path_value = _require_key(raw, "binary_path", str).strip()
    expected_sha256 = _require_key(raw, "expected_sha256", str).strip().lower()

    if enabled:
        if not binary_path_value:
            raise FingerprintValidationError(
                "Config key 'binary_path' cannot be empty when enabled is true."
            )
        if not _SHA256_REGEX.fullmatch(expected_sha256):
            raise FingerprintValidationError(
                "Config key 'expected_sha256' must be a 64-character hex SHA256 value "
                "when enabled is true."
            )

    return BinaryFingerprintConfig(
        enabled=enabled,
        binary_path=Path(binary_path_value).expanduser(),
        expected_sha256=expected_sha256,
    )


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 for a file path."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_binary_fingerprint_valid(config_path: Path | None = None) -> None:
    """Validate executable hash against pinned configuration.

    Raises:
        FingerprintValidationError: If configuration or file hash is invalid.
    """
    config = load_binary_fingerprint_config(config_path=config_path)
    if not config.enabled:
        return

    if not config.binary_path.exists():
        raise FingerprintValidationError(
            f"Configured binary_path does not exist: {config.binary_path}. "
            "Update src/config/binary_fingerprint.json with the correct executable path."
        )

    actual_sha256 = compute_sha256(config.binary_path)
    if actual_sha256.lower() != config.expected_sha256.lower():
        raise FingerprintValidationError(
            "Executable fingerprint mismatch. "
            f"Expected {config.expected_sha256}, got {actual_sha256} for {config.binary_path}. "
            "If this is an intentional binary update, compute the new SHA256 and update "
            "src/config/binary_fingerprint.json."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary fingerprint helper")
    parser.add_argument("--print-sha256", type=Path, help="Print SHA256 for the given file path")
    return parser


def main() -> None:
    """CLI entrypoint for local fingerprint utility actions."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.print_sha256 is None:
        parser.print_help()
        return

    print(compute_sha256(args.print_sha256))


if __name__ == "__main__":
    main()

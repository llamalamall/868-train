"""Application entrypoint for the 868-train scaffold."""

from __future__ import annotations

from src.config.fingerprint import FingerprintValidationError, ensure_binary_fingerprint_valid
from src.config.offsets import OffsetRegistryValidationError, ensure_offsets_registry_valid


def main() -> None:
    """Run startup checks and print a bootstrap banner."""
    try:
        ensure_binary_fingerprint_valid()
        ensure_offsets_registry_valid()
    except (FingerprintValidationError, OffsetRegistryValidationError) as error:
        raise SystemExit(f"Startup validation failed: {error}") from error

    print("868-train bootstrap ready")


if __name__ == "__main__":
    main()

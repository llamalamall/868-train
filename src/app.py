"""Application entrypoint for the 868-train scaffold."""

from __future__ import annotations

from src.config.fingerprint import FingerprintValidationError, ensure_binary_fingerprint_valid


def main() -> None:
    """Run startup checks and print a bootstrap banner."""
    try:
        ensure_binary_fingerprint_valid()
    except FingerprintValidationError as error:
        raise SystemExit(f"Startup validation failed: {error}") from error

    print("868-train bootstrap ready")


if __name__ == "__main__":
    main()

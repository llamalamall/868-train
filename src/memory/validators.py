"""Memory validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MIN_USER_ADDRESS = 0x10000
DEFAULT_MAX_USER_ADDRESS = 0x00007FFFFFFFFFFF


@dataclass(frozen=True)
class AddressRange:
    """Allowed virtual address range for read operations."""

    min_address: int = DEFAULT_MIN_USER_ADDRESS
    max_address: int = DEFAULT_MAX_USER_ADDRESS

    def contains(self, address: int) -> bool:
        """Return whether an address is inside the allowed range."""
        return self.min_address <= address <= self.max_address


@dataclass(frozen=True)
class ValidationIssue:
    """Structured validation issue with a machine-readable code."""

    code: str
    message: str


def validate_address(
    address: int,
    *,
    allowed_range: AddressRange | None = None,
    allow_null: bool = False,
) -> ValidationIssue | None:
    """Validate a process virtual address."""
    if not isinstance(address, int):
        return ValidationIssue(
            code="invalid_address_type",
            message=f"Address must be int, got {type(address).__name__}.",
        )

    if address < 0:
        return ValidationIssue(
            code="negative_address",
            message=f"Address must be >= 0, got {address}.",
        )

    if address == 0 and not allow_null:
        return ValidationIssue(
            code="null_address",
            message="Address cannot be null.",
        )

    if address == 0 and allow_null:
        return None

    active_range = allowed_range or AddressRange()
    if not active_range.contains(address):
        return ValidationIssue(
            code="address_out_of_range",
            message=(
                f"Address 0x{address:X} is outside allowed user range "
                f"0x{active_range.min_address:X}..0x{active_range.max_address:X}."
            ),
        )

    return None


def validate_read_size(size: int) -> ValidationIssue | None:
    """Validate byte count requested from memory."""
    if not isinstance(size, int):
        return ValidationIssue(
            code="invalid_size_type",
            message=f"Read size must be int, got {type(size).__name__}.",
        )
    if size <= 0:
        return ValidationIssue(
            code="invalid_size_value",
            message=f"Read size must be > 0, got {size}.",
        )
    return None


def validate_numeric_range(
    value: int | float,
    *,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    field_name: str = "value",
) -> ValidationIssue | None:
    """Validate a numeric value against optional min/max bounds."""
    if min_value is not None and value < min_value:
        return ValidationIssue(
            code="value_below_minimum",
            message=f"{field_name}={value} is below minimum {min_value}.",
        )
    if max_value is not None and value > max_value:
        return ValidationIssue(
            code="value_above_maximum",
            message=f"{field_name}={value} is above maximum {max_value}.",
        )
    return None

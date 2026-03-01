"""Tests for memory validators."""

from __future__ import annotations

from src.memory.validators import (
    AddressRange,
    validate_address,
    validate_numeric_range,
    validate_read_size,
)


def test_validate_address_accepts_in_range() -> None:
    issue = validate_address(0x200000, allowed_range=AddressRange(0x10000, 0x300000))
    assert issue is None


def test_validate_address_rejects_out_of_range() -> None:
    issue = validate_address(0x400000, allowed_range=AddressRange(0x10000, 0x300000))
    assert issue is not None
    assert issue.code == "address_out_of_range"


def test_validate_read_size_rejects_non_positive() -> None:
    issue = validate_read_size(0)
    assert issue is not None
    assert issue.code == "invalid_size_value"


def test_validate_numeric_range_rejects_below_minimum() -> None:
    issue = validate_numeric_range(3, min_value=5, field_name="energy")
    assert issue is not None
    assert issue.code == "value_below_minimum"

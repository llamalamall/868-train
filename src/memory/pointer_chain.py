"""Pointer-chain resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.config.offsets import OffsetEntry
from src.memory.reader import ReadFailure, ReadResult, ProcessMemoryReader
from src.memory.validators import validate_address


@dataclass(frozen=True)
class PointerChainFailure:
    """Structured pointer-chain resolution failure."""

    code: str
    message: str
    step_index: int | None = None
    address: int | None = None
    pointer_value: int | None = None
    read_failure: ReadFailure | None = None


@dataclass(frozen=True)
class PointerChainResult:
    """Pointer-chain resolution result."""

    value: int | None
    error: PointerChainFailure | None = None
    traversed_pointers: tuple[int, ...] = ()

    @property
    def is_ok(self) -> bool:
        """Return true when resolution succeeded."""
        return self.error is None

    @classmethod
    def ok(cls, value: int, traversed_pointers: tuple[int, ...]) -> PointerChainResult:
        """Create a successful result."""
        return cls(value=value, error=None, traversed_pointers=traversed_pointers)

    @classmethod
    def fail(
        cls,
        failure: PointerChainFailure,
        traversed_pointers: tuple[int, ...] = (),
    ) -> PointerChainResult:
        """Create a failed result."""
        return cls(value=None, error=failure, traversed_pointers=traversed_pointers)


class ModuleBaseResolver(Protocol):
    """Callable protocol for resolving a module base address."""

    def __call__(self, module_name: str) -> ReadResult[int]:
        """Resolve module base address by module name."""


def resolve_pointer_chain(
    *,
    reader: ProcessMemoryReader,
    base_address: int,
    pointer_chain: Sequence[int],
    final_offset: int = 0,
    max_depth: int = 32,
) -> PointerChainResult:
    """Resolve a multilevel pointer chain to a final address."""
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1.")

    if len(pointer_chain) > max_depth:
        return PointerChainResult.fail(
            PointerChainFailure(
                code="pointer_chain_too_deep",
                message=f"Pointer chain depth {len(pointer_chain)} exceeds max_depth={max_depth}.",
            )
        )

    base_issue = validate_address(base_address)
    if base_issue is not None:
        return PointerChainResult.fail(
            PointerChainFailure(code=base_issue.code, message=base_issue.message, address=base_address)
        )

    cursor = base_address
    traversed: list[int] = []
    for step_index, chain_offset in enumerate(pointer_chain):
        if chain_offset < 0:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="negative_chain_offset",
                    message=f"Pointer chain offset at step {step_index} is negative ({chain_offset}).",
                    step_index=step_index,
                ),
                traversed_pointers=tuple(traversed),
            )

        pointer_address = cursor + chain_offset
        pointer_address_issue = validate_address(pointer_address)
        if pointer_address_issue is not None:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code=pointer_address_issue.code,
                    message=pointer_address_issue.message,
                    step_index=step_index,
                    address=pointer_address,
                ),
                traversed_pointers=tuple(traversed),
            )

        pointer_read = reader.read_pointer(pointer_address)
        if not pointer_read.is_ok:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="pointer_read_failed",
                    message=f"Failed reading pointer at step {step_index} from 0x{pointer_address:X}.",
                    step_index=step_index,
                    address=pointer_address,
                    read_failure=pointer_read.error,
                ),
                traversed_pointers=tuple(traversed),
            )

        next_pointer = pointer_read.value if pointer_read.value is not None else 0
        if next_pointer == 0:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="null_pointer",
                    message=f"Pointer chain hit null pointer at step {step_index}.",
                    step_index=step_index,
                    address=pointer_address,
                    pointer_value=0,
                ),
                traversed_pointers=tuple(traversed),
            )

        pointer_issue = validate_address(next_pointer)
        if pointer_issue is not None:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code=pointer_issue.code,
                    message=pointer_issue.message,
                    step_index=step_index,
                    address=pointer_address,
                    pointer_value=next_pointer,
                ),
                traversed_pointers=tuple(traversed),
            )

        traversed.append(next_pointer)
        cursor = next_pointer

    final_address = cursor + final_offset
    final_issue = validate_address(final_address)
    if final_issue is not None:
        return PointerChainResult.fail(
            PointerChainFailure(
                code=final_issue.code,
                message=final_issue.message,
                address=final_address,
            ),
            traversed_pointers=tuple(traversed),
        )

    return PointerChainResult.ok(value=final_address, traversed_pointers=tuple(traversed))


def resolve_offset_entry_address(
    *,
    reader: ProcessMemoryReader,
    entry: OffsetEntry,
    module_base_resolver: ModuleBaseResolver | None = None,
    max_depth: int = 32,
) -> PointerChainResult:
    """Resolve a final field address from an offset-entry definition."""
    if entry.base.kind == "module":
        if module_base_resolver is None:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="module_base_resolver_missing",
                    message=(
                        f"Cannot resolve module-base entry '{entry.name}' without a module "
                        "base resolver."
                    ),
                )
            )
        base_result = module_base_resolver(entry.base.value)
        if not base_result.is_ok:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="module_base_resolve_failed",
                    message=f"Failed resolving module base '{entry.base.value}' for '{entry.name}'.",
                    read_failure=base_result.error,
                )
            )
        if base_result.value is None:
            return PointerChainResult.fail(
                PointerChainFailure(
                    code="module_base_resolve_failed",
                    message=f"Module base resolver returned empty value for '{entry.base.value}'.",
                )
            )
        base_address = base_result.value
    else:
        base_address = int(entry.base.value, 16)

    return resolve_pointer_chain(
        reader=reader,
        base_address=base_address,
        pointer_chain=entry.pointer_chain,
        final_offset=entry.read_offset,
        max_depth=max_depth,
    )

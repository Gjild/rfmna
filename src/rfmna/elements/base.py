from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Protocol, runtime_checkable


class StampContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class MatrixCoord:
    row: int
    col: int

    def __post_init__(self) -> None:
        _ensure_non_negative_index(self.row, "row")
        _ensure_non_negative_index(self.col, "col")


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    row: int
    col: int
    value: complex

    def __post_init__(self) -> None:
        _ensure_non_negative_index(self.row, "row")
        _ensure_non_negative_index(self.col, "col")
        value = complex(self.value)
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise StampContractError(
                "E_MODEL_STAMP_VALUE_INVALID", "matrix entry value must be finite"
            )
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class RhsEntry:
    row: int
    value: complex

    def __post_init__(self) -> None:
        _ensure_non_negative_index(self.row, "row")
        value = complex(self.value)
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise StampContractError(
                "E_MODEL_STAMP_VALUE_INVALID", "rhs entry value must be finite"
            )
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise StampContractError(
                "E_MODEL_VALIDATION_ISSUE_INVALID", "validation code must be non-empty"
            )
        if not self.message:
            raise StampContractError(
                "E_MODEL_VALIDATION_ISSUE_INVALID", "validation message must be non-empty"
            )
        if self.context is not None:
            copied = {key: self.context[key] for key in sorted(self.context)}
            object.__setattr__(self, "context", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class StampContext:
    omega_rad_s: float
    resolved_params: Mapping[str, float]
    sweep_index: int | None = None
    frequency_index: int | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.omega_rad_s):
            raise StampContractError("E_MODEL_STAMP_CONTEXT_INVALID", "omega_rad_s must be finite")
        if self.sweep_index is not None:
            _ensure_non_negative_index(self.sweep_index, "sweep_index")
        if self.frequency_index is not None:
            _ensure_non_negative_index(self.frequency_index, "frequency_index")

        copied: dict[str, float] = {}
        for key in sorted(self.resolved_params):
            if not key:
                raise StampContractError(
                    "E_MODEL_STAMP_CONTEXT_INVALID", "resolved parameter keys must be non-empty"
                )
            value = float(self.resolved_params[key])
            if not isfinite(value):
                raise StampContractError(
                    "E_MODEL_STAMP_CONTEXT_INVALID", f"resolved parameter '{key}' must be finite"
                )
            copied[key] = value
        object.__setattr__(self, "resolved_params", MappingProxyType(copied))


@runtime_checkable
class ElementStamp(Protocol):
    element_id: str

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]: ...

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]: ...

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]: ...

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]: ...

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]: ...


def _ensure_non_negative_index(value: int, field_name: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise StampContractError(
            "E_MODEL_STAMP_INDEX_INVALID", f"{field_name} must be a non-negative integer"
        )


def canonicalize_indices(indices: Sequence[int]) -> tuple[int, ...]:
    ordered = sorted(set(indices))
    for value in ordered:
        _ensure_non_negative_index(value, "index")
    return tuple(ordered)


def canonicalize_coords(coords: Sequence[MatrixCoord]) -> tuple[MatrixCoord, ...]:
    ordered = sorted(coords, key=lambda coord: (coord.row, coord.col))
    canonical: list[MatrixCoord] = []
    previous_key: tuple[int, int] | None = None
    for coord in ordered:
        key = (coord.row, coord.col)
        if key == previous_key:
            continue
        canonical.append(coord)
        previous_key = key
    return tuple(canonical)


def canonicalize_matrix_entries(entries: Sequence[MatrixEntry]) -> tuple[MatrixEntry, ...]:
    ordered = sorted(entries, key=lambda entry: (entry.row, entry.col))
    return _aggregate_matrix_entries(ordered)


def _aggregate_matrix_entries(entries: Sequence[MatrixEntry]) -> tuple[MatrixEntry, ...]:
    canonical: list[MatrixEntry] = []
    key: tuple[int, int] | None = None
    value = 0.0j
    for entry in entries:
        current = (entry.row, entry.col)
        if key is None:
            key = current
            value = entry.value
            continue
        if current == key:
            value += entry.value
            continue
        canonical.append(MatrixEntry(row=key[0], col=key[1], value=value))
        key = current
        value = entry.value
    if key is not None:
        canonical.append(MatrixEntry(row=key[0], col=key[1], value=value))
    return tuple(canonical)


def canonicalize_rhs_entries(entries: Sequence[RhsEntry]) -> tuple[RhsEntry, ...]:
    ordered = sorted(entries, key=lambda entry: entry.row)
    canonical: list[RhsEntry] = []
    current_row: int | None = None
    current_value = 0.0j
    for entry in ordered:
        if current_row is None:
            current_row = entry.row
            current_value = entry.value
            continue
        if entry.row == current_row:
            current_value += entry.value
            continue
        canonical.append(RhsEntry(row=current_row, value=current_value))
        current_row = entry.row
        current_value = entry.value
    if current_row is not None:
        canonical.append(RhsEntry(row=current_row, value=current_value))
    return tuple(canonical)

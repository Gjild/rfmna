from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rfmna.elements.base import ElementStamp, MatrixCoord, StampContext


class AssemblerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class CompiledPattern:
    n_unknowns: int
    shape: tuple[int, int]
    coords: tuple[MatrixCoord, ...]
    rhs_rows: tuple[int, ...]
    coord_to_slot: Mapping[tuple[int, int], int]
    rhs_row_to_slot: Mapping[int, int]

    def __post_init__(self) -> None:
        _validate_n_unknowns_and_shape(self.n_unknowns, self.shape)
        _validate_coords(self.coords, self.n_unknowns)
        _validate_rhs_rows(self.rhs_rows, self.n_unknowns)
        _validate_slot_maps(
            self.coords,
            self.rhs_rows,
            self.coord_to_slot,
            self.rhs_row_to_slot,
        )


def _validate_n_unknowns_and_shape(n_unknowns: int, shape: tuple[int, int]) -> None:
    if n_unknowns < 0:
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "n_unknowns must be >= 0")
    if shape != (n_unknowns, n_unknowns):
        raise AssemblerError(
            "E_ASSEMBLER_PATTERN_INVALID", "shape must match (n_unknowns, n_unknowns)"
        )


def _validate_coords(coords: tuple[MatrixCoord, ...], n_unknowns: int) -> None:
    coord_keys = [(coord.row, coord.col) for coord in coords]
    if coord_keys != sorted(coord_keys):
        raise AssemblerError(
            "E_ASSEMBLER_PATTERN_INVALID", "coords must be lexicographically sorted"
        )
    if len(coord_keys) != len(set(coord_keys)):
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "coords must be unique")
    for row, col in coord_keys:
        if row < 0 or row >= n_unknowns or col < 0 or col >= n_unknowns:
            raise AssemblerError("E_ASSEMBLER_INDEX_OUT_OF_RANGE", "matrix coordinate out of range")


def _validate_rhs_rows(rhs_rows: tuple[int, ...], n_unknowns: int) -> None:
    if rhs_rows != tuple(sorted(set(rhs_rows))):
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "rhs_rows must be sorted unique")
    for row in rhs_rows:
        if row < 0 or row >= n_unknowns:
            raise AssemblerError("E_ASSEMBLER_INDEX_OUT_OF_RANGE", "rhs row out of range")


def _validate_slot_maps(
    coords: tuple[MatrixCoord, ...],
    rhs_rows: tuple[int, ...],
    coord_to_slot: Mapping[tuple[int, int], int],
    rhs_row_to_slot: Mapping[int, int],
) -> None:
    if len(coord_to_slot) != len(coords):
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "coord_to_slot size mismatch")
    if len(rhs_row_to_slot) != len(rhs_rows):
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "rhs_row_to_slot size mismatch")
    for slot, coord in enumerate(coords):
        if coord_to_slot.get((coord.row, coord.col)) != slot:
            raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "coord_to_slot mapping mismatch")
    for slot, row in enumerate(rhs_rows):
        if rhs_row_to_slot.get(row) != slot:
            raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "rhs_row_to_slot mapping mismatch")


def _canonical_structural_context(ctx: StampContext) -> StampContext:
    return StampContext(
        omega_rad_s=1.0,
        resolved_params=ctx.resolved_params,
        sweep_index=ctx.sweep_index,
        frequency_index=ctx.frequency_index,
    )


def _validate_coord_range(coord: MatrixCoord, n_unknowns: int) -> None:
    if coord.row < 0 or coord.row >= n_unknowns or coord.col < 0 or coord.col >= n_unknowns:
        raise AssemblerError(
            "E_ASSEMBLER_INDEX_OUT_OF_RANGE",
            f"matrix coordinate out of range: ({coord.row}, {coord.col})",
        )


def _validate_rhs_row_range(row: int, n_unknowns: int) -> None:
    if row < 0 or row >= n_unknowns:
        raise AssemblerError("E_ASSEMBLER_INDEX_OUT_OF_RANGE", f"rhs row out of range: {row}")


def compile_pattern(
    n_unknowns: int,
    elements: Sequence[ElementStamp],
    structural_ctx: StampContext,
) -> CompiledPattern:
    if n_unknowns < 0:
        raise AssemblerError("E_ASSEMBLER_PATTERN_INVALID", "n_unknowns must be >= 0")

    ctx = _canonical_structural_context(structural_ctx)
    coords_seen: set[tuple[int, int]] = set()
    rhs_rows_seen: set[int] = set()

    for element in elements:
        for coord in element.footprint(ctx):
            _validate_coord_range(coord, n_unknowns)
            coords_seen.add((coord.row, coord.col))
        for rhs_entry in element.stamp_b(ctx):
            _validate_rhs_row_range(rhs_entry.row, n_unknowns)
            rhs_rows_seen.add(rhs_entry.row)

    sorted_coords = tuple(MatrixCoord(row=row, col=col) for row, col in sorted(coords_seen))
    sorted_rhs_rows = tuple(sorted(rhs_rows_seen))
    coord_to_slot = MappingProxyType(
        {(coord.row, coord.col): slot for slot, coord in enumerate(sorted_coords)}
    )
    rhs_row_to_slot = MappingProxyType({row: slot for slot, row in enumerate(sorted_rhs_rows)})

    return CompiledPattern(
        n_unknowns=n_unknowns,
        shape=(n_unknowns, n_unknowns),
        coords=sorted_coords,
        rhs_rows=sorted_rhs_rows,
        coord_to_slot=coord_to_slot,
        rhs_row_to_slot=rhs_row_to_slot,
    )


def pattern_projection(
    pattern: CompiledPattern,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    return (
        tuple((coord.row, coord.col) for coord in pattern.coords),
        pattern.rhs_rows,
    )

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]

from rfmna.elements.base import ElementStamp, StampContext

from .pattern import AssemblerError, CompiledPattern


@dataclass(frozen=True, slots=True)
class FilledSystem:
    pattern: CompiledPattern
    A: csr_matrix
    b: NDArray[np.complex128]
    slot_values: tuple[complex, ...]


def fill_numeric(
    compiled_pattern: CompiledPattern,
    elements: Sequence[ElementStamp],
    ctx: StampContext,
) -> FilledSystem:
    n_slots = len(compiled_pattern.coords)
    slot_values = np.zeros(n_slots, dtype=np.complex128)
    b = np.zeros(compiled_pattern.n_unknowns, dtype=np.complex128)

    for element in elements:
        for entry in element.stamp_A(ctx):
            key = (entry.row, entry.col)
            slot = compiled_pattern.coord_to_slot.get(key)
            if slot is None:
                raise AssemblerError(
                    "E_ASSEMBLER_FILL_COORD_NOT_COMPILED",
                    f"matrix coordinate not in compiled pattern: ({entry.row}, {entry.col})",
                )
            slot_values[slot] += np.complex128(entry.value)

        for rhs_entry in element.stamp_b(ctx):
            slot = compiled_pattern.rhs_row_to_slot.get(rhs_entry.row)
            if slot is None:
                raise AssemblerError(
                    "E_ASSEMBLER_FILL_RHS_NOT_COMPILED",
                    f"rhs row not in compiled pattern: {rhs_entry.row}",
                )
            b[rhs_entry.row] += np.complex128(rhs_entry.value)

    rows = np.fromiter(
        (coord.row for coord in compiled_pattern.coords), dtype=np.int64, count=n_slots
    )
    cols = np.fromiter(
        (coord.col for coord in compiled_pattern.coords), dtype=np.int64, count=n_slots
    )
    A = coo_matrix(
        (slot_values, (rows, cols)), shape=compiled_pattern.shape, dtype=np.complex128
    ).tocsr()

    return FilledSystem(
        pattern=compiled_pattern,
        A=A,
        b=b,
        slot_values=tuple(complex(value) for value in slot_values),
    )


def filled_projection(
    filled: FilledSystem,
) -> tuple[tuple[tuple[int, int, complex], ...], tuple[complex, ...]]:
    coords_values = tuple(
        (coord.row, coord.col, filled.slot_values[slot])
        for slot, coord in enumerate(filled.pattern.coords)
    )
    rhs = tuple(complex(value) for value in filled.b)
    return coords_values, rhs

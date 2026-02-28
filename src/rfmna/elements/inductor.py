from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from .base import (
    ElementStamp,
    MatrixCoord,
    MatrixEntry,
    RhsEntry,
    StampContext,
    StampContractError,
    ValidationIssue,
    canonicalize_coords,
    canonicalize_indices,
    canonicalize_matrix_entries,
)


class InductorIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...

    def aux_index(self, aux_id: str) -> int: ...


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise StampContractError("E_MODEL_STAMP_ELEMENT_INVALID", f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class InductorStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    aux_id: str
    inductance_h: float
    indexer: InductorIndexer

    def __post_init__(self) -> None:
        _validate_identifier("element_id", self.element_id)
        _validate_identifier("node_a", self.node_a)
        _validate_identifier("node_b", self.node_b)
        _validate_identifier("aux_id", self.aux_id)

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        index_k = self.indexer.aux_index(self.aux_id)
        indices = (
            index_k,
            self.indexer.node_index(self.node_a),
            self.indexer.node_index(self.node_b),
        )
        return canonicalize_indices(tuple(index for index in indices if index is not None))

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
        entries = self.stamp_A(ctx)
        return canonicalize_coords(
            tuple(MatrixCoord(row=entry.row, col=entry.col) for entry in entries)
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        index_k = self.indexer.aux_index(self.aux_id)
        if self.validate(ctx):
            return ()

        index_a = self.indexer.node_index(self.node_a)
        index_b = self.indexer.node_index(self.node_b)
        entries: list[MatrixEntry] = []
        if index_a is not None:
            entries.append(MatrixEntry(row=index_a, col=index_k, value=1.0 + 0.0j))
            entries.append(MatrixEntry(row=index_k, col=index_a, value=1.0 + 0.0j))
        if index_b is not None:
            entries.append(MatrixEntry(row=index_b, col=index_k, value=-1.0 + 0.0j))
            entries.append(MatrixEntry(row=index_k, col=index_b, value=-1.0 + 0.0j))

        entries.append(
            MatrixEntry(
                row=index_k,
                col=index_k,
                value=-1j * complex(ctx.omega_rad_s * self.inductance_h),
            )
        )
        canonical = canonicalize_matrix_entries(entries)
        return tuple(entry for entry in canonical if entry.value != 0.0 + 0.0j)

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.inductance_h) or self.inductance_h <= 0.0:
            return (
                ValidationIssue(
                    code="E_MODEL_L_NONPOSITIVE",
                    message="inductance_h must be finite and > 0",
                    context={"element_id": self.element_id, "inductance_h": self.inductance_h},
                ),
            )
        return ()

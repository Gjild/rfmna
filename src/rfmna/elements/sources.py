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
    canonicalize_rhs_entries,
)


class CurrentSourceIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...


class VoltageSourceIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...

    def aux_index(self, aux_id: str) -> int: ...


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise StampContractError("E_MODEL_STAMP_ELEMENT_INVALID", f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class CurrentSourceStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    current_a: float
    node_indexer: CurrentSourceIndexer

    def __post_init__(self) -> None:
        _validate_identifier("element_id", self.element_id)
        _validate_identifier("node_a", self.node_a)
        _validate_identifier("node_b", self.node_b)

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        indices = (
            self.node_indexer.node_index(self.node_a),
            self.node_indexer.node_index(self.node_b),
        )
        return canonicalize_indices(tuple(index for index in indices if index is not None))

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
        del ctx
        return ()

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        del ctx
        return ()

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()
        index_a = self.node_indexer.node_index(self.node_a)
        index_b = self.node_indexer.node_index(self.node_b)
        entries: list[RhsEntry] = []
        if index_a is not None:
            entries.append(RhsEntry(row=index_a, value=-complex(self.current_a)))
        if index_b is not None:
            entries.append(RhsEntry(row=index_b, value=complex(self.current_a)))
        return canonicalize_rhs_entries(entries)

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.current_a):
            return (
                ValidationIssue(
                    code="E_MODEL_ISRC_INVALID",
                    message="current_a must be finite",
                    context={"current_a": self.current_a, "element_id": self.element_id},
                ),
            )
        return ()


@dataclass(frozen=True, slots=True)
class VoltageSourceStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    aux_id: str
    voltage_v: float
    indexer: VoltageSourceIndexer

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
        del ctx
        index_k = self.indexer.aux_index(self.aux_id)
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
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
        return canonicalize_matrix_entries(entries)

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()
        index_k = self.indexer.aux_index(self.aux_id)
        return canonicalize_rhs_entries((RhsEntry(row=index_k, value=complex(self.voltage_v)),))

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.voltage_v):
            return (
                ValidationIssue(
                    code="E_MODEL_VSRC_INVALID",
                    message="voltage_v must be finite",
                    context={"element_id": self.element_id, "voltage_v": self.voltage_v},
                ),
            )
        return ()

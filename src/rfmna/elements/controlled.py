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


class VCCSIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...


class VCVSIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...

    def aux_index(self, aux_id: str) -> int: ...


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise StampContractError("E_MODEL_STAMP_ELEMENT_INVALID", f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class VCCSStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    node_c: str
    node_d: str
    transconductance_s: float
    node_indexer: VCCSIndexer

    def __post_init__(self) -> None:
        _validate_identifier("element_id", self.element_id)
        _validate_identifier("node_a", self.node_a)
        _validate_identifier("node_b", self.node_b)
        _validate_identifier("node_c", self.node_c)
        _validate_identifier("node_d", self.node_d)

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        index_a = self.node_indexer.node_index(self.node_a)
        index_b = self.node_indexer.node_index(self.node_b)
        if index_a is None and index_b is None:
            return ()

        indices = (
            index_a,
            index_b,
            self.node_indexer.node_index(self.node_c),
            self.node_indexer.node_index(self.node_d),
        )
        return canonicalize_indices(tuple(index for index in indices if index is not None))

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
        entries = self.stamp_A(ctx)
        return canonicalize_coords(
            tuple(MatrixCoord(row=entry.row, col=entry.col) for entry in entries)
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        del ctx
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()

        index_a = self.node_indexer.node_index(self.node_a)
        index_b = self.node_indexer.node_index(self.node_b)
        index_c = self.node_indexer.node_index(self.node_c)
        index_d = self.node_indexer.node_index(self.node_d)
        gm = complex(self.transconductance_s)

        entries: list[MatrixEntry] = []
        if index_a is not None and index_c is not None:
            entries.append(MatrixEntry(row=index_a, col=index_c, value=gm))
        if index_a is not None and index_d is not None:
            entries.append(MatrixEntry(row=index_a, col=index_d, value=-gm))
        if index_b is not None and index_c is not None:
            entries.append(MatrixEntry(row=index_b, col=index_c, value=-gm))
        if index_b is not None and index_d is not None:
            entries.append(MatrixEntry(row=index_b, col=index_d, value=gm))

        canonical = canonicalize_matrix_entries(entries)
        return tuple(entry for entry in canonical if entry.value != 0.0 + 0.0j)

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.transconductance_s):
            return (
                ValidationIssue(
                    code="E_MODEL_VCCS_INVALID",
                    message="transconductance_s must be finite",
                    context={
                        "element_id": self.element_id,
                        "transconductance_s": self.transconductance_s,
                    },
                ),
            )
        return ()


@dataclass(frozen=True, slots=True)
class VCVSStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    node_c: str
    node_d: str
    aux_id: str
    gain_mu: float
    indexer: VCVSIndexer

    def __post_init__(self) -> None:
        _validate_identifier("element_id", self.element_id)
        _validate_identifier("node_a", self.node_a)
        _validate_identifier("node_b", self.node_b)
        _validate_identifier("node_c", self.node_c)
        _validate_identifier("node_d", self.node_d)
        _validate_identifier("aux_id", self.aux_id)

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        index_k = self.indexer.aux_index(self.aux_id)
        indices = (
            index_k,
            self.indexer.node_index(self.node_a),
            self.indexer.node_index(self.node_b),
            self.indexer.node_index(self.node_c),
            self.indexer.node_index(self.node_d),
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
        index_c = self.indexer.node_index(self.node_c)
        index_d = self.indexer.node_index(self.node_d)
        mu = complex(self.gain_mu)

        entries: list[MatrixEntry] = []
        if index_a is not None:
            entries.append(MatrixEntry(row=index_a, col=index_k, value=1.0 + 0.0j))
            entries.append(MatrixEntry(row=index_k, col=index_a, value=1.0 + 0.0j))
        if index_b is not None:
            entries.append(MatrixEntry(row=index_b, col=index_k, value=-1.0 + 0.0j))
            entries.append(MatrixEntry(row=index_k, col=index_b, value=-1.0 + 0.0j))
        if index_c is not None:
            entries.append(MatrixEntry(row=index_k, col=index_c, value=-mu))
        if index_d is not None:
            entries.append(MatrixEntry(row=index_k, col=index_d, value=mu))

        canonical = canonicalize_matrix_entries(entries)
        return tuple(entry for entry in canonical if entry.value != 0.0 + 0.0j)

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.gain_mu):
            return (
                ValidationIssue(
                    code="E_MODEL_VCVS_INVALID",
                    message="gain_mu must be finite",
                    context={"element_id": self.element_id, "gain_mu": self.gain_mu},
                ),
            )
        return ()

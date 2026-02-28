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


class NodeIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise StampContractError("E_MODEL_STAMP_ELEMENT_INVALID", f"{field_name} must be non-empty")


def _is_zero_admittance(value: complex) -> bool:
    return value == 0.0 + 0.0j


def _two_terminal_entries(
    *,
    node_indexer: NodeIndexer,
    node_a: str,
    node_b: str,
    admittance: complex,
) -> tuple[MatrixEntry, ...]:
    if _is_zero_admittance(admittance):
        return ()

    index_a = node_indexer.node_index(node_a)
    index_b = node_indexer.node_index(node_b)
    entries: list[MatrixEntry] = []
    if index_a is not None:
        entries.append(MatrixEntry(row=index_a, col=index_a, value=admittance))
    if index_b is not None:
        entries.append(MatrixEntry(row=index_b, col=index_b, value=admittance))
    if index_a is not None and index_b is not None:
        entries.append(MatrixEntry(row=index_a, col=index_b, value=-admittance))
        entries.append(MatrixEntry(row=index_b, col=index_a, value=-admittance))

    canonical = canonicalize_matrix_entries(entries)
    return tuple(entry for entry in canonical if entry.value != 0.0 + 0.0j)


def _two_terminal_footprint(
    *,
    node_indexer: NodeIndexer,
    node_a: str,
    node_b: str,
    admittance: complex,
) -> tuple[MatrixCoord, ...]:
    entries = _two_terminal_entries(
        node_indexer=node_indexer,
        node_a=node_a,
        node_b=node_b,
        admittance=admittance,
    )
    return canonicalize_coords(
        tuple(MatrixCoord(row=entry.row, col=entry.col) for entry in entries)
    )


@dataclass(frozen=True, slots=True)
class ResistorStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    resistance_ohm: float
    node_indexer: NodeIndexer

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
        validation = self.validate(StampContext(omega_rad_s=0.0, resolved_params={}))
        if validation:
            return ()
        admittance = complex(1.0 / self.resistance_ohm)
        return _two_terminal_footprint(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        del ctx
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()
        admittance = complex(1.0 / self.resistance_ohm)
        return _two_terminal_entries(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.resistance_ohm) or self.resistance_ohm <= 0.0:
            return (
                ValidationIssue(
                    code="E_MODEL_R_NONPOSITIVE",
                    message="resistance_ohm must be > 0",
                    context={"element_id": self.element_id, "resistance_ohm": self.resistance_ohm},
                ),
            )
        return ()


@dataclass(frozen=True, slots=True)
class CapacitorStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    capacitance_f: float
    node_indexer: NodeIndexer

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
        if self.validate(ctx):
            return ()
        admittance = 1j * complex(ctx.omega_rad_s * self.capacitance_f)
        return _two_terminal_footprint(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        if self.validate(ctx):
            return ()
        admittance = 1j * complex(ctx.omega_rad_s * self.capacitance_f)
        return _two_terminal_entries(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.capacitance_f) or self.capacitance_f < 0.0:
            return (
                ValidationIssue(
                    code="E_MODEL_C_NEGATIVE",
                    message="capacitance_f must be >= 0",
                    context={"capacitance_f": self.capacitance_f, "element_id": self.element_id},
                ),
            )
        return ()


@dataclass(frozen=True, slots=True)
class ConductanceStamp(ElementStamp):
    element_id: str
    node_a: str
    node_b: str
    conductance_s: float
    node_indexer: NodeIndexer

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
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()
        admittance = complex(self.conductance_s)
        return _two_terminal_footprint(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        del ctx
        if self.validate(StampContext(omega_rad_s=0.0, resolved_params={})):
            return ()
        admittance = complex(self.conductance_s)
        return _two_terminal_entries(
            node_indexer=self.node_indexer,
            node_a=self.node_a,
            node_b=self.node_b,
            admittance=admittance,
        )

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return ()

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        if not isfinite(self.conductance_s) or self.conductance_s < 0.0:
            return (
                ValidationIssue(
                    code="E_MODEL_G_NEGATIVE",
                    message="conductance_s must be >= 0",
                    context={"conductance_s": self.conductance_s, "element_id": self.element_id},
                ),
            )
        return ()

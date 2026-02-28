from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import (  # type: ignore[import-untyped]
    coo_matrix,
    csc_matrix,
    hstack,
    isspmatrix,
    vstack,
)

from rfmna.diagnostics import (
    DiagnosticEvent,
    PortContext,
    Severity,
    SolverStage,
    sort_diagnostics,
)
from rfmna.solver.backend import SparseComplexMatrix

type BoundaryKind = Literal["voltage", "current"]
type BoundarySource = Literal["imposed", "inactive"]

_RF_BOUNDARY_ELEMENT_ID = "rf_boundary"
_BOUNDARY_VALUE_ABS_TOL = 1e-12
_SOURCE_RANK: dict[BoundarySource, int] = {"imposed": 0, "inactive": 1}
_KIND_RANK: dict[BoundaryKind, int] = {"voltage": 0, "current": 1}
_AUX_CURRENT_SIGN_INTO_DUT = -1.0


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _validate_unknown_index(field_name: str, value: int | None) -> None:
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0 when provided")


def _is_finite_complex(value: complex) -> bool:
    return isfinite(value.real) and isfinite(value.imag)


def _complex_sort_key(value: complex) -> tuple[float, float]:
    return (float(value.real), float(value.imag))


def _terminal_sort_key(index: int | None) -> int:
    if index is None:
        return -1
    return index


def _terminal_token(index: int | None) -> str:
    if index is None:
        return "gnd"
    return str(index)


def _complex_witness(value: complex) -> dict[str, float]:
    return {
        "imag": float(value.imag),
        "real": float(value.real),
    }


def _complex_equal(
    left: complex, right: complex, *, abs_tol: float = _BOUNDARY_VALUE_ABS_TOL
) -> bool:
    return bool(abs(left - right) <= abs_tol)


@dataclass(frozen=True, slots=True)
class PortBoundary:
    port_id: str
    p_plus_index: int | None
    p_minus_index: int | None

    def __post_init__(self) -> None:
        _validate_identifier("port_id", self.port_id)
        _validate_unknown_index("p_plus_index", self.p_plus_index)
        _validate_unknown_index("p_minus_index", self.p_minus_index)


@dataclass(frozen=True, slots=True)
class BoundaryRequest:
    port_id: str
    kind: BoundaryKind
    value: complex
    source: BoundarySource = "imposed"

    def __post_init__(self) -> None:
        _validate_identifier("port_id", self.port_id)
        if self.kind not in _KIND_RANK:
            raise ValueError("kind must be one of: voltage,current")
        if self.source not in _SOURCE_RANK:
            raise ValueError("source must be one of: imposed,inactive")
        complex_value = complex(self.value)
        if not _is_finite_complex(complex_value):
            raise ValueError("boundary value must be finite")
        object.__setattr__(self, "value", complex_value)


@dataclass(frozen=True, slots=True)
class AppliedBoundary:
    port_id: str
    kind: BoundaryKind
    source: BoundarySource
    value: complex
    p_plus_index: int | None
    p_minus_index: int | None
    equation_row_index: int | None = None
    aux_unknown_index: int | None = None
    current_into_dut_from_aux_sign: float | None = None

    def __post_init__(self) -> None:
        _validate_identifier("port_id", self.port_id)
        _validate_unknown_index("p_plus_index", self.p_plus_index)
        _validate_unknown_index("p_minus_index", self.p_minus_index)
        _validate_unknown_index("equation_row_index", self.equation_row_index)
        _validate_unknown_index("aux_unknown_index", self.aux_unknown_index)
        if self.current_into_dut_from_aux_sign is not None and self.kind != "voltage":
            raise ValueError("current_into_dut_from_aux_sign is only valid for voltage boundaries")


@dataclass(frozen=True, slots=True)
class BoundaryMetadata:
    canonical_port_ids: tuple[str, ...]
    requested: tuple[BoundaryRequest, ...]
    applied: tuple[AppliedBoundary, ...]

    def __post_init__(self) -> None:
        canonical_port_ids = tuple(sorted(self.canonical_port_ids))
        requested = tuple(sorted(self.requested, key=_request_sort_key))
        applied = tuple(sorted(self.applied, key=_applied_sort_key))
        object.__setattr__(self, "canonical_port_ids", canonical_port_ids)
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "applied", applied)


@dataclass(frozen=True, slots=True)
class BoundaryInjectionResult:
    matrix: SparseComplexMatrix | None
    rhs: NDArray[np.complex128] | None
    metadata: BoundaryMetadata
    diagnostics: tuple[DiagnosticEvent, ...]

    def __post_init__(self) -> None:
        if self.matrix is not None and not isspmatrix(self.matrix):
            raise ValueError("matrix must be sparse when provided")
        if self.rhs is not None:
            vector = np.asarray(self.rhs, dtype=np.complex128)
            if vector.ndim != 1:
                raise ValueError("rhs must be rank-1 when provided")
            object.__setattr__(self, "rhs", vector)
        object.__setattr__(self, "diagnostics", tuple(sort_diagnostics(self.diagnostics)))


def apply_voltage_boundaries(  # noqa: PLR0913
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    ports: Sequence[PortBoundary],
    *,
    imposed_port_voltages: Sequence[tuple[str, complex]],
    inactive_port_ids: Sequence[str] = (),
    inactive_voltage: complex = 0.0 + 0.0j,
) -> BoundaryInjectionResult:
    imposed = tuple(
        BoundaryRequest(port_id=port_id, kind="voltage", value=value, source="imposed")
        for port_id, value in imposed_port_voltages
    )
    inactive = tuple(
        BoundaryRequest(port_id=port_id, kind="voltage", value=inactive_voltage, source="inactive")
        for port_id in inactive_port_ids
    )
    return apply_boundary_conditions(A, b, ports, imposed=imposed, inactive=inactive)


def apply_current_boundaries(  # noqa: PLR0913
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    ports: Sequence[PortBoundary],
    *,
    imposed_port_currents: Sequence[tuple[str, complex]],
    inactive_port_ids: Sequence[str] = (),
    inactive_current: complex = 0.0 + 0.0j,
) -> BoundaryInjectionResult:
    imposed = tuple(
        BoundaryRequest(port_id=port_id, kind="current", value=value, source="imposed")
        for port_id, value in imposed_port_currents
    )
    inactive = tuple(
        BoundaryRequest(port_id=port_id, kind="current", value=inactive_current, source="inactive")
        for port_id in inactive_port_ids
    )
    return apply_boundary_conditions(A, b, ports, imposed=imposed, inactive=inactive)


def apply_boundary_conditions(
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    ports: Sequence[PortBoundary],
    *,
    imposed: Sequence[BoundaryRequest] = (),
    inactive: Sequence[BoundaryRequest] = (),
) -> BoundaryInjectionResult:
    if not isspmatrix(A):
        raise TypeError("matrix A must be sparse")
    matrix_csc = A if isinstance(A, csc_matrix) else A.tocsc()
    if matrix_csc.shape[0] != matrix_csc.shape[1]:
        raise ValueError("matrix A must be square")

    rhs = np.asarray(b, dtype=np.complex128)
    if rhs.ndim != 1:
        raise ValueError("rhs b must be rank-1")
    if rhs.shape[0] != matrix_csc.shape[0]:
        raise ValueError("rhs b length must equal matrix dimension")

    requested = tuple(
        sorted(
            (
                *(
                    BoundaryRequest(
                        port_id=req.port_id,
                        kind=req.kind,
                        value=req.value,
                        source="imposed",
                    )
                    for req in imposed
                ),
                *(
                    BoundaryRequest(
                        port_id=req.port_id,
                        kind=req.kind,
                        value=req.value,
                        source="inactive",
                    )
                    for req in inactive
                ),
            ),
            key=_request_sort_key,
        )
    )
    metadata_base = BoundaryMetadata(
        canonical_port_ids=tuple(
            port.port_id for port in sorted(ports, key=lambda item: item.port_id)
        ),
        requested=requested,
        applied=(),
    )

    diagnostics: list[DiagnosticEvent] = []
    port_map = _canonical_port_map(
        ports=ports, n_unknowns=matrix_csc.shape[0], diagnostics=diagnostics
    )
    resolved = _resolve_requests(requested=requested, port_map=port_map, diagnostics=diagnostics)

    voltage_resolved = tuple(item for item in resolved if item.kind == "voltage")
    _check_voltage_constraint_collisions(
        resolved=voltage_resolved,
        port_map=port_map,
        diagnostics=diagnostics,
    )

    if diagnostics:
        return BoundaryInjectionResult(
            matrix=None,
            rhs=None,
            metadata=metadata_base,
            diagnostics=tuple(diagnostics),
        )

    rhs_aug = rhs.copy()
    applied_current = _apply_current_boundaries(
        rhs=rhs_aug,
        resolved=tuple(item for item in resolved if item.kind == "current"),
        port_map=port_map,
    )
    matrix_aug, rhs_final, applied_voltage = _apply_voltage_boundaries(
        matrix=matrix_csc,
        rhs=rhs_aug,
        resolved=voltage_resolved,
        port_map=port_map,
    )

    metadata = BoundaryMetadata(
        canonical_port_ids=tuple(port_map),
        requested=requested,
        applied=tuple((*applied_current, *applied_voltage)),
    )
    return BoundaryInjectionResult(
        matrix=matrix_aug,
        rhs=rhs_final,
        metadata=metadata,
        diagnostics=(),
    )


def _request_sort_key(request: BoundaryRequest) -> tuple[str, int, int, tuple[float, float]]:
    return (
        request.port_id,
        _KIND_RANK[request.kind],
        _SOURCE_RANK[request.source],
        _complex_sort_key(request.value),
    )


def _applied_sort_key(
    applied: AppliedBoundary,
) -> tuple[str, int, int, tuple[float, float], int, int]:
    eq = applied.equation_row_index if applied.equation_row_index is not None else -1
    aux = applied.aux_unknown_index if applied.aux_unknown_index is not None else -1
    return (
        applied.port_id,
        _KIND_RANK[applied.kind],
        _SOURCE_RANK[applied.source],
        _complex_sort_key(applied.value),
        eq,
        aux,
    )


def _canonical_port_map(
    *,
    ports: Sequence[PortBoundary],
    n_unknowns: int,
    diagnostics: list[DiagnosticEvent],
) -> dict[str, PortBoundary]:
    port_map: dict[str, PortBoundary] = {}
    for port in sorted(ports, key=lambda item: item.port_id):
        if port.port_id in port_map:
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"duplicate boundary port declaration for '{port.port_id}'",
                    suggested_action="provide each boundary port_id once",
                    port_id=port.port_id,
                    witness={
                        "issue": "duplicate_port_id",
                        "port_id": port.port_id,
                    },
                )
            )
            continue
        for field_name, index in (
            ("p_plus_index", port.p_plus_index),
            ("p_minus_index", port.p_minus_index),
        ):
            if index is not None and index >= n_unknowns:
                diagnostics.append(
                    _boundary_error(
                        code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                        message=f"port '{port.port_id}' has out-of-range {field_name}",
                        suggested_action="ensure boundary port node indices are within matrix unknown range",
                        port_id=port.port_id,
                        witness={
                            "field": field_name,
                            "index": index,
                            "issue": "index_out_of_range",
                            "n_unknowns": n_unknowns,
                        },
                    )
                )
        if port.p_plus_index == port.p_minus_index:
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"port '{port.port_id}' has degenerate terminal declaration",
                    suggested_action="declare distinct p_plus and p_minus indices",
                    port_id=port.port_id,
                    witness={
                        "issue": "degenerate_port",
                        "p_minus_index": port.p_minus_index,
                        "p_plus_index": port.p_plus_index,
                    },
                )
            )
        port_map[port.port_id] = port
    return port_map


def _resolve_requests(
    *,
    requested: tuple[BoundaryRequest, ...],
    port_map: dict[str, PortBoundary],
    diagnostics: list[DiagnosticEvent],
) -> tuple[BoundaryRequest, ...]:
    grouped: dict[tuple[str, BoundaryKind], list[BoundaryRequest]] = defaultdict(list)
    for request in requested:
        grouped[(request.port_id, request.kind)].append(request)

    kinds_by_port: dict[str, set[BoundaryKind]] = defaultdict(set)
    for port_id, kind in grouped:
        kinds_by_port[port_id].add(kind)

    for port_id in sorted(kinds_by_port):
        if len(kinds_by_port[port_id]) > 1:
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"port '{port_id}' mixes voltage and current boundary declarations",
                    suggested_action="declare exactly one boundary kind per port for a solve pass",
                    port_id=port_id if port_id in port_map else None,
                    witness={
                        "issue": "mixed_boundary_kind",
                        "kinds": sorted(kinds_by_port[port_id]),
                        "port_id": port_id,
                    },
                )
            )

    resolved: list[BoundaryRequest] = []
    for key in sorted(grouped):
        port_id, kind = key
        entries = tuple(sorted(grouped[key], key=_request_sort_key))
        if port_id not in port_map:
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"unknown boundary port_id '{port_id}'",
                    suggested_action="declare boundary requests only for known ports",
                    witness={
                        "issue": "unknown_port_id",
                        "known_port_ids": sorted(port_map),
                        "port_id": port_id,
                    },
                )
            )
            continue

        values = [entry.value for entry in entries]
        all_equal = all(_complex_equal(value, values[0]) for value in values[1:])
        if len(entries) > 1 and not all_equal:
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"port '{port_id}' has conflicting {kind} boundary values",
                    suggested_action="provide one consistent boundary value per port and kind",
                    port_id=port_id,
                    witness={
                        "issue": "conflicting_values",
                        "kind": kind,
                        "port_id": port_id,
                        "values": [_complex_witness(value) for value in values],
                    },
                )
            )
            continue

        if len(entries) > 1 and all_equal and kind == "voltage":
            diagnostics.append(
                _boundary_error(
                    code="E_NUM_RF_BOUNDARY_SINGULAR",
                    message=f"port '{port_id}' has redundant voltage boundary declarations",
                    suggested_action="remove redundant voltage boundaries to keep full-rank boundary equations",
                    port_id=port_id,
                    witness={
                        "issue": "redundant_voltage_port_boundary",
                        "kind": kind,
                        "port_id": port_id,
                        "sources": [entry.source for entry in entries],
                        "value": _complex_witness(values[0]),
                    },
                )
            )
            continue

        if len(entries) > 1 and all_equal and kind == "current":
            diagnostics.append(
                _boundary_error(
                    code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                    message=f"port '{port_id}' has duplicate current boundary declarations",
                    suggested_action="provide one current boundary declaration per port",
                    port_id=port_id,
                    witness={
                        "issue": "duplicate_current_port_boundary",
                        "kind": kind,
                        "port_id": port_id,
                        "sources": [entry.source for entry in entries],
                        "value": _complex_witness(values[0]),
                    },
                )
            )
            continue

        resolved.append(entries[0])

    return tuple(sorted(resolved, key=_request_sort_key))


def _check_voltage_constraint_collisions(
    *,
    resolved: tuple[BoundaryRequest, ...],
    port_map: dict[str, PortBoundary],
    diagnostics: list[DiagnosticEvent],
) -> None:
    grouped: dict[tuple[int | None, int | None], list[tuple[str, complex]]] = defaultdict(list)
    for request in resolved:
        port = port_map[request.port_id]
        if _terminal_sort_key(port.p_plus_index) <= _terminal_sort_key(port.p_minus_index):
            key = (port.p_plus_index, port.p_minus_index)
            normalized_value = request.value
        else:
            key = (port.p_minus_index, port.p_plus_index)
            normalized_value = -request.value
        grouped[key].append((request.port_id, normalized_value))

    for key in sorted(
        grouped, key=lambda item: (_terminal_sort_key(item[0]), _terminal_sort_key(item[1]))
    ):
        entries = sorted(grouped[key], key=lambda item: item[0])
        if len(entries) <= 1:
            continue
        values = [value for _, value in entries]
        port_ids = [port_id for port_id, _ in entries]
        all_equal = all(_complex_equal(value, values[0]) for value in values[1:])
        if all_equal:
            diagnostics.append(
                _boundary_error(
                    code="E_NUM_RF_BOUNDARY_SINGULAR",
                    message="redundant voltage constraints create a singular boundary block",
                    suggested_action="declare at most one voltage constraint per terminal pair",
                    witness={
                        "canonical_terminals": [_terminal_token(key[0]), _terminal_token(key[1])],
                        "issue": "redundant_voltage_pair",
                        "port_ids": port_ids,
                        "value": _complex_witness(values[0]),
                    },
                )
            )
            continue
        diagnostics.append(
            _boundary_error(
                code="E_TOPO_RF_BOUNDARY_INCONSISTENT",
                message="conflicting voltage constraints detected for one terminal pair",
                suggested_action="use one consistent voltage boundary per terminal pair",
                witness={
                    "canonical_terminals": [_terminal_token(key[0]), _terminal_token(key[1])],
                    "issue": "conflicting_voltage_pair",
                    "port_ids": port_ids,
                    "values": [_complex_witness(value) for value in values],
                },
            )
        )


def _apply_current_boundaries(
    *,
    rhs: NDArray[np.complex128],
    resolved: tuple[BoundaryRequest, ...],
    port_map: dict[str, PortBoundary],
) -> tuple[AppliedBoundary, ...]:
    applied: list[AppliedBoundary] = []
    for request in resolved:
        port = port_map[request.port_id]
        if port.p_plus_index is not None:
            rhs[port.p_plus_index] += request.value
        if port.p_minus_index is not None:
            rhs[port.p_minus_index] -= request.value
        applied.append(
            AppliedBoundary(
                port_id=request.port_id,
                kind=request.kind,
                source=request.source,
                value=request.value,
                p_plus_index=port.p_plus_index,
                p_minus_index=port.p_minus_index,
            )
        )
    return tuple(applied)


def _apply_voltage_boundaries(
    *,
    matrix: csc_matrix,
    rhs: NDArray[np.complex128],
    resolved: tuple[BoundaryRequest, ...],
    port_map: dict[str, PortBoundary],
) -> tuple[csc_matrix, NDArray[np.complex128], tuple[AppliedBoundary, ...]]:
    n_unknowns = int(matrix.shape[0])
    n_constraints = len(resolved)
    if n_constraints == 0:
        return (matrix.copy(), rhs, ())

    top = hstack(
        (matrix, csc_matrix((n_unknowns, n_constraints), dtype=np.complex128)), format="csc"
    )
    bottom = hstack(
        (
            csc_matrix((n_constraints, n_unknowns), dtype=np.complex128),
            csc_matrix((n_constraints, n_constraints), dtype=np.complex128),
        ),
        format="csc",
    )
    augmented = vstack((top, bottom), format="csc")

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    rhs_aug = np.zeros(n_unknowns + n_constraints, dtype=np.complex128)
    rhs_aug[:n_unknowns] = rhs
    applied: list[AppliedBoundary] = []

    for offset, request in enumerate(resolved):
        port = port_map[request.port_id]
        eq_index = n_unknowns + offset
        aux_index = n_unknowns + offset
        if port.p_plus_index is not None:
            rows.extend((port.p_plus_index, eq_index))
            cols.extend((aux_index, port.p_plus_index))
            data.extend((1.0 + 0.0j, 1.0 + 0.0j))
        if port.p_minus_index is not None:
            rows.extend((port.p_minus_index, eq_index))
            cols.extend((aux_index, port.p_minus_index))
            data.extend((-1.0 + 0.0j, -1.0 + 0.0j))
        rhs_aug[eq_index] = request.value
        applied.append(
            AppliedBoundary(
                port_id=request.port_id,
                kind=request.kind,
                source=request.source,
                value=request.value,
                p_plus_index=port.p_plus_index,
                p_minus_index=port.p_minus_index,
                equation_row_index=eq_index,
                aux_unknown_index=aux_index,
                current_into_dut_from_aux_sign=_AUX_CURRENT_SIGN_INTO_DUT,
            )
        )

    coupling = coo_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows), np.asarray(cols))),
        shape=(n_unknowns + n_constraints, n_unknowns + n_constraints),
        dtype=np.complex128,
    ).tocsc()
    return (augmented + coupling, rhs_aug, tuple(applied))


def _boundary_error(
    *,
    code: str,
    message: str,
    suggested_action: str,
    port_id: str | None = None,
    witness: object | None = None,
) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=code,
        severity=Severity.ERROR,
        message=message,
        suggested_action=suggested_action,
        solver_stage=SolverStage.ASSEMBLE,
        element_id=_RF_BOUNDARY_ELEMENT_ID,
        port_context=PortContext(port_id=port_id) if port_id is not None else None,
        witness=witness,
    )

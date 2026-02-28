from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rfmna.diagnostics import DiagnosticEvent, PortContext, Severity, SolverStage, sort_diagnostics
from rfmna.solver import SolveResult, solve_linear_system
from rfmna.solver.backend import SparseComplexMatrix

from .boundary import BoundaryInjectionResult, PortBoundary, apply_voltage_boundaries

type YPointStatus = Literal["pass", "degraded", "fail"]

_Y_EXTRACT_ELEMENT_ID = "rf_y_params"
_COLUMN_VOLTAGE = 1.0 + 0.0j
_INACTIVE_VOLTAGE = 0.0 + 0.0j
_Y_PAYLOAD_RANK = 3


@runtime_checkable
class YAssemblePointFn(Protocol):
    def __call__(
        self,
        point_index: int,
        frequency_hz: float,
    ) -> tuple[SparseComplexMatrix, NDArray[np.complex128]]: ...


@runtime_checkable
class YSolvePointFn(Protocol):
    def __call__(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
    ) -> SolveResult: ...


@dataclass(frozen=True, slots=True)
class YParameterResult:
    frequencies_hz: NDArray[np.float64]
    port_ids: tuple[str, ...]
    y: NDArray[np.complex128]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...]

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        y_values = np.asarray(self.y, dtype=np.complex128)
        status_values = np.asarray(self.status, dtype=np.str_)

        if frequencies.ndim != 1:
            raise ValueError("frequencies_hz must be one-dimensional")
        if y_values.ndim != _Y_PAYLOAD_RANK:
            raise ValueError("y must be rank-3 with shape [n_points, n_ports, n_ports]")
        if status_values.ndim != 1:
            raise ValueError("status must be one-dimensional")

        n_points = int(frequencies.shape[0])
        n_ports = len(self.port_ids)
        expected_shape = (n_points, n_ports, n_ports)
        if y_values.shape != expected_shape:
            raise ValueError("y shape must match [n_points, n_ports, n_ports]")
        if status_values.shape != (n_points,):
            raise ValueError("status shape must match [n_points]")
        if len(self.diagnostics_by_point) != n_points:
            raise ValueError("diagnostics_by_point length must equal n_points")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "y", y_values)
        object.__setattr__(self, "status", status_values)
        object.__setattr__(
            self,
            "diagnostics_by_point",
            tuple(tuple(sort_diagnostics(point)) for point in self.diagnostics_by_point),
        )


def extract_y_parameters(  # noqa: PLR0912, PLR0915
    freq_hz: Sequence[float] | NDArray[np.float64],
    ports: Sequence[PortBoundary],
    assemble_point: YAssemblePointFn,
    *,
    solve_point: YSolvePointFn | None = None,
    node_voltage_count: int | None = None,
) -> YParameterResult:
    frequencies = np.asarray(freq_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("freq_hz must be one-dimensional")
    if not np.isfinite(frequencies).all():
        raise ValueError("freq_hz entries must be finite")

    canonical_ports = tuple(sorted(ports, key=lambda port: port.port_id))
    n_ports = len(canonical_ports)
    if n_ports not in (1, 2):
        raise ValueError("Phase 1 Y extraction supports exactly 1 or 2 ports")
    inferred_node_voltage_count = _infer_node_voltage_count(canonical_ports)
    resolved_node_voltage_count = (
        inferred_node_voltage_count if node_voltage_count is None else node_voltage_count
    )

    n_points = int(frequencies.shape[0])
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    y_values = np.full((n_points, n_ports, n_ports), complex_nan, dtype=np.complex128)
    status = np.full(n_points, "fail", dtype=np.dtype("<U8"))
    diagnostics_lists: list[list[DiagnosticEvent]] = [list() for _ in range(n_points)]
    solver = (
        solve_point
        if solve_point is not None
        else _default_mna_solve_point(node_voltage_count=resolved_node_voltage_count)
    )

    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        try:
            base_matrix, base_rhs = assemble_point(point_index, frequency_hz)
        except Exception as exc:
            diagnostics_lists[point_index].append(
                _point_error(
                    code="E_NUM_SOLVE_FAILED",
                    message=str(exc),
                    suggested_action="verify Y assembly inputs and per-point matrix construction",
                    stage=SolverStage.ASSEMBLE,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness={
                        "exception_type": type(exc).__name__,
                    },
                )
            )
            continue

        point_matrix = np.full((n_ports, n_ports), complex_nan, dtype=np.complex128)
        point_status: YPointStatus = "pass"
        point_failed = False

        for column_index, driven_port in enumerate(canonical_ports):
            inactive_port_ids = tuple(
                port.port_id for port in canonical_ports if port.port_id != driven_port.port_id
            )
            boundary_result = apply_voltage_boundaries(
                base_matrix,
                base_rhs,
                canonical_ports,
                imposed_port_voltages=((driven_port.port_id, _COLUMN_VOLTAGE),),
                inactive_port_ids=inactive_port_ids,
                inactive_voltage=_INACTIVE_VOLTAGE,
            )
            if boundary_result.diagnostics:
                diagnostics_lists[point_index].extend(
                    _mapped_boundary_diagnostics(
                        result=boundary_result,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        column_index=column_index,
                        driven_port_id=driven_port.port_id,
                    )
                )
                point_failed = True
                point_status = "fail"
                break

            assert boundary_result.matrix is not None
            assert boundary_result.rhs is not None

            try:
                solve_result = solver(
                    boundary_result.matrix,
                    np.asarray(boundary_result.rhs, dtype=np.complex128),
                )
            except Exception as exc:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code="E_NUM_SOLVE_FAILED",
                        message=str(exc),
                        suggested_action="inspect solver configuration and boundary-conditioned matrix",
                        stage=SolverStage.SOLVE,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        port_id=driven_port.port_id,
                        witness={
                            "column_index": column_index,
                            "driven_port_id": driven_port.port_id,
                            "exception_type": type(exc).__name__,
                        },
                    )
                )
                point_failed = True
                point_status = "fail"
                break

            if solve_result.status == "fail" or solve_result.x is None:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=solve_result.failure_code or "E_NUM_SOLVE_FAILED",
                        message=solve_result.failure_message or "boundary-conditioned solve failed",
                        suggested_action="inspect topology, boundaries, and solver trace for this point",
                        stage=SolverStage.SOLVE,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        port_id=driven_port.port_id,
                        witness={
                            "column_index": column_index,
                            "driven_port_id": driven_port.port_id,
                            "failure_category": solve_result.failure_category,
                            "failure_reason": solve_result.failure_reason,
                        },
                    )
                )
                point_failed = True
                point_status = "fail"
                break

            if solve_result.status == "degraded" and point_status == "pass":
                point_status = "degraded"
            diagnostics_lists[point_index].extend(
                _mapped_solver_warnings(
                    solve_result=solve_result,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    column_index=column_index,
                    driven_port_id=driven_port.port_id,
                )
            )

            currents, current_error = _extract_port_currents(
                solve_result=solve_result,
                boundary_result=boundary_result,
                canonical_ports=canonical_ports,
                point_index=point_index,
                frequency_hz=frequency_hz,
                column_index=column_index,
                driven_port_id=driven_port.port_id,
            )
            if current_error is not None:
                diagnostics_lists[point_index].append(current_error)
                point_failed = True
                point_status = "fail"
                break

            assert currents is not None
            point_matrix[:, column_index] = currents

        if point_failed:
            status[point_index] = "fail"
            if not diagnostics_lists[point_index]:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code="E_NUM_SOLVE_FAILED",
                        message="Y extraction fail sentinel applied for this point",
                        suggested_action="inspect per-column boundary and solve diagnostics",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "reason": "fail_sentinel_applied",
                        },
                    )
                )
            continue

        y_values[point_index, :, :] = point_matrix
        status[point_index] = point_status

    diagnostics_by_point = tuple(
        tuple(sort_diagnostics(point_diagnostics)) for point_diagnostics in diagnostics_lists
    )
    return YParameterResult(
        frequencies_hz=frequencies,
        port_ids=tuple(port.port_id for port in canonical_ports),
        y=y_values,
        status=status,
        diagnostics_by_point=diagnostics_by_point,
    )


def _default_mna_solve_point(*, node_voltage_count: int) -> YSolvePointFn:
    if node_voltage_count < 0:
        raise ValueError("node_voltage_count must be >= 0")

    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, node_voltage_count=node_voltage_count)

    return _solve


def _infer_node_voltage_count(ports: Sequence[PortBoundary]) -> int:
    max_index = -1
    for port in ports:
        if port.p_plus_index is not None:
            max_index = max(max_index, port.p_plus_index)
        if port.p_minus_index is not None:
            max_index = max(max_index, port.p_minus_index)
    return max_index + 1


def _mapped_boundary_diagnostics(
    *,
    result: BoundaryInjectionResult,
    point_index: int,
    frequency_hz: float,
    column_index: int,
    driven_port_id: str,
) -> tuple[DiagnosticEvent, ...]:
    mapped: list[DiagnosticEvent] = []
    for diagnostic in result.diagnostics:
        mapped.append(
            DiagnosticEvent(
                code=diagnostic.code,
                severity=diagnostic.severity,
                message=diagnostic.message,
                suggested_action=diagnostic.suggested_action,
                solver_stage=diagnostic.solver_stage,
                element_id=diagnostic.element_id,
                node_context=diagnostic.node_context,
                port_context=diagnostic.port_context,
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness={
                    "boundary_witness": diagnostic.witness,
                    "column_index": column_index,
                    "driven_port_id": driven_port_id,
                },
            )
        )
    return tuple(sort_diagnostics(mapped))


def _mapped_solver_warnings(  # noqa: PLR0913
    *,
    solve_result: SolveResult,
    point_index: int,
    frequency_hz: float,
    column_index: int,
    driven_port_id: str,
) -> tuple[DiagnosticEvent, ...]:
    mapped: list[DiagnosticEvent] = []
    for warning in solve_result.warnings:
        mapped.append(
            DiagnosticEvent(
                code=warning.code,
                severity=Severity.WARNING,
                message=warning.message,
                suggested_action="review warning and solver metadata",
                solver_stage=SolverStage.SOLVE,
                element_id=_Y_EXTRACT_ELEMENT_ID,
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness={
                    "column_index": column_index,
                    "driven_port_id": driven_port_id,
                    "warning_code": warning.code,
                },
            )
        )
    return tuple(sort_diagnostics(mapped))


def _extract_port_currents(  # noqa: PLR0913
    *,
    solve_result: SolveResult,
    boundary_result: BoundaryInjectionResult,
    canonical_ports: tuple[PortBoundary, ...],
    point_index: int,
    frequency_hz: float,
    column_index: int,
    driven_port_id: str,
) -> tuple[NDArray[np.complex128] | None, DiagnosticEvent | None]:
    assert solve_result.x is not None
    solution = np.asarray(solve_result.x, dtype=np.complex128)

    voltage_applied = [
        entry for entry in boundary_result.metadata.applied if entry.kind == "voltage"
    ]
    current_by_port: dict[str, complex] = {}
    for applied in voltage_applied:
        aux_index = applied.aux_unknown_index
        sign = applied.current_into_dut_from_aux_sign
        if aux_index is None or sign is None:
            return (
                None,
                _point_error(
                    code="E_NUM_SOLVE_FAILED",
                    message="boundary metadata missing voltage auxiliary current mapping",
                    suggested_action="ensure boundary engine returns voltage aux mapping metadata",
                    stage=SolverStage.POSTPROCESS,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    port_id=applied.port_id,
                    witness={
                        "column_index": column_index,
                        "driven_port_id": driven_port_id,
                        "port_id": applied.port_id,
                    },
                ),
            )
        if aux_index >= int(solution.shape[0]):
            return (
                None,
                _point_error(
                    code="E_NUM_SOLVE_FAILED",
                    message="boundary metadata auxiliary index is out of solver solution range",
                    suggested_action="verify unknown indexing alignment for boundary-conditioned solves",
                    stage=SolverStage.POSTPROCESS,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    port_id=applied.port_id,
                    witness={
                        "aux_unknown_index": aux_index,
                        "column_index": column_index,
                        "driven_port_id": driven_port_id,
                        "solution_size": int(solution.shape[0]),
                    },
                ),
            )
        current_by_port[applied.port_id] = complex(sign) * solution[aux_index]

    currents: list[complex] = []
    for port in canonical_ports:
        if port.port_id not in current_by_port:
            return (
                None,
                _point_error(
                    code="E_NUM_SOLVE_FAILED",
                    message="missing port current mapping in boundary metadata",
                    suggested_action="ensure voltage boundaries are applied for all canonical ports",
                    stage=SolverStage.POSTPROCESS,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    port_id=port.port_id,
                    witness={
                        "available_port_ids": sorted(current_by_port),
                        "column_index": column_index,
                        "driven_port_id": driven_port_id,
                        "port_id": port.port_id,
                    },
                ),
            )
        currents.append(current_by_port[port.port_id])
    return (np.asarray(currents, dtype=np.complex128), None)


def _point_error(  # noqa: PLR0913
    *,
    code: str,
    message: str,
    suggested_action: str,
    stage: SolverStage,
    point_index: int,
    frequency_hz: float,
    port_id: str | None = None,
    witness: object | None = None,
) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=code,
        severity=Severity.ERROR,
        message=message,
        suggested_action=suggested_action,
        solver_stage=stage,
        element_id=_Y_EXTRACT_ELEMENT_ID,
        port_context=PortContext(port_id=port_id) if port_id is not None else None,
        frequency_hz=frequency_hz,
        frequency_index=point_index,
        witness=witness,
    )

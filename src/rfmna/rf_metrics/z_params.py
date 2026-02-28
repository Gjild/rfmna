from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.diagnostics import (
    DiagnosticEvent,
    Severity,
    SolverStage,
    build_diagnostic_event,
    prefixed_witness,
    remap_diagnostic_event,
    sort_diagnostics,
)
from rfmna.solver import (
    FallbackRunConfig,
    SolveResult,
    load_solver_threshold_config,
    solve_linear_system,
)
from rfmna.solver.backend import SparseComplexMatrix

from .boundary import BoundaryInjectionResult, PortBoundary, apply_current_boundaries
from .y_params import extract_y_parameters

type ZPointStatus = Literal["pass", "degraded", "fail"]
type ZExtractionMode = Literal["direct", "y_to_z"]

_Z_EXTRACT_ELEMENT_ID = "rf_z_params"
_DIRECT_CURRENT = 1.0 + 0.0j
_INACTIVE_CURRENT = 0.0 + 0.0j
_Z_PAYLOAD_RANK = 3
_ZBLOCK_SINGULAR_CODE = "E_NUM_ZBLOCK_SINGULAR"
_ZBLOCK_ILL_CONDITIONED_CODE = "E_NUM_ZBLOCK_ILL_CONDITIONED"
_COND_WARN_MAX = load_solver_threshold_config().conditioning.warn_max
_COND_FAIL_MAX = load_solver_threshold_config().conditioning.fail_max


@runtime_checkable
class ZAssemblePointFn(Protocol):
    def __call__(
        self,
        point_index: int,
        frequency_hz: float,
    ) -> tuple[SparseComplexMatrix, NDArray[np.complex128]]: ...


@runtime_checkable
class ZSolvePointFn(Protocol):
    def __call__(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
    ) -> SolveResult: ...


@dataclass(frozen=True, slots=True)
class ZParameterResult:
    frequencies_hz: NDArray[np.float64]
    port_ids: tuple[str, ...]
    z: NDArray[np.complex128]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...]
    extraction_mode: ZExtractionMode

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        z_values = np.asarray(self.z, dtype=np.complex128)
        status_values = np.asarray(self.status, dtype=np.str_)

        if frequencies.ndim != 1:
            raise ValueError("frequencies_hz must be one-dimensional")
        if z_values.ndim != _Z_PAYLOAD_RANK:
            raise ValueError("z must be rank-3 with shape [n_points, n_ports, n_ports]")
        if status_values.ndim != 1:
            raise ValueError("status must be one-dimensional")

        n_points = int(frequencies.shape[0])
        n_ports = len(self.port_ids)
        if z_values.shape != (n_points, n_ports, n_ports):
            raise ValueError("z shape must match [n_points, n_ports, n_ports]")
        if status_values.shape != (n_points,):
            raise ValueError("status shape must match [n_points]")
        if len(self.diagnostics_by_point) != n_points:
            raise ValueError("diagnostics_by_point length must equal n_points")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "z", z_values)
        object.__setattr__(self, "status", status_values)
        object.__setattr__(
            self,
            "diagnostics_by_point",
            tuple(tuple(sort_diagnostics(point)) for point in self.diagnostics_by_point),
        )


def extract_z_parameters(  # noqa: PLR0912, PLR0913, PLR0915
    freq_hz: Sequence[float] | NDArray[np.float64],
    ports: Sequence[PortBoundary],
    assemble_point: ZAssemblePointFn,
    *,
    solve_point: ZSolvePointFn | None = None,
    node_voltage_count: int | None = None,
    extraction_mode: ZExtractionMode = "direct",
) -> ZParameterResult:
    frequencies = np.asarray(freq_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("freq_hz must be one-dimensional")
    if not np.isfinite(frequencies).all():
        raise ValueError("freq_hz entries must be finite")

    canonical_ports = tuple(sorted(ports, key=lambda port: port.port_id))
    n_ports = len(canonical_ports)
    if n_ports not in (1, 2):
        raise ValueError("Phase 1 Z extraction supports exactly 1 or 2 ports")
    inferred_node_voltage_count = _infer_node_voltage_count(canonical_ports)
    resolved_node_voltage_count = (
        inferred_node_voltage_count if node_voltage_count is None else node_voltage_count
    )

    n_points = int(frequencies.shape[0])
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    z_values = np.full((n_points, n_ports, n_ports), complex_nan, dtype=np.complex128)
    status = np.full(n_points, "fail", dtype=np.dtype("<U8"))
    diagnostics_lists: list[list[DiagnosticEvent]] = [list() for _ in range(n_points)]
    solver = (
        solve_point
        if solve_point is not None
        else _default_mna_solve_point(node_voltage_count=resolved_node_voltage_count)
    )
    conversion_solver = _default_conversion_solve_point()

    if extraction_mode == "direct":
        _extract_direct_columns(
            frequencies=frequencies,
            canonical_ports=canonical_ports,
            assemble_point=assemble_point,
            solver=solver,
            z_values=z_values,
            status=status,
            diagnostics_lists=diagnostics_lists,
        )
    else:
        _extract_via_y_to_z(
            frequencies=frequencies,
            canonical_ports=canonical_ports,
            assemble_point=assemble_point,
            solver=solver,
            conversion_solver=conversion_solver,
            z_values=z_values,
            status=status,
            diagnostics_lists=diagnostics_lists,
        )

    return ZParameterResult(
        frequencies_hz=frequencies,
        port_ids=tuple(port.port_id for port in canonical_ports),
        z=z_values,
        status=status,
        diagnostics_by_point=tuple(tuple(sort_diagnostics(point)) for point in diagnostics_lists),
        extraction_mode=extraction_mode,
    )


def _extract_direct_columns(  # noqa: PLR0913, PLR0915
    *,
    frequencies: NDArray[np.float64],
    canonical_ports: tuple[PortBoundary, ...],
    assemble_point: ZAssemblePointFn,
    solver: ZSolvePointFn,
    z_values: NDArray[np.complex128],
    status: NDArray[np.str_],
    diagnostics_lists: list[list[DiagnosticEvent]],
) -> None:
    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        try:
            base_matrix, base_rhs = assemble_point(point_index, frequency_hz)
        except Exception as exc:
            diagnostics_lists[point_index].append(
                _point_error(
                    code="E_NUM_SOLVE_FAILED",
                    message=str(exc),
                    suggested_action="verify Z assembly inputs and per-point matrix construction",
                    stage=SolverStage.ASSEMBLE,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness={"exception_type": type(exc).__name__},
                )
            )
            continue

        n_ports = len(canonical_ports)
        point_matrix = np.full(
            (n_ports, n_ports), np.complex128(complex(float("nan"), float("nan")))
        )
        point_status: ZPointStatus = "pass"
        point_failed = False

        for column_index, driven_port in enumerate(canonical_ports):
            inactive_port_ids = tuple(
                port.port_id for port in canonical_ports if port.port_id != driven_port.port_id
            )
            boundary_result = apply_current_boundaries(
                base_matrix,
                base_rhs,
                canonical_ports,
                imposed_port_currents=((driven_port.port_id, _DIRECT_CURRENT),),
                inactive_port_ids=inactive_port_ids,
                inactive_current=_INACTIVE_CURRENT,
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
                    boundary_result.matrix, np.asarray(boundary_result.rhs, dtype=np.complex128)
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

            voltages, voltage_error = _extract_port_voltages(
                solve_result=solve_result,
                canonical_ports=canonical_ports,
                point_index=point_index,
                frequency_hz=frequency_hz,
                column_index=column_index,
                driven_port_id=driven_port.port_id,
            )
            if voltage_error is not None:
                diagnostics_lists[point_index].append(voltage_error)
                point_failed = True
                point_status = "fail"
                break

            assert voltages is not None
            point_matrix[:, column_index] = voltages

        if point_failed:
            status[point_index] = "fail"
            if not diagnostics_lists[point_index]:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code="E_NUM_SOLVE_FAILED",
                        message="Z extraction fail sentinel applied for this point",
                        suggested_action="inspect per-column boundary and solve diagnostics",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={"reason": "fail_sentinel_applied"},
                    )
                )
            continue

        z_values[point_index, :, :] = point_matrix
        status[point_index] = point_status


def _extract_via_y_to_z(  # noqa: PLR0913
    *,
    frequencies: NDArray[np.float64],
    canonical_ports: tuple[PortBoundary, ...],
    assemble_point: ZAssemblePointFn,
    solver: ZSolvePointFn,
    conversion_solver: ZSolvePointFn,
    z_values: NDArray[np.complex128],
    status: NDArray[np.str_],
    diagnostics_lists: list[list[DiagnosticEvent]],
) -> None:
    y_result = extract_y_parameters(
        frequencies,
        canonical_ports,
        assemble_point,
        solve_point=solver,
    )
    n_ports = len(canonical_ports)
    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        point_status = str(y_result.status[point_index])
        if point_status == "fail":
            diagnostics_lists[point_index].extend(
                _mapped_upstream_diagnostics(
                    diagnostics=y_result.diagnostics_by_point[point_index],
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness_prefix="y_extraction",
                )
            )
            status[point_index] = "fail"
            continue

        y_block = np.asarray(y_result.y[point_index, :, :], dtype=np.complex128)
        y_sparse = csc_matrix(y_block)
        z_block = np.full((n_ports, n_ports), np.complex128(complex(float("nan"), float("nan"))))
        conversion_failed = False
        conversion_status: ZPointStatus = "degraded" if point_status == "degraded" else "pass"

        for column_index in range(n_ports):
            rhs = np.zeros(n_ports, dtype=np.complex128)
            rhs[column_index] = 1.0 + 0.0j
            solve_result = conversion_solver(y_sparse, rhs)
            if _conversion_attempt_uses_gmin(solve_result):
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_ZBLOCK_SINGULAR_CODE,
                        message="Y->Z conversion rejected: gmin regularization is forbidden in conversion math",
                        suggested_action="use default conversion solver controls without gmin regularization",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "reason": "gmin_regularization_forbidden",
                        },
                    )
                )
                conversion_failed = True
                conversion_status = "fail"
                break
            if solve_result.status == "fail" or solve_result.x is None:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_ZBLOCK_SINGULAR_CODE,
                        message="Y->Z conversion failed due singular Z block solve",
                        suggested_action="use direct Z extraction or fix singular/contradictory boundary conditions",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "cond_ind": solve_result.cond_ind,
                            "failure_category": solve_result.failure_category,
                            "failure_code": solve_result.failure_code,
                            "failure_reason": solve_result.failure_reason,
                        },
                    )
                )
                conversion_failed = True
                conversion_status = "fail"
                break

            cond_ind = float(solve_result.cond_ind)
            if np.isfinite(cond_ind) and cond_ind <= _COND_FAIL_MAX:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_ZBLOCK_SINGULAR_CODE,
                        message="Y->Z conversion failed due singular condition-indicator gate",
                        suggested_action="use direct Z extraction or adjust topology to remove singularity",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "cond_fail_max": _COND_FAIL_MAX,
                            "cond_ind": cond_ind,
                        },
                    )
                )
                conversion_failed = True
                conversion_status = "fail"
                break

            if np.isfinite(cond_ind) and cond_ind <= _COND_WARN_MAX:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_ZBLOCK_ILL_CONDITIONED_CODE,
                        message="Y->Z conversion failed due ill-conditioned Z block gate",
                        suggested_action="use direct Z extraction for this point or improve conditioning",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "cond_ind": cond_ind,
                            "cond_warn_max": _COND_WARN_MAX,
                        },
                    )
                )
                conversion_failed = True
                conversion_status = "fail"
                break

            if solve_result.status == "degraded" and conversion_status == "pass":
                conversion_status = "degraded"
            diagnostics_lists[point_index].extend(
                _mapped_solver_warnings(
                    solve_result=solve_result,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    column_index=column_index,
                    driven_port_id=canonical_ports[column_index].port_id,
                )
            )

            z_block[:, column_index] = np.asarray(solve_result.x, dtype=np.complex128)

        if conversion_failed:
            status[point_index] = "fail"
            continue

        z_values[point_index, :, :] = z_block
        status[point_index] = conversion_status


def _default_mna_solve_point(*, node_voltage_count: int) -> ZSolvePointFn:
    if node_voltage_count < 0:
        raise ValueError("node_voltage_count must be >= 0")

    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, node_voltage_count=node_voltage_count)

    return _solve


def _default_conversion_solve_point() -> ZSolvePointFn:
    conversion_run_config = FallbackRunConfig(enable_gmin=False)

    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, run_config=conversion_run_config)

    return _solve


def _infer_node_voltage_count(ports: Sequence[PortBoundary]) -> int:
    max_index = -1
    for port in ports:
        if port.p_plus_index is not None:
            max_index = max(max_index, port.p_plus_index)
        if port.p_minus_index is not None:
            max_index = max(max_index, port.p_minus_index)
    return max_index + 1


def _conversion_attempt_uses_gmin(solve_result: SolveResult) -> bool:
    return any(
        row.stage == "gmin" and row.stage_state == "run" for row in solve_result.attempt_trace
    )


def _extract_port_voltages(  # noqa: PLR0913
    *,
    solve_result: SolveResult,
    canonical_ports: tuple[PortBoundary, ...],
    point_index: int,
    frequency_hz: float,
    column_index: int,
    driven_port_id: str,
) -> tuple[NDArray[np.complex128] | None, DiagnosticEvent | None]:
    assert solve_result.x is not None
    solution = np.asarray(solve_result.x, dtype=np.complex128)
    voltages: list[complex] = []

    for port in canonical_ports:
        p_plus_voltage, plus_error = _node_voltage(
            solution=solution,
            node_index=port.p_plus_index,
            point_index=point_index,
            frequency_hz=frequency_hz,
            column_index=column_index,
            driven_port_id=driven_port_id,
            port_id=port.port_id,
        )
        if plus_error is not None:
            return (None, plus_error)
        p_minus_voltage, minus_error = _node_voltage(
            solution=solution,
            node_index=port.p_minus_index,
            point_index=point_index,
            frequency_hz=frequency_hz,
            column_index=column_index,
            driven_port_id=driven_port_id,
            port_id=port.port_id,
        )
        if minus_error is not None:
            return (None, minus_error)
        assert p_plus_voltage is not None
        assert p_minus_voltage is not None
        voltages.append(p_plus_voltage - p_minus_voltage)
    return (np.asarray(voltages, dtype=np.complex128), None)


def _node_voltage(  # noqa: PLR0913
    *,
    solution: NDArray[np.complex128],
    node_index: int | None,
    point_index: int,
    frequency_hz: float,
    column_index: int,
    driven_port_id: str,
    port_id: str,
) -> tuple[complex | None, DiagnosticEvent | None]:
    if node_index is None:
        return (0.0 + 0.0j, None)
    if node_index >= int(solution.shape[0]):
        return (
            None,
            _point_error(
                code="E_NUM_SOLVE_FAILED",
                message="port node index is out of solver solution range",
                suggested_action="verify port node indexing alignment for Z extraction",
                stage=SolverStage.POSTPROCESS,
                point_index=point_index,
                frequency_hz=frequency_hz,
                port_id=port_id,
                witness={
                    "column_index": column_index,
                    "driven_port_id": driven_port_id,
                    "node_index": node_index,
                    "solution_size": int(solution.shape[0]),
                },
            ),
        )
    return (complex(solution[node_index]), None)


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
            remap_diagnostic_event(
                diagnostic,
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness=prefixed_witness(
                    prefix="boundary_witness",
                    payload=diagnostic.witness,
                    extras={
                        "column_index": column_index,
                        "driven_port_id": driven_port_id,
                    },
                ),
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
            build_diagnostic_event(
                code=warning.code,
                severity=Severity.WARNING,
                message=warning.message,
                suggested_action="review warning and solver metadata",
                solver_stage=SolverStage.SOLVE,
                element_id=_Z_EXTRACT_ELEMENT_ID,
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


def _mapped_upstream_diagnostics(
    *,
    diagnostics: tuple[DiagnosticEvent, ...],
    point_index: int,
    frequency_hz: float,
    witness_prefix: str,
) -> tuple[DiagnosticEvent, ...]:
    mapped: list[DiagnosticEvent] = []
    for diagnostic in diagnostics:
        mapped.append(
            remap_diagnostic_event(
                diagnostic,
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness=prefixed_witness(
                    prefix=witness_prefix,
                    payload=diagnostic.witness,
                ),
            )
        )
    return tuple(sort_diagnostics(mapped))


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
    return build_diagnostic_event(
        code=code,
        message=message,
        suggested_action=suggested_action,
        solver_stage=stage,
        element_id=_Z_EXTRACT_ELEMENT_ID,
        port_id=port_id,
        frequency_hz=frequency_hz,
        frequency_index=point_index,
        witness=witness,
    )

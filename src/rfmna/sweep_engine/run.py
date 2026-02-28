from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rfmna.diagnostics import (
    DiagnosticEvent,
    Severity,
    SolverStage,
    build_diagnostic_event,
    diagnostic_sort_key,
)
from rfmna.rf_metrics import (
    convert_y_to_s,
    convert_z_to_s,
    extract_y_parameters,
    extract_z_parameters,
    extract_zin_zout,
)
from rfmna.rf_metrics.impedance import ZinZoutResult
from rfmna.solver import FallbackRunConfig, SolveResult, solve_linear_system
from rfmna.solver.backend import SparseComplexMatrix

from .types import (
    RFMetricName,
    SweepRFMetricPayload,
    SweepRFPayloads,
    SweepRFRequest,
    SweepRFScalarResult,
)

type SweepStatus = Literal["pass", "degraded", "fail"]
type SweepStage = Literal["parse", "preflight", "assemble", "solve", "postprocess"]
type DiagnosticSeverity = Literal["error", "warning"]

_SWEEP_SEVERITY_TO_CANONICAL: dict[DiagnosticSeverity, Severity] = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
}

_SWEEP_STAGE_TO_CANONICAL: dict[SweepStage, SolverStage] = {
    "parse": SolverStage.PARSE,
    "preflight": SolverStage.PREFLIGHT,
    "assemble": SolverStage.ASSEMBLE,
    "solve": SolverStage.SOLVE,
    "postprocess": SolverStage.POSTPROCESS,
}


@dataclass(frozen=True, slots=True)
class SweepLayout:
    n_nodes: int
    n_aux: int

    def __post_init__(self) -> None:
        if self.n_nodes < 0:
            raise ValueError("n_nodes must be >= 0")
        if self.n_aux < 0:
            raise ValueError("n_aux must be >= 0")


@dataclass(frozen=True, slots=True)
class SweepDiagnostic:
    code: str
    severity: DiagnosticSeverity
    message: str
    suggested_action: str
    solver_stage: SweepStage
    point_index: int
    frequency_hz: float
    element_id: str = "sweep_engine"
    frequency_index: int | None = None
    sweep_index: int | None = None
    witness: object | None = None

    def __post_init__(self) -> None:
        if self.point_index < 0:
            raise ValueError("point_index must be >= 0")
        if not np.isfinite(self.frequency_hz):
            raise ValueError("frequency_hz must be finite")
        if not self.element_id:
            raise ValueError("element_id must be non-empty")
        if self.frequency_index is None:
            object.__setattr__(self, "frequency_index", self.point_index)
        elif self.frequency_index < 0:
            raise ValueError("frequency_index must be >= 0")
        if self.sweep_index is not None and self.sweep_index < 0:
            raise ValueError("sweep_index must be >= 0")


@dataclass(frozen=True, slots=True)
class SweepResult:
    n_points: int
    n_nodes: int
    n_aux: int
    V_nodes: NDArray[np.complex128]
    I_aux: NDArray[np.complex128]
    res_l2: NDArray[np.float64]
    res_linf: NDArray[np.float64]
    res_rel: NDArray[np.float64]
    cond_ind: NDArray[np.float64]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[SweepDiagnostic, ...], ...]
    rf_payloads: SweepRFPayloads | None = None


@runtime_checkable
class AssemblePointFn(Protocol):
    def __call__(
        self,
        point_index: int,
        frequency_hz: float,
    ) -> tuple[SparseComplexMatrix, NDArray[np.complex128]]: ...


@runtime_checkable
class SolvePointFn(Protocol):
    def __call__(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
    ) -> SolveResult: ...


def _default_mna_solve_point(*, node_voltage_count: int) -> SolvePointFn:
    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, node_voltage_count=node_voltage_count)

    return _solve


def _default_conversion_solve_point() -> SolvePointFn:
    conversion_run_config = FallbackRunConfig(enable_gmin=False)

    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, run_config=conversion_run_config)

    return _solve


@dataclass(frozen=True, slots=True)
class _SweepBuffers:
    V_nodes: NDArray[np.complex128]
    I_aux: NDArray[np.complex128]
    res_l2: NDArray[np.float64]
    res_linf: NDArray[np.float64]
    res_rel: NDArray[np.float64]
    cond_ind: NDArray[np.float64]
    status: NDArray[np.str_]


def run_sweep(  # noqa: PLR0913
    freq_hz: Sequence[float] | NDArray[np.float64],
    layout: SweepLayout,
    assemble_point: AssemblePointFn,
    *,
    solve_point: SolvePointFn | None = None,
    conversion_solve_point: SolvePointFn | None = None,
    rf_request: SweepRFRequest | None = None,
) -> SweepResult:
    frequencies = np.asarray(freq_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("freq_hz must be one-dimensional")
    if not np.isfinite(frequencies).all():
        raise ValueError("freq_hz entries must be finite")

    n_points = int(frequencies.shape[0])
    buffers = _allocate_buffers(n_points=n_points, n_nodes=layout.n_nodes, n_aux=layout.n_aux)
    diagnostics_lists: list[list[SweepDiagnostic]] = [list() for _ in range(n_points)]
    solver = (
        solve_point
        if solve_point is not None
        else _default_mna_solve_point(node_voltage_count=layout.n_nodes)
    )
    conversion_solver = (
        conversion_solve_point
        if conversion_solve_point is not None
        else _default_conversion_solve_point()
    )

    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        try:
            matrix, rhs = assemble_point(point_index, frequency_hz)
        except Exception as exc:
            diagnostics_lists[point_index].append(
                _error_diagnostic(
                    code="E_NUM_SOLVE_FAILED",
                    message=str(exc),
                    suggested_action="verify assembly inputs and per-point matrix construction",
                    stage="assemble",
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness={"exception_type": type(exc).__name__},
                )
            )
            _apply_fail_sentinel(buffers, point_index)
            continue

        try:
            solve_result = solver(matrix, np.asarray(rhs, dtype=np.complex128))
        except Exception as exc:
            diagnostics_lists[point_index].append(
                _error_diagnostic(
                    code="E_NUM_SOLVE_FAILED",
                    message=str(exc),
                    suggested_action="inspect solver configuration and matrix conditioning",
                    stage="solve",
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    element_id="solver",
                    witness={"exception_type": type(exc).__name__},
                )
            )
            _apply_fail_sentinel(buffers, point_index)
            continue

        for warning in solve_result.warnings:
            warning_event = build_diagnostic_event(
                code=warning.code,
                severity=Severity.WARNING,
                message=warning.message,
                suggested_action="review warning and solver metadata",
                solver_stage=SolverStage.SOLVE,
                element_id="solver",
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness={"warning_code": warning.code},
            )
            diagnostics_lists[point_index].append(
                _event_to_sweep_diagnostic(
                    warning_event,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                )
            )

        if solve_result.status == "fail":
            diagnostics_lists[point_index].append(
                _error_diagnostic(
                    code=solve_result.failure_code or "E_NUM_SOLVE_FAILED",
                    message=solve_result.failure_message or "point solve failed",
                    suggested_action="inspect topology, parameters, and solver trace for this point",
                    stage="solve",
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    element_id="solver",
                    witness={
                        "failure_category": solve_result.failure_category,
                        "failure_reason": solve_result.failure_reason,
                    },
                )
            )
            _apply_fail_sentinel(buffers, point_index)
            continue

        if not _can_map_solution(solve_result, layout):
            diagnostics_lists[point_index].append(
                _error_diagnostic(
                    code="E_NUM_SOLVE_FAILED",
                    message="solver solution length does not match n_nodes + n_aux",
                    suggested_action="ensure indexing metadata matches solver unknown ordering",
                    stage="postprocess",
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness={
                        "actual_solution_size": None
                        if solve_result.x is None
                        else int(solve_result.x.shape[0]),
                        "expected_solution_size": layout.n_nodes + layout.n_aux,
                    },
                )
            )
            _apply_fail_sentinel(buffers, point_index)
            continue

        _write_success_point(
            buffers=buffers,
            point_index=point_index,
            solve_result=solve_result,
            n_nodes=layout.n_nodes,
            n_aux=layout.n_aux,
        )

    diagnostics_by_point = _finalize_diagnostics(
        diagnostics_lists=diagnostics_lists,
        status=buffers.status,
        frequencies=frequencies,
    )
    rf_payloads = (
        _compute_rf_payloads(
            frequencies=frequencies,
            rf_request=rf_request,
            assemble_point=assemble_point,
            solve_point=solver,
            conversion_solve_point=conversion_solver,
            node_voltage_count=layout.n_nodes,
        )
        if rf_request is not None
        else None
    )

    return SweepResult(
        n_points=n_points,
        n_nodes=layout.n_nodes,
        n_aux=layout.n_aux,
        V_nodes=buffers.V_nodes,
        I_aux=buffers.I_aux,
        res_l2=buffers.res_l2,
        res_linf=buffers.res_linf,
        res_rel=buffers.res_rel,
        cond_ind=buffers.cond_ind,
        status=buffers.status,
        diagnostics_by_point=diagnostics_by_point,
        rf_payloads=rf_payloads,
    )


def _compute_rf_payloads(  # noqa: PLR0912, PLR0913
    *,
    frequencies: NDArray[np.float64],
    rf_request: SweepRFRequest,
    assemble_point: AssemblePointFn,
    solve_point: SolvePointFn | None,
    conversion_solve_point: SolvePointFn | None = None,
    node_voltage_count: int | None = None,
) -> SweepRFPayloads:
    y_result = None
    z_result = None
    impedance_result = None
    entries: list[tuple[RFMetricName, SweepRFMetricPayload]] = []
    conversion_solver = (
        conversion_solve_point
        if conversion_solve_point is not None
        else _default_conversion_solve_point()
    )

    for metric_name in rf_request.metrics:
        if metric_name == "y":
            if y_result is None:
                y_result = extract_y_parameters(
                    frequencies,
                    rf_request.ports,
                    assemble_point,
                    solve_point=solve_point,
                    node_voltage_count=node_voltage_count,
                )
            entries.append(("y", y_result))
            continue

        if metric_name == "z":
            if z_result is None:
                z_result = extract_z_parameters(
                    frequencies,
                    rf_request.ports,
                    assemble_point,
                    solve_point=solve_point,
                    node_voltage_count=node_voltage_count,
                    extraction_mode="direct",
                )
            entries.append(("z", z_result))
            continue

        if metric_name == "s":
            if rf_request.s_conversion_source == "from_y":
                if y_result is None:
                    y_result = extract_y_parameters(
                        frequencies,
                        rf_request.ports,
                        assemble_point,
                        solve_point=solve_point,
                        node_voltage_count=node_voltage_count,
                    )
                entries.append(
                    (
                        "s",
                        convert_y_to_s(
                            y_result,
                            z0_ohm=rf_request.z0_ohm,
                            solve_point=conversion_solver,
                        ),
                    )
                )
            else:
                if z_result is None:
                    z_result = extract_z_parameters(
                        frequencies,
                        rf_request.ports,
                        assemble_point,
                        solve_point=solve_point,
                        node_voltage_count=node_voltage_count,
                        extraction_mode="direct",
                    )
                entries.append(
                    (
                        "s",
                        convert_z_to_s(
                            z_result,
                            z0_ohm=rf_request.z0_ohm,
                            solve_point=conversion_solver,
                        ),
                    )
                )
            continue

        if metric_name in {"zin", "zout"}:
            if impedance_result is None:
                impedance_result = extract_zin_zout(
                    frequencies,
                    rf_request.ports,
                    assemble_point,
                    solve_point=solve_point,
                    node_voltage_count=node_voltage_count,
                    input_port_id=rf_request.input_port_id,
                    output_port_id=rf_request.output_port_id,
                )
            if metric_name == "zin":
                entries.append(("zin", _project_scalar_metric(impedance_result, metric_name)))
            else:
                entries.append(("zout", _project_scalar_metric(impedance_result, metric_name)))
            continue

        raise ValueError(f"unsupported rf metric '{metric_name}'")

    return SweepRFPayloads(by_metric=tuple(entries))


def _project_scalar_metric(
    impedance_result: ZinZoutResult,
    metric_name: Literal["zin", "zout"],
) -> SweepRFScalarResult:
    if metric_name == "zin":
        values = np.asarray(impedance_result.zin, dtype=np.complex128)
        selected_port_id = impedance_result.input_port_id
    else:
        values = np.asarray(impedance_result.zout, dtype=np.complex128)
        selected_port_id = impedance_result.output_port_id

    return SweepRFScalarResult(
        frequencies_hz=np.asarray(impedance_result.frequencies_hz, dtype=np.float64),
        port_ids=tuple(impedance_result.port_ids),
        port_id=selected_port_id,
        metric_name=metric_name,
        values=values,
        status=np.asarray(impedance_result.status, dtype=np.str_),
        diagnostics_by_point=impedance_result.diagnostics_by_point,
    )


def _allocate_buffers(*, n_points: int, n_nodes: int, n_aux: int) -> _SweepBuffers:
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    V_nodes = np.full((n_points, n_nodes), complex_nan, dtype=np.complex128)
    I_aux = np.full((n_points, n_aux), complex_nan, dtype=np.complex128)
    res_l2 = np.full(n_points, np.nan, dtype=np.float64)
    res_linf = np.full(n_points, np.nan, dtype=np.float64)
    res_rel = np.full(n_points, np.nan, dtype=np.float64)
    cond_ind = np.full(n_points, np.nan, dtype=np.float64)
    status = np.full(n_points, "fail", dtype=np.dtype("<U8"))
    return _SweepBuffers(
        V_nodes=V_nodes,
        I_aux=I_aux,
        res_l2=res_l2,
        res_linf=res_linf,
        res_rel=res_rel,
        cond_ind=cond_ind,
        status=status,
    )


def _apply_fail_sentinel(buffers: _SweepBuffers, point_index: int) -> None:
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    buffers.V_nodes[point_index, :] = complex_nan
    buffers.I_aux[point_index, :] = complex_nan
    buffers.res_l2[point_index] = np.nan
    buffers.res_linf[point_index] = np.nan
    buffers.res_rel[point_index] = np.nan
    buffers.cond_ind[point_index] = np.nan
    buffers.status[point_index] = "fail"


def _can_map_solution(result: SolveResult, layout: SweepLayout) -> bool:
    if result.x is None:
        return False
    expected = layout.n_nodes + layout.n_aux
    return bool(result.x.ndim == 1 and result.x.shape[0] == expected)


def _write_success_point(
    *,
    buffers: _SweepBuffers,
    point_index: int,
    solve_result: SolveResult,
    n_nodes: int,
    n_aux: int,
) -> None:
    assert solve_result.x is not None
    x = solve_result.x
    buffers.V_nodes[point_index, :] = x[0:n_nodes]
    buffers.I_aux[point_index, :] = x[n_nodes : n_nodes + n_aux]
    buffers.res_l2[point_index] = solve_result.residual.res_l2
    buffers.res_linf[point_index] = solve_result.residual.res_linf
    buffers.res_rel[point_index] = solve_result.residual.res_rel
    buffers.cond_ind[point_index] = solve_result.cond_ind
    buffers.status[point_index] = solve_result.status


def _error_diagnostic(  # noqa: PLR0913
    *,
    code: str,
    message: str,
    suggested_action: str,
    stage: SweepStage,
    point_index: int,
    frequency_hz: float,
    element_id: str = "sweep_engine",
    witness: object | None = None,
) -> SweepDiagnostic:
    event = build_diagnostic_event(
        code=code,
        severity=Severity.ERROR,
        message=message,
        suggested_action=suggested_action,
        solver_stage=_SWEEP_STAGE_TO_CANONICAL[stage],
        element_id=element_id,
        frequency_hz=frequency_hz,
        frequency_index=point_index,
        witness=witness,
    )
    return _event_to_sweep_diagnostic(
        event,
        point_index=point_index,
        frequency_hz=frequency_hz,
    )


def _finalize_diagnostics(
    *,
    diagnostics_lists: list[list[SweepDiagnostic]],
    status: NDArray[np.str_],
    frequencies: NDArray[np.float64],
) -> tuple[tuple[SweepDiagnostic, ...], ...]:
    finalized: list[tuple[SweepDiagnostic, ...]] = []
    for point_index, point_diagnostics in enumerate(diagnostics_lists):
        if status[point_index] == "fail" and not point_diagnostics:
            point_diagnostics.append(
                _error_diagnostic(
                    code="E_NUM_SOLVE_FAILED",
                    message="fail sentinel applied for this point",
                    suggested_action="inspect per-point assembly and solver diagnostics",
                    stage="postprocess",
                    point_index=point_index,
                    frequency_hz=float(frequencies[point_index]),
                    witness={
                        "reason": "fail_sentinel_applied",
                    },
                )
            )
        ordered = tuple(sorted(point_diagnostics, key=sweep_diagnostic_sort_key))
        finalized.append(ordered)
    return tuple(finalized)


def sweep_diagnostic_sort_key(
    diag: SweepDiagnostic,
) -> tuple[int, int, str, tuple[int, str], tuple[int, int, int, int], str, str]:
    return diagnostic_sort_key(_sweep_diagnostic_to_event(diag))


def _sweep_diagnostic_to_event(diag: SweepDiagnostic) -> DiagnosticEvent:
    return build_diagnostic_event(
        code=diag.code,
        severity=_SWEEP_SEVERITY_TO_CANONICAL[diag.severity],
        message=diag.message,
        suggested_action=diag.suggested_action,
        solver_stage=_SWEEP_STAGE_TO_CANONICAL[diag.solver_stage],
        element_id=diag.element_id,
        frequency_hz=diag.frequency_hz,
        frequency_index=diag.frequency_index,
        sweep_index=diag.sweep_index,
        witness=diag.witness,
    )


def _event_to_sweep_diagnostic(
    event: DiagnosticEvent,
    *,
    point_index: int,
    frequency_hz: float,
) -> SweepDiagnostic:
    resolved_frequency_hz = frequency_hz if event.frequency_hz is None else event.frequency_hz
    return SweepDiagnostic(
        code=event.code,
        severity=event.severity.value,
        message=event.message,
        suggested_action=event.suggested_action,
        solver_stage=event.solver_stage.value,
        point_index=point_index,
        frequency_hz=resolved_frequency_hz,
        element_id=event.element_id or "sweep_engine",
        frequency_index=event.frequency_index,
        sweep_index=event.sweep_index,
        witness=event.witness,
    )

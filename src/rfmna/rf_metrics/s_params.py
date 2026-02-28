from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags, eye  # type: ignore[import-untyped]

from rfmna.diagnostics import DiagnosticEvent, PortContext, Severity, SolverStage, sort_diagnostics
from rfmna.solver import FallbackRunConfig, SolveResult, solve_linear_system
from rfmna.solver.backend import SparseComplexMatrix

from .y_params import YParameterResult
from .z_params import ZParameterResult

type SPointStatus = Literal["pass", "degraded", "fail"]
type SConversionSource = Literal["from_z", "from_y"]

_S_PARAM_ELEMENT_ID = "rf_s_params"
_S_PAYLOAD_RANK = 3
_SINGULAR_CODE = "E_NUM_S_CONVERSION_SINGULAR"
_S_COMPLEX_Z0_CODE = "E_MODEL_PORT_Z0_COMPLEX"
_S_NONPOSITIVE_Z0_CODE = "E_MODEL_PORT_Z0_NONPOSITIVE"
_S_DEFAULT_Z0_OHM = 50.0


@runtime_checkable
class SSolvePointFn(Protocol):
    def __call__(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
    ) -> SolveResult: ...


@dataclass(frozen=True, slots=True)
class SParameterResult:
    frequencies_hz: NDArray[np.float64]
    port_ids: tuple[str, ...]
    s: NDArray[np.complex128]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...]
    conversion_source: SConversionSource

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        s_values = np.asarray(self.s, dtype=np.complex128)
        status_values = np.asarray(self.status, dtype=np.str_)

        if frequencies.ndim != 1:
            raise ValueError("frequencies_hz must be one-dimensional")
        if s_values.ndim != _S_PAYLOAD_RANK:
            raise ValueError("s must be rank-3 with shape [n_points, n_ports, n_ports]")
        if status_values.ndim != 1:
            raise ValueError("status must be one-dimensional")

        n_points = int(frequencies.shape[0])
        n_ports = len(self.port_ids)
        if s_values.shape != (n_points, n_ports, n_ports):
            raise ValueError("s shape must match [n_points, n_ports, n_ports]")
        if status_values.shape != (n_points,):
            raise ValueError("status shape must match [n_points]")
        if len(self.diagnostics_by_point) != n_points:
            raise ValueError("diagnostics_by_point length must equal n_points")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "s", s_values)
        object.__setattr__(self, "status", status_values)
        object.__setattr__(
            self,
            "diagnostics_by_point",
            tuple(tuple(sort_diagnostics(point)) for point in self.diagnostics_by_point),
        )


def convert_z_to_s(
    z_result: ZParameterResult,
    *,
    z0_ohm: object = _S_DEFAULT_Z0_OHM,
    solve_point: SSolvePointFn | None = None,
) -> SParameterResult:
    return _convert_to_s(
        frequencies=np.asarray(z_result.frequencies_hz, dtype=np.float64),
        port_ids=tuple(z_result.port_ids),
        matrix=np.asarray(z_result.z, dtype=np.complex128),
        status=np.asarray(z_result.status, dtype=np.str_),
        diagnostics_by_point=z_result.diagnostics_by_point,
        conversion_source="from_z",
        z0_ohm=z0_ohm,
        solve_point=solve_point,
    )


def convert_y_to_s(
    y_result: YParameterResult,
    *,
    z0_ohm: object = _S_DEFAULT_Z0_OHM,
    solve_point: SSolvePointFn | None = None,
) -> SParameterResult:
    return _convert_to_s(
        frequencies=np.asarray(y_result.frequencies_hz, dtype=np.float64),
        port_ids=tuple(y_result.port_ids),
        matrix=np.asarray(y_result.y, dtype=np.complex128),
        status=np.asarray(y_result.status, dtype=np.str_),
        diagnostics_by_point=y_result.diagnostics_by_point,
        conversion_source="from_y",
        z0_ohm=z0_ohm,
        solve_point=solve_point,
    )


def _convert_to_s(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    frequencies: NDArray[np.float64],
    port_ids: tuple[str, ...],
    matrix: NDArray[np.complex128],
    status: NDArray[np.str_],
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...],
    conversion_source: SConversionSource,
    z0_ohm: object,
    solve_point: SSolvePointFn | None,
) -> SParameterResult:
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional")
    if not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz entries must be finite")

    if matrix.ndim != _S_PAYLOAD_RANK:
        raise ValueError("input matrix must be rank-3 with shape [n_points, n_ports, n_ports]")
    n_points = int(frequencies.shape[0])
    n_ports = len(port_ids)
    if matrix.shape != (n_points, n_ports, n_ports):
        raise ValueError("input matrix shape must match [n_points, n_ports, n_ports]")
    if status.shape != (n_points,):
        raise ValueError("status shape must match [n_points]")
    if len(diagnostics_by_point) != n_points:
        raise ValueError("diagnostics_by_point length must equal n_points")
    if n_ports not in (1, 2):
        raise ValueError("Phase 1 S conversion supports exactly 1 or 2 ports")

    solver = solve_point if solve_point is not None else _default_conversion_solve_point()
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    s_values = np.full((n_points, n_ports, n_ports), complex_nan, dtype=np.complex128)
    output_status = np.full(n_points, "fail", dtype=np.dtype("<U8"))
    diagnostics_lists: list[list[DiagnosticEvent]] = [list(point) for point in diagnostics_by_point]

    z0_values, z0_failures = _validate_z0(
        z0_ohm=z0_ohm,
        n_ports=n_ports,
        port_ids=port_ids,
    )
    if z0_failures:
        for point_index, frequency in enumerate(frequencies):
            frequency_hz = float(frequency)
            output_status[point_index] = "fail"
            for failure in z0_failures:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=failure.code,
                        message=failure.message,
                        suggested_action="set port z0_ohm to a finite real value > 0",
                        stage=SolverStage.ASSEMBLE,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        port_id=failure.port_id,
                        witness=failure.witness,
                    )
                )
        return SParameterResult(
            frequencies_hz=frequencies,
            port_ids=port_ids,
            s=s_values,
            status=output_status,
            diagnostics_by_point=tuple(
                tuple(sort_diagnostics(point)) for point in diagnostics_lists
            ),
            conversion_source=conversion_source,
        )
    assert z0_values is not None

    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        point_status = str(status[point_index])
        if point_status not in {"pass", "degraded", "fail"}:
            raise ValueError(f"status[{point_index}] must be one of: pass, degraded, fail")
        if point_status == "fail":
            output_status[point_index] = "fail"
            if not diagnostics_lists[point_index]:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_SINGULAR_CODE,
                        message="S conversion skipped because upstream point status was fail",
                        suggested_action="inspect upstream Y/Z extraction diagnostics for this point",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "conversion_source": conversion_source,
                            "reason": "upstream_point_failed",
                        },
                    )
                )
            continue

        matrix_point = np.asarray(matrix[point_index, :, :], dtype=np.complex128)
        if conversion_source == "from_z":
            numerator, denominator = _z_formula_terms(
                matrix_point=matrix_point, z0_values=z0_values
            )
        else:
            numerator, denominator = _y_formula_terms(
                matrix_point=matrix_point, z0_values=z0_values
            )

        point_s = np.full((n_ports, n_ports), complex_nan, dtype=np.complex128)
        conversion_failed = False
        output_point_status: SPointStatus = "degraded" if point_status == "degraded" else "pass"

        for column_index in range(n_ports):
            rhs = np.asarray(numerator[:, column_index], dtype=np.complex128)
            try:
                solve_result = solver(denominator, rhs)
            except Exception as exc:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_SINGULAR_CODE,
                        message="S conversion failed due singular conversion matrix",
                        suggested_action="adjust Z0 or inspect Y/Z conditioning for this point",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "conversion_source": conversion_source,
                            "exception_type": type(exc).__name__,
                        },
                    )
                )
                conversion_failed = True
                output_point_status = "fail"
                break

            if _conversion_attempt_uses_gmin(solve_result):
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_SINGULAR_CODE,
                        message="S conversion rejected: gmin regularization is forbidden in conversion math",
                        suggested_action="use default conversion solver controls without gmin regularization",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "conversion_source": conversion_source,
                            "reason": "gmin_regularization_forbidden",
                        },
                    )
                )
                conversion_failed = True
                output_point_status = "fail"
                break

            if solve_result.status == "fail" or solve_result.x is None:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=_SINGULAR_CODE,
                        message="S conversion failed due singular conversion matrix",
                        suggested_action="adjust Z0 or inspect Y/Z conditioning for this point",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "column_index": column_index,
                            "cond_ind": solve_result.cond_ind,
                            "conversion_source": conversion_source,
                            "failure_category": solve_result.failure_category,
                            "failure_code": solve_result.failure_code,
                            "failure_reason": solve_result.failure_reason,
                        },
                    )
                )
                conversion_failed = True
                output_point_status = "fail"
                break

            if solve_result.status == "degraded" and output_point_status == "pass":
                output_point_status = "degraded"
            diagnostics_lists[point_index].extend(
                _mapped_solver_warnings(
                    solve_result=solve_result,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    column_index=column_index,
                    conversion_source=conversion_source,
                )
            )

            point_s[:, column_index] = np.asarray(solve_result.x, dtype=np.complex128)

        if conversion_failed:
            output_status[point_index] = "fail"
            continue

        s_values[point_index, :, :] = point_s
        output_status[point_index] = output_point_status

    return SParameterResult(
        frequencies_hz=frequencies,
        port_ids=port_ids,
        s=s_values,
        status=output_status,
        diagnostics_by_point=tuple(tuple(sort_diagnostics(point)) for point in diagnostics_lists),
        conversion_source=conversion_source,
    )


def _z_formula_terms(
    *,
    matrix_point: NDArray[np.complex128],
    z0_values: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], SparseComplexMatrix]:
    n_ports = int(matrix_point.shape[0])
    numerator = np.asarray(matrix_point, dtype=np.complex128).copy()
    numerator[np.diag_indices(n_ports)] -= z0_values.astype(np.complex128)
    denominator = csc_matrix(np.asarray(matrix_point, dtype=np.complex128)) + diags(
        z0_values.astype(np.complex128),
        offsets=0,
        shape=(n_ports, n_ports),
        format="csc",
    )
    return (numerator, denominator)


def _y_formula_terms(
    *,
    matrix_point: NDArray[np.complex128],
    z0_values: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], SparseComplexMatrix]:
    n_ports = int(matrix_point.shape[0])
    y_dense = np.asarray(matrix_point, dtype=np.complex128)
    z0_complex = z0_values.astype(np.complex128)
    z0_diag = diags(z0_complex, offsets=0, shape=(n_ports, n_ports), format="csc")
    y_sparse = csc_matrix(y_dense)

    numerator = -(z0_complex.reshape(n_ports, 1) * y_dense)
    numerator[np.diag_indices(n_ports)] += 1.0 + 0.0j

    denominator = (z0_diag @ y_sparse) + eye(n_ports, dtype=np.complex128, format="csc")
    return (numerator, denominator)


@dataclass(frozen=True, slots=True)
class _Z0Failure:
    code: str
    message: str
    port_id: str | None
    witness: dict[str, object]


def _default_conversion_solve_point() -> SSolvePointFn:
    conversion_run_config = FallbackRunConfig(enable_gmin=False)

    def _solve(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        return solve_linear_system(A, b, run_config=conversion_run_config)

    return _solve


def _mapped_solver_warnings(  # noqa: PLR0913
    *,
    solve_result: SolveResult,
    point_index: int,
    frequency_hz: float,
    column_index: int,
    conversion_source: SConversionSource,
) -> tuple[DiagnosticEvent, ...]:
    mapped: list[DiagnosticEvent] = []
    for warning in solve_result.warnings:
        mapped.append(
            DiagnosticEvent(
                code=warning.code,
                severity=Severity.WARNING,
                message=warning.message,
                suggested_action="review warning and conversion matrix conditioning",
                solver_stage=SolverStage.SOLVE,
                element_id=_S_PARAM_ELEMENT_ID,
                frequency_hz=frequency_hz,
                frequency_index=point_index,
                witness={
                    "column_index": column_index,
                    "conversion_source": conversion_source,
                    "warning_code": warning.code,
                },
            )
        )
    return tuple(sort_diagnostics(mapped))


def _conversion_attempt_uses_gmin(solve_result: SolveResult) -> bool:
    return any(
        row.stage == "gmin" and row.stage_state == "run" for row in solve_result.attempt_trace
    )


def _validate_z0(
    *,
    z0_ohm: object,
    n_ports: int,
    port_ids: tuple[str, ...],
) -> tuple[NDArray[np.float64] | None, tuple[_Z0Failure, ...]]:
    values, z0_source = _z0_values(z0_ohm=z0_ohm, n_ports=n_ports)
    failures: list[_Z0Failure] = []
    normalized = np.zeros(n_ports, dtype=np.float64)

    for index, raw_value in enumerate(values):
        port_id = port_ids[index] if z0_source == "vector" else None
        try:
            complex_value = complex(cast(complex | float | int | str, raw_value))
        except TypeError, ValueError:
            failures.append(
                _Z0Failure(
                    code=_S_NONPOSITIVE_Z0_CODE,
                    message="port z0_ohm must be finite and > 0",
                    port_id=port_id,
                    witness={
                        "z0_index": index if z0_source == "vector" else None,
                        "z0_source": z0_source,
                        "z0_value_type": type(raw_value).__name__,
                    },
                )
            )
            continue

        if complex_value.imag != 0.0:
            failures.append(
                _Z0Failure(
                    code=_S_COMPLEX_Z0_CODE,
                    message="port z0_ohm must be real-valued",
                    port_id=port_id,
                    witness={
                        "z0_index": index if z0_source == "vector" else None,
                        "z0_source": z0_source,
                        "z0_value": {
                            "imag": float(complex_value.imag),
                            "real": float(complex_value.real),
                        },
                    },
                )
            )
            continue

        value = float(complex_value.real)
        if not np.isfinite(value) or value <= 0.0:
            failures.append(
                _Z0Failure(
                    code=_S_NONPOSITIVE_Z0_CODE,
                    message="port z0_ohm must be finite and > 0",
                    port_id=port_id,
                    witness={
                        "z0_index": index if z0_source == "vector" else None,
                        "z0_source": z0_source,
                        "z0_value": value,
                    },
                )
            )
            continue
        normalized[index] = value

    if failures:
        return (None, tuple(failures))

    if z0_source == "scalar":
        return (np.full(n_ports, normalized[0], dtype=np.float64), ())
    return (normalized, ())


def _z0_values(
    *,
    z0_ohm: object,
    n_ports: int,
) -> tuple[tuple[object, ...], Literal["scalar", "vector"]]:
    if np.isscalar(z0_ohm):
        return ((z0_ohm,), "scalar")

    vector = np.asarray(z0_ohm)
    if vector.ndim == 0:
        return ((vector.item(),), "scalar")
    if vector.ndim != 1:
        raise ValueError("z0_ohm must be scalar or one-dimensional")
    if int(vector.shape[0]) != n_ports:
        raise ValueError("z0_ohm vector length must equal number of ports")
    return (tuple(vector.tolist()), "vector")


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
        element_id=_S_PARAM_ELEMENT_ID,
        port_context=PortContext(port_id=port_id) if port_id is not None else None,
        frequency_hz=frequency_hz,
        frequency_index=point_index,
        witness=witness,
    )

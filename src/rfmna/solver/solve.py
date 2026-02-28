from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import isspmatrix  # type: ignore[import-untyped]

from .backend import (
    BackendMetadata,
    SciPySparseBackend,
    SolverBackend,
    SparseComplexMatrix,
)
from .fallback import (
    AttemptTraceRecord,
    ConditionEstimator,
    FallbackRunConfig,
    SolverThresholdConfig,
    SolveWarning,
    load_solver_threshold_config,
    run_fallback_ladder,
)

_DEFAULT_THRESHOLDS = load_solver_threshold_config()
EPSILON = _DEFAULT_THRESHOLDS.residual.epsilon
PASS_MAX = _DEFAULT_THRESHOLDS.residual.pass_max
DEGRADED_MAX = _DEFAULT_THRESHOLDS.residual.degraded_max

SolveStatus = Literal["pass", "degraded", "fail"]


@dataclass(frozen=True, slots=True)
class ResidualMetrics:
    res_l2: float
    res_linf: float
    res_rel: float


@dataclass(frozen=True, slots=True)
class SolveResult:
    x: NDArray[np.complex128] | None
    residual: ResidualMetrics
    status: SolveStatus
    backend: BackendMetadata
    failure_category: str | None
    failure_code: str | None
    failure_reason: str | None
    failure_message: str | None
    cond_ind: float
    warnings: tuple[SolveWarning, ...]
    attempt_trace: tuple[AttemptTraceRecord, ...]


def compute_residual_metrics(
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    x: NDArray[np.complex128],
    *,
    epsilon: float = EPSILON,
) -> ResidualMetrics:
    residual = (A @ x) - b
    res_l2 = _vector_l2_norm(residual)
    res_linf = _vector_inf_norm(residual)
    a_linf = _matrix_inf_norm(A)
    x_linf = _vector_inf_norm(x)
    b_linf = _vector_inf_norm(b)
    denominator = (a_linf * x_linf) + b_linf + epsilon
    res_rel = res_linf / denominator
    return ResidualMetrics(res_l2=res_l2, res_linf=res_linf, res_rel=res_rel)


def classify_status(res_rel: float) -> SolveStatus:
    if res_rel <= PASS_MAX:
        return "pass"
    if res_rel <= DEGRADED_MAX:
        return "degraded"
    return "fail"


def solve_linear_system(  # noqa: PLR0913
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    *,
    backend: SolverBackend | None = None,
    node_voltage_count: int | None = None,
    run_config: FallbackRunConfig | None = None,
    thresholds: SolverThresholdConfig | None = None,
    condition_estimator: ConditionEstimator | None = None,
) -> SolveResult:
    if not isspmatrix(A):
        raise TypeError("matrix A must be a SciPy sparse matrix")

    solver_backend = backend if backend is not None else SciPySparseBackend()
    vector = np.asarray(b, dtype=np.complex128)
    ladder_result = run_fallback_ladder(
        A,
        vector,
        backend=solver_backend,
        node_voltage_count=node_voltage_count,
        run_config=run_config,
        thresholds=thresholds,
        condition_estimator=condition_estimator,
    )
    return SolveResult(
        x=ladder_result.x,
        residual=ResidualMetrics(
            res_l2=ladder_result.res_l2,
            res_linf=ladder_result.res_linf,
            res_rel=ladder_result.res_rel,
        ),
        status=ladder_result.status,
        backend=ladder_result.backend,
        failure_category=ladder_result.failure_category,
        failure_code=ladder_result.failure_code,
        failure_reason=ladder_result.failure_reason,
        failure_message=ladder_result.failure_message,
        cond_ind=ladder_result.cond_ind,
        warnings=ladder_result.warnings,
        attempt_trace=ladder_result.attempt_trace,
    )


def _vector_l2_norm(vector: NDArray[np.complex128]) -> float:
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector, ord=2))


def _vector_inf_norm(vector: NDArray[np.complex128]) -> float:
    if vector.size == 0:
        return 0.0
    return float(np.max(np.abs(vector)))


def _matrix_inf_norm(matrix: SparseComplexMatrix) -> float:
    row_abs_sums = np.asarray(np.abs(matrix).sum(axis=1), dtype=np.float64).reshape(-1)
    if row_abs_sums.size == 0:
        return 0.0
    return float(np.max(row_abs_sums))

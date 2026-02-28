from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags, isspmatrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import splu  # type: ignore[import-untyped]

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - import availability is environment-specific
    yaml = None

from .backend import (
    BackendMetadata,
    BackendNotes,
    BackendSolveOptions,
    SciPySparseBackend,
    SolverBackend,
    SparseComplexMatrix,
)

type SolveStatus = Literal["pass", "degraded", "fail"]
type AttemptStage = Literal["baseline", "alt_pivot", "scaling", "gmin", "final_fail"]
type StageState = Literal["run", "skipped"]

DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "docs/spec/thresholds_v4_0_0.yaml"
EXPECTED_COND_ESTIMATOR_ID = "lu_rcond_proxy_v1"
_FALLBACK_ORDER_TOKEN_MAP: dict[str, AttemptStage] = {
    "baseline": "baseline",
    "alt_permutation_or_pivot": "alt_pivot",
    "scaling_enabled_retry": "scaling",
    "gmin_ladder_retry": "gmin",
    "final_fail_bundle": "final_fail",
}


class SolverConfigError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ResidualThresholds:
    epsilon: float
    pass_max: float
    degraded_max: float
    fail_min_exclusive: float


@dataclass(frozen=True, slots=True)
class ConditioningThresholds:
    estimator_id: str
    unavailable_warning_code: str
    ill_conditioned_warning_code: str
    warn_max: float
    fail_max: float


@dataclass(frozen=True, slots=True)
class SolverThresholdConfig:
    residual: ResidualThresholds
    conditioning: ConditioningThresholds
    scaling_enabled_by_default: bool
    gmin_values: tuple[float, ...]
    fallback_order: tuple[AttemptStage, ...]
    artifact_path: str


@dataclass(frozen=True, slots=True)
class FallbackRunConfig:
    enable_alt_pivot: bool = True
    enable_scaling: bool = True
    enable_gmin: bool = True
    gmin_values_override: tuple[float, ...] | None = None
    thresholds_artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class SolveWarning:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AttemptTraceRecord:
    attempt_index: int
    stage: AttemptStage
    stage_state: StageState
    scaling_enabled: bool
    pivot_profile: str
    gmin_value: float | None
    success: bool
    backend_failure_category: str | None
    backend_failure_reason: str | None
    backend_failure_code: str | None
    backend_failure_message: str | None
    res_l2: float
    res_linf: float
    res_rel: float
    status: SolveStatus | None
    skip_reason: str | None


@dataclass(frozen=True, slots=True)
class FallbackSolveResult:
    x: NDArray[np.complex128] | None
    res_l2: float
    res_linf: float
    res_rel: float
    status: SolveStatus
    backend: BackendMetadata
    failure_category: str | None
    failure_code: str | None
    failure_reason: str | None
    failure_message: str | None
    cond_ind: float
    warnings: tuple[SolveWarning, ...]
    attempt_trace: tuple[AttemptTraceRecord, ...]


@runtime_checkable
class ConditionEstimator(Protocol):
    def estimate(self, A: SparseComplexMatrix) -> float | None: ...


@dataclass(frozen=True, slots=True)
class LuRcondProxyEstimator:
    def estimate(self, A: SparseComplexMatrix) -> float | None:
        if not isspmatrix(A):
            return None
        try:
            lu = splu(A.tocsc())
        except Exception:
            return None
        diagonal = np.asarray(lu.U.diagonal(), dtype=np.complex128)
        if diagonal.size == 0:
            return 1.0
        magnitudes = np.abs(diagonal)
        if not np.isfinite(magnitudes).all():
            return None
        max_mag = float(np.max(magnitudes))
        min_mag = float(np.min(magnitudes))
        if max_mag <= 0.0:
            return 0.0
        return min_mag / max_mag


@dataclass(frozen=True, slots=True)
class _FailureInfo:
    category: str | None
    code: str | None
    reason: str | None
    message: str | None


@dataclass(frozen=True, slots=True)
class _SuccessfulAttempt:
    x: NDArray[np.complex128]
    res_l2: float
    res_linf: float
    res_rel: float
    status: SolveStatus
    backend: BackendMetadata
    matrix: SparseComplexMatrix


@dataclass(frozen=True, slots=True)
class _StagePlan:
    stage: AttemptStage
    enabled: bool
    options: BackendSolveOptions
    gmin_value: float | None
    skip_reason: str | None
    matrix_factory: Callable[[], SparseComplexMatrix]


def load_solver_threshold_config(path: str | Path | None = None) -> SolverThresholdConfig:
    selected_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    return _load_solver_threshold_config_cached(str(selected_path.resolve()))


@cache
def _load_solver_threshold_config_cached(path: str) -> SolverThresholdConfig:
    target = Path(path)
    raw = _read_yaml_file(target)
    numeric_contract = _require_mapping(raw, "numeric_contract")
    residual_block = _require_mapping(numeric_contract, "residual")
    residual_bands = _require_mapping(residual_block, "status_bands")
    residual = ResidualThresholds(
        epsilon=_require_float(residual_block, "relative_epsilon"),
        pass_max=_require_float(residual_bands, "pass_max"),
        degraded_max=_require_float(residual_bands, "degraded_max"),
        fail_min_exclusive=_require_float(residual_bands, "fail_min_exclusive"),
    )
    _validate_residual_thresholds(residual)

    condition_block = _require_mapping(numeric_contract, "condition_indicator")
    estimator_block = _require_mapping(condition_block, "estimator")
    unavailable_policy = _require_mapping(estimator_block, "unavailable_policy")
    condition_bands = _require_mapping(condition_block, "bands")
    conditioning = ConditioningThresholds(
        estimator_id=_require_string(estimator_block, "id"),
        unavailable_warning_code=_require_string(unavailable_policy, "warning_code"),
        ill_conditioned_warning_code=_require_string(condition_bands, "warning_code"),
        warn_max=_require_float(condition_bands, "warn_max"),
        fail_max=_require_float(condition_bands, "fail_max"),
    )

    solver_defaults = _require_mapping(raw, "solver_defaults")
    scaling_block = _require_mapping(solver_defaults, "scaling")
    gmin_block = _require_mapping(solver_defaults, "gmin_ladder")
    fallback_ladder_block = _require_mapping(solver_defaults, "fallback_ladder")
    return SolverThresholdConfig(
        residual=residual,
        conditioning=conditioning,
        scaling_enabled_by_default=_require_bool(scaling_block, "enabled_by_default"),
        gmin_values=_require_float_tuple(gmin_block, "values"),
        fallback_order=_require_fallback_order(fallback_ladder_block, "order"),
        artifact_path=str(target),
    )


def apply_gmin_shunt(
    A: SparseComplexMatrix,
    node_voltage_count: int,
    gmin: float,
) -> csc_matrix:
    if not isspmatrix(A):
        raise TypeError("matrix A must be a SciPy sparse matrix")
    if node_voltage_count < 0:
        raise ValueError("node_voltage_count must be >= 0")
    if gmin < 0.0 or not np.isfinite(gmin):
        raise ValueError("gmin must be finite and >= 0")

    matrix = A.tocsc()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix A must be square")
    size = int(matrix.shape[0])
    if node_voltage_count > size:
        raise ValueError("node_voltage_count exceeds matrix dimension")

    diagonal = np.zeros(size, dtype=np.complex128)
    diagonal[:node_voltage_count] = np.complex128(gmin + 0.0j)
    shunt = diags(diagonal, offsets=0, shape=matrix.shape, format="csc", dtype=np.complex128)
    return cast(csc_matrix, matrix + shunt)


def run_fallback_ladder(  # noqa: PLR0913, PLR0915
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    *,
    backend: SolverBackend | None = None,
    node_voltage_count: int | None = None,
    run_config: FallbackRunConfig | None = None,
    thresholds: SolverThresholdConfig | None = None,
    condition_estimator: ConditionEstimator | None = None,
) -> FallbackSolveResult:
    if not isspmatrix(A):
        raise TypeError("matrix A must be a SciPy sparse matrix")

    vector = np.asarray(b, dtype=np.complex128)
    base_matrix = A.copy()
    config = run_config if run_config is not None else FallbackRunConfig()
    threshold_config = _resolve_threshold_config(config, thresholds)
    gmin_values = (
        config.gmin_values_override
        if config.gmin_values_override is not None
        else threshold_config.gmin_values
    )
    _validate_gmin_values(gmin_values)
    _validate_node_voltage_count(node_voltage_count, int(base_matrix.shape[0]))
    estimator = _resolve_condition_estimator(threshold_config, condition_estimator)
    solver_backend = backend if backend is not None else SciPySparseBackend()

    plans = _build_stage_plans(
        base_matrix=base_matrix,
        threshold_config=threshold_config,
        run_config=config,
        gmin_values=gmin_values,
        node_voltage_count=node_voltage_count,
    )

    trace: list[AttemptTraceRecord] = []
    solved: _SuccessfulAttempt | None = None
    last_backend_metadata: BackendMetadata | None = None
    last_failure = _FailureInfo(None, None, None, None)
    attempt_index = 0

    for plan in plans:
        if solved is not None:
            trace.append(_skipped_attempt(attempt_index, plan, "prior_stage_succeeded"))
            attempt_index += 1
            continue
        if not plan.enabled:
            trace.append(
                _skipped_attempt(attempt_index, plan, plan.skip_reason or "stage_disabled")
            )
            attempt_index += 1
            continue

        matrix = plan.matrix_factory()
        backend_result = solver_backend.solve(matrix, vector, options=plan.options)
        metadata = backend_result.metadata
        last_backend_metadata = metadata
        if metadata.success and backend_result.x is not None:
            res_l2, res_linf, res_rel = _compute_residual_metrics(
                matrix, vector, backend_result.x, threshold_config.residual.epsilon
            )
            status = _classify_status(
                res_rel,
                threshold_config.residual.pass_max,
                threshold_config.residual.degraded_max,
                threshold_config.residual.fail_min_exclusive,
            )
            solved = _SuccessfulAttempt(
                x=backend_result.x,
                res_l2=res_l2,
                res_linf=res_linf,
                res_rel=res_rel,
                status=status,
                backend=metadata,
                matrix=matrix,
            )
            trace.append(
                AttemptTraceRecord(
                    attempt_index=attempt_index,
                    stage=plan.stage,
                    stage_state="run",
                    scaling_enabled=plan.options.scaling_enabled,
                    pivot_profile=plan.options.pivot_profile,
                    gmin_value=plan.gmin_value,
                    success=True,
                    backend_failure_category=None,
                    backend_failure_reason=None,
                    backend_failure_code=None,
                    backend_failure_message=None,
                    res_l2=res_l2,
                    res_linf=res_linf,
                    res_rel=res_rel,
                    status=status,
                    skip_reason=None,
                )
            )
        else:
            last_failure = _FailureInfo(
                category=metadata.failure_category,
                code=metadata.failure_code,
                reason=metadata.failure_reason,
                message=metadata.failure_message,
            )
            trace.append(
                AttemptTraceRecord(
                    attempt_index=attempt_index,
                    stage=plan.stage,
                    stage_state="run",
                    scaling_enabled=plan.options.scaling_enabled,
                    pivot_profile=plan.options.pivot_profile,
                    gmin_value=plan.gmin_value,
                    success=False,
                    backend_failure_category=metadata.failure_category,
                    backend_failure_reason=metadata.failure_reason,
                    backend_failure_code=metadata.failure_code,
                    backend_failure_message=metadata.failure_message,
                    res_l2=float("nan"),
                    res_linf=float("nan"),
                    res_rel=float("nan"),
                    status=None,
                    skip_reason=None,
                )
            )
        attempt_index += 1

    trace.append(
        AttemptTraceRecord(
            attempt_index=attempt_index,
            stage="final_fail",
            stage_state="run" if solved is None else "skipped",
            scaling_enabled=False,
            pivot_profile="none",
            gmin_value=None,
            success=False,
            backend_failure_category=None,
            backend_failure_reason=None,
            backend_failure_code=None,
            backend_failure_message=None,
            res_l2=float("nan"),
            res_linf=float("nan"),
            res_rel=float("nan"),
            status=None,
            skip_reason="prior_stage_succeeded" if solved is not None else None,
        )
    )

    condition_matrix = solved.matrix if solved is not None else base_matrix
    cond_raw = estimator.estimate(condition_matrix)
    warnings: list[SolveWarning] = []
    if cond_raw is None or not np.isfinite(cond_raw):
        cond_ind = float("nan")
        warnings.append(
            SolveWarning(
                code=threshold_config.conditioning.unavailable_warning_code,
                message="condition indicator unavailable",
            )
        )
    else:
        cond_ind = float(cond_raw)
        if cond_ind <= threshold_config.conditioning.fail_max:
            solved = _force_condition_failure(
                solved=solved,
            )
        elif cond_ind <= threshold_config.conditioning.warn_max:
            warnings.append(
                SolveWarning(
                    code=threshold_config.conditioning.ill_conditioned_warning_code,
                    message="condition indicator is in ill-conditioned warning band",
                )
            )

    if solved is not None:
        failure = _failure_from_status(
            solved.res_rel,
            solved.status,
            threshold_config.residual.degraded_max,
            cond_ind=cond_ind,
            cond_fail_max=threshold_config.conditioning.fail_max,
        )
        return FallbackSolveResult(
            x=solved.x,
            res_l2=solved.res_l2,
            res_linf=solved.res_linf,
            res_rel=solved.res_rel,
            status=solved.status,
            backend=solved.backend,
            failure_category=failure.category,
            failure_code=failure.code,
            failure_reason=failure.reason,
            failure_message=failure.message,
            cond_ind=cond_ind,
            warnings=tuple(warnings),
            attempt_trace=tuple(trace),
        )

    backend_metadata = (
        last_backend_metadata
        if last_backend_metadata is not None
        else _default_backend_failure_metadata()
    )
    fallback_failure = (
        last_failure
        if last_failure.code is not None
        else _FailureInfo(
            category="numeric",
            code="E_NUM_SOLVE_FAILED",
            reason="all_attempts_failed",
            message="all configured solve attempts failed",
        )
    )
    nan = float("nan")
    return FallbackSolveResult(
        x=None,
        res_l2=nan,
        res_linf=nan,
        res_rel=nan,
        status="fail",
        backend=backend_metadata,
        failure_category=fallback_failure.category,
        failure_code=fallback_failure.code,
        failure_reason=fallback_failure.reason,
        failure_message=fallback_failure.message,
        cond_ind=cond_ind,
        warnings=tuple(warnings),
        attempt_trace=tuple(trace),
    )


def _build_stage_plans(
    *,
    base_matrix: SparseComplexMatrix,
    threshold_config: SolverThresholdConfig,
    run_config: FallbackRunConfig,
    gmin_values: tuple[float, ...],
    node_voltage_count: int | None,
) -> tuple[_StagePlan, ...]:
    default_scaling = threshold_config.scaling_enabled_by_default
    base_factory = _constant_matrix_factory(base_matrix)
    plans: list[_StagePlan] = []
    for stage in threshold_config.fallback_order:
        if stage == "baseline":
            plans.append(
                _StagePlan(
                    stage="baseline",
                    enabled=True,
                    options=BackendSolveOptions(
                        pivot_profile="default",
                        scaling_enabled=default_scaling,
                    ),
                    gmin_value=None,
                    skip_reason=None,
                    matrix_factory=base_factory,
                )
            )
            continue

        if stage == "alt_pivot":
            plans.append(
                _StagePlan(
                    stage="alt_pivot",
                    enabled=run_config.enable_alt_pivot,
                    options=BackendSolveOptions(
                        pivot_profile="alt",
                        scaling_enabled=default_scaling,
                    ),
                    gmin_value=None,
                    skip_reason="stage_disabled" if not run_config.enable_alt_pivot else None,
                    matrix_factory=base_factory,
                )
            )
            continue

        if stage == "scaling":
            plans.append(
                _StagePlan(
                    stage="scaling",
                    enabled=run_config.enable_scaling,
                    options=BackendSolveOptions(
                        pivot_profile="default",
                        scaling_enabled=True,
                    ),
                    gmin_value=None,
                    skip_reason="stage_disabled" if not run_config.enable_scaling else None,
                    matrix_factory=base_factory,
                )
            )
            continue

        if stage == "gmin":
            for value in gmin_values:
                gmin_enabled = run_config.enable_gmin and node_voltage_count is not None
                if not run_config.enable_gmin:
                    skip_reason = "gmin_stage_disabled"
                elif node_voltage_count is None:
                    skip_reason = "node_voltage_count_unavailable"
                else:
                    skip_reason = None
                plans.append(
                    _StagePlan(
                        stage="gmin",
                        enabled=gmin_enabled,
                        options=BackendSolveOptions(
                            pivot_profile="default",
                            scaling_enabled=default_scaling,
                        ),
                        gmin_value=value,
                        skip_reason=skip_reason,
                        matrix_factory=_gmin_matrix_factory(
                            base_matrix, node_voltage_count or 0, value
                        ),
                    )
                )
            continue

        if stage == "final_fail":
            continue

        raise SolverConfigError("E_SOLVER_CONFIG_INVALID", f"unsupported fallback stage '{stage}'")
    return tuple(plans)


def _constant_matrix_factory(matrix: SparseComplexMatrix) -> Callable[[], SparseComplexMatrix]:
    def _factory() -> SparseComplexMatrix:
        return matrix

    return _factory


def _gmin_matrix_factory(
    matrix: SparseComplexMatrix,
    node_voltage_count: int,
    gmin_value: float,
) -> Callable[[], SparseComplexMatrix]:
    def _factory() -> SparseComplexMatrix:
        return apply_gmin_shunt(matrix, node_voltage_count, gmin_value)

    return _factory


def _skipped_attempt(attempt_index: int, plan: _StagePlan, reason: str) -> AttemptTraceRecord:
    return AttemptTraceRecord(
        attempt_index=attempt_index,
        stage=plan.stage,
        stage_state="skipped",
        scaling_enabled=plan.options.scaling_enabled,
        pivot_profile=plan.options.pivot_profile,
        gmin_value=plan.gmin_value,
        success=False,
        backend_failure_category=None,
        backend_failure_reason=None,
        backend_failure_code=None,
        backend_failure_message=None,
        res_l2=float("nan"),
        res_linf=float("nan"),
        res_rel=float("nan"),
        status=None,
        skip_reason=reason,
    )


def _compute_residual_metrics(
    A: SparseComplexMatrix,
    b: NDArray[np.complex128],
    x: NDArray[np.complex128],
    epsilon: float,
) -> tuple[float, float, float]:
    residual = (A @ x) - b
    res_l2 = _vector_l2_norm(residual)
    res_linf = _vector_inf_norm(residual)
    a_linf = _matrix_inf_norm(A)
    x_linf = _vector_inf_norm(x)
    b_linf = _vector_inf_norm(b)
    denominator = (a_linf * x_linf) + b_linf + epsilon
    return res_l2, res_linf, res_linf / denominator


def _classify_status(
    res_rel: float,
    pass_max: float,
    degraded_max: float,
    fail_min_exclusive: float,
) -> SolveStatus:
    if res_rel <= pass_max:
        return "pass"
    if res_rel <= degraded_max:
        return "degraded"
    if res_rel > fail_min_exclusive:
        return "fail"
    raise SolverConfigError(
        "E_SOLVER_CONFIG_INVALID",
        "residual status bands leave an undefined interval between degraded_max and fail_min_exclusive",
    )


def _validate_residual_thresholds(residual: ResidualThresholds) -> None:
    if residual.pass_max > residual.degraded_max:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            "residual pass_max must be <= degraded_max",
        )
    if residual.fail_min_exclusive != residual.degraded_max:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            "residual fail_min_exclusive must equal degraded_max under v4 frozen status-band semantics",
        )


def _failure_from_status(
    res_rel: float,
    status: SolveStatus,
    degraded_max: float,
    *,
    cond_ind: float,
    cond_fail_max: float,
) -> _FailureInfo:
    if status != "fail":
        return _FailureInfo(None, None, None, None)
    if np.isfinite(cond_ind) and cond_ind <= cond_fail_max and res_rel <= degraded_max:
        return _FailureInfo(
            category="numeric",
            code="E_NUM_SOLVE_FAILED",
            reason="condition_indicator_below_fail_threshold",
            message=f"condition indicator {cond_ind:.3e} is <= fail_max {cond_fail_max:.1e}",
        )
    return _FailureInfo(
        category="numeric",
        code="E_NUM_SOLVE_FAILED",
        reason="residual_above_threshold",
        message=f"relative residual {res_rel:.3e} exceeded {degraded_max:.1e}",
    )


def _force_condition_failure(
    *,
    solved: _SuccessfulAttempt | None,
) -> _SuccessfulAttempt:
    if solved is None:
        raise ValueError("condition-based failure requires successful solve attempt")
    if solved.status == "fail":
        return solved
    return _SuccessfulAttempt(
        x=solved.x,
        res_l2=solved.res_l2,
        res_linf=solved.res_linf,
        res_rel=solved.res_rel,
        status="fail",
        backend=solved.backend,
        matrix=solved.matrix,
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


def _resolve_threshold_config(
    run_config: FallbackRunConfig,
    thresholds: SolverThresholdConfig | None,
) -> SolverThresholdConfig:
    if thresholds is not None:
        return thresholds
    return load_solver_threshold_config(run_config.thresholds_artifact_path)


def _resolve_condition_estimator(
    threshold_config: SolverThresholdConfig,
    condition_estimator: ConditionEstimator | None,
) -> ConditionEstimator:
    if condition_estimator is not None:
        return condition_estimator
    if threshold_config.conditioning.estimator_id != EXPECTED_COND_ESTIMATOR_ID:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            f"unsupported condition estimator id '{threshold_config.conditioning.estimator_id}'",
        )
    return LuRcondProxyEstimator()


def _validate_gmin_values(values: tuple[float, ...]) -> None:
    if not values:
        raise SolverConfigError("E_SOLVER_CONFIG_INVALID", "gmin ladder values must be non-empty")
    for index, value in enumerate(values):
        if not np.isfinite(value) or value < 0.0:
            raise SolverConfigError(
                "E_SOLVER_CONFIG_INVALID",
                f"gmin ladder value at index {index} must be finite and >= 0",
            )


def _validate_node_voltage_count(node_voltage_count: int | None, matrix_dim: int) -> None:
    if node_voltage_count is None:
        return
    if node_voltage_count < 0 or node_voltage_count > matrix_dim:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            "node_voltage_count must be within [0, n_unknowns]",
        )


def _default_backend_failure_metadata() -> BackendMetadata:
    return BackendMetadata(
        backend_id="unknown_backend",
        backend_name="unknown_backend",
        python_version=None,
        scipy_version=None,
        permutation_row=None,
        permutation_col=None,
        pivot_order=None,
        success=False,
        failure_category="numeric",
        failure_code="E_NUM_SOLVE_FAILED",
        failure_reason="all_attempts_failed",
        failure_message="all configured solve attempts failed",
        notes=BackendNotes(
            input_format="unknown",
            normalized_format="unknown",
            n_unknowns=0,
            nnz=0,
            pivot_profile="unknown",
            scaling_enabled=False,
        ),
    )


def _read_yaml_file(path: Path) -> dict[str, object]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_READ_FAILED",
            f"unable to read solver thresholds artifact '{path}': {exc}",
        ) from exc
    if yaml is None:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_PARSE_FAILED",
            "yaml parser unavailable for solver thresholds artifact",
        )
    try:
        payload = yaml.safe_load(raw_text)
    except Exception as exc:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_PARSE_FAILED",
            f"invalid solver thresholds yaml in '{path}': {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            "solver thresholds artifact root must be a mapping",
        )
    return cast(dict[str, object], payload)


def _require_mapping(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    raise SolverConfigError(
        "E_SOLVER_CONFIG_INVALID", f"missing or invalid mapping for key '{key}'"
    )


def _require_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str) and value:
        return value
    raise SolverConfigError("E_SOLVER_CONFIG_INVALID", f"missing or invalid string for key '{key}'")


def _require_bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    raise SolverConfigError("E_SOLVER_CONFIG_INVALID", f"missing or invalid bool for key '{key}'")


def _require_float(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise SolverConfigError("E_SOLVER_CONFIG_INVALID", f"invalid numeric value for key '{key}'")
    if isinstance(value, int | float):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
    raise SolverConfigError("E_SOLVER_CONFIG_INVALID", f"missing or invalid float for key '{key}'")


def _require_float_tuple(data: dict[str, object], key: str) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID", f"missing or invalid list for key '{key}'"
        )
    out: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise SolverConfigError(
                "E_SOLVER_CONFIG_INVALID", f"invalid gmin value at index {index}"
            )
        numeric = float(item)
        if not np.isfinite(numeric):
            raise SolverConfigError(
                "E_SOLVER_CONFIG_INVALID", f"invalid gmin value at index {index}"
            )
        out.append(numeric)
    return tuple(out)


def _require_fallback_order(data: dict[str, object], key: str) -> tuple[AttemptStage, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID", f"missing or invalid list for key '{key}'"
        )

    mapped: list[AttemptStage] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise SolverConfigError(
                "E_SOLVER_CONFIG_INVALID",
                f"invalid fallback order token at index {index}",
            )
        stage = _FALLBACK_ORDER_TOKEN_MAP.get(item)
        if stage is None:
            raise SolverConfigError(
                "E_SOLVER_CONFIG_INVALID",
                f"unsupported fallback order token '{item}'",
            )
        mapped.append(stage)

    if len(mapped) != len(set(mapped)):
        raise SolverConfigError("E_SOLVER_CONFIG_INVALID", "fallback order tokens must be unique")
    required = {"baseline", "alt_pivot", "scaling", "gmin", "final_fail"}
    if set(mapped) != required:
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID",
            "fallback order must include baseline, alt, scaling, gmin, and final_fail exactly once",
        )
    if mapped[-1] != "final_fail":
        raise SolverConfigError(
            "E_SOLVER_CONFIG_INVALID", "final_fail must be the last fallback order stage"
        )
    return tuple(mapped)

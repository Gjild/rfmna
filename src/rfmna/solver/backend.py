from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import scipy  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags, isspmatrix, spmatrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import splu as _scipy_splu  # type: ignore[import-untyped]

type SparseComplexMatrix = Any

_SCALING_MODE_NONE = "none"
_SCALING_MODE_ROW_COL_MAX_ABS = "row_col_max_abs_v1"


@dataclass(frozen=True, slots=True)
class _PivotProfileConfig:
    profile: str
    permc_spec: str
    diag_pivot_thresh: float
    pivot_mode: str


_PIVOT_PROFILE_CONFIGS: dict[str, _PivotProfileConfig] = {
    "default": _PivotProfileConfig(
        profile="default",
        permc_spec="COLAMD",
        diag_pivot_thresh=1.0,
        pivot_mode="superlu_colamd_diag1p0_v1",
    ),
    "alt": _PivotProfileConfig(
        profile="alt",
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.01,
        pivot_mode="superlu_mmd_at_plus_a_diag0p01_v1",
    ),
}


@dataclass(frozen=True, slots=True)
class BackendSolveOptions:
    pivot_profile: str = "default"
    scaling_enabled: bool = False


@dataclass(frozen=True, slots=True)
class BackendNotes:
    input_format: str
    normalized_format: str
    n_unknowns: int
    nnz: int
    pivot_profile: str
    scaling_enabled: bool
    effective_pivot_profile: str = "unknown"
    effective_scaling_enabled: bool = False
    pivot_mode: str = "unknown"
    scaling_mode: str = _SCALING_MODE_NONE


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    backend_id: str
    backend_name: str
    python_version: str | None
    scipy_version: str | None
    permutation_row: tuple[int, ...] | None
    permutation_col: tuple[int, ...] | None
    pivot_order: tuple[int, ...] | None
    success: bool
    failure_category: str | None
    failure_code: str | None
    failure_reason: str | None
    failure_message: str | None
    notes: BackendNotes


@dataclass(frozen=True, slots=True)
class BackendSolveResult:
    x: NDArray[np.complex128] | None
    metadata: BackendMetadata


@dataclass(frozen=True, slots=True)
class _PreparedSolveInputs:
    matrix: csc_matrix
    vector: NDArray[np.complex128]
    undo_column_scaling: NDArray[np.float64] | None
    pivot_config: _PivotProfileConfig
    notes: BackendNotes


@dataclass(frozen=True, slots=True)
class _RawSolveArtifacts:
    raw_x: object
    permutation_row: tuple[int, ...]
    permutation_col: tuple[int, ...]
    pivot_order: tuple[int, ...]


@runtime_checkable
class SolverBackend(Protocol):
    def solve(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
        *,
        options: BackendSolveOptions | None = None,
    ) -> BackendSolveResult: ...


@dataclass(frozen=True, slots=True)
class SciPySparseBackend:
    backend_id: str = "scipy_sparse_lu"
    backend_name: str = "SciPy sparse SuperLU"

    def solve(
        self,
        A: SparseComplexMatrix,
        b: NDArray[np.complex128],
        *,
        options: BackendSolveOptions | None = None,
    ) -> BackendSolveResult:
        solve_options = options if options is not None else BackendSolveOptions()
        vector = np.asarray(b, dtype=np.complex128)
        notes = BackendNotes(
            input_format=_sparse_format_name(A),
            normalized_format="csc",
            n_unknowns=int(vector.shape[0]) if vector.ndim == 1 else -1,
            nnz=int(getattr(A, "nnz", -1)),
            pivot_profile=str(solve_options.pivot_profile),
            scaling_enabled=bool(solve_options.scaling_enabled),
        )
        input_failure = self._validate_inputs(A, vector, notes)
        if input_failure is not None:
            return input_failure

        assert isspmatrix(A)
        normalized_A = A if isinstance(A, csc_matrix) else A.tocsc()
        prepared_inputs, prepare_failure = self._prepare_solve_inputs(
            matrix=normalized_A,
            vector=vector,
            notes=notes,
        )
        if prepare_failure is not None:
            return prepare_failure
        assert prepared_inputs is not None

        solve_artifacts, solver_failure = self._run_solver(
            prepared_inputs.matrix,
            prepared_inputs.vector,
            prepared_inputs.pivot_config,
            prepared_inputs.notes,
        )
        if solver_failure is not None:
            return solver_failure
        assert solve_artifacts is not None

        recovered_x = self._recover_solution(
            solve_artifacts.raw_x,
            prepared_inputs.undo_column_scaling,
        )
        x, solution_failure = self._normalize_solution(
            recovered_x,
            vector.shape[0],
            prepared_inputs.notes,
        )
        if solution_failure is not None:
            return solution_failure
        assert x is not None

        metadata = BackendMetadata(
            backend_id=self.backend_id,
            backend_name=self.backend_name,
            python_version=platform.python_version(),
            scipy_version=scipy.__version__,
            permutation_row=solve_artifacts.permutation_row,
            permutation_col=solve_artifacts.permutation_col,
            pivot_order=solve_artifacts.pivot_order,
            success=True,
            failure_category=None,
            failure_code=None,
            failure_reason=None,
            failure_message=None,
            notes=prepared_inputs.notes,
        )
        return BackendSolveResult(x=x, metadata=metadata)

    def _prepare_solve_inputs(
        self,
        *,
        matrix: csc_matrix,
        vector: NDArray[np.complex128],
        notes: BackendNotes,
    ) -> tuple[_PreparedSolveInputs | None, BackendSolveResult | None]:
        pivot_config, pivot_failure = self._resolve_pivot_profile(notes, notes.pivot_profile)
        if pivot_failure is not None:
            return (None, pivot_failure)
        assert pivot_config is not None

        normalized_format = _sparse_format_name(matrix)
        if notes.scaling_enabled:
            scaled_matrix, scaled_vector, undo_column_scaling = _apply_row_col_max_abs_scaling(
                matrix,
                vector,
            )
            scaling_mode = _SCALING_MODE_ROW_COL_MAX_ABS
            effective_scaling_enabled = True
        else:
            scaled_matrix = matrix
            scaled_vector = vector
            undo_column_scaling = None
            scaling_mode = _SCALING_MODE_NONE
            effective_scaling_enabled = False

        prepared_notes = BackendNotes(
            input_format=notes.input_format,
            normalized_format=normalized_format,
            n_unknowns=notes.n_unknowns,
            nnz=int(scaled_matrix.nnz),
            pivot_profile=notes.pivot_profile,
            scaling_enabled=notes.scaling_enabled,
            effective_pivot_profile=pivot_config.profile,
            effective_scaling_enabled=effective_scaling_enabled,
            pivot_mode=pivot_config.pivot_mode,
            scaling_mode=scaling_mode,
        )
        return (
            _PreparedSolveInputs(
                matrix=scaled_matrix,
                vector=scaled_vector,
                undo_column_scaling=undo_column_scaling,
                pivot_config=pivot_config,
                notes=prepared_notes,
            ),
            None,
        )

    def _resolve_pivot_profile(
        self,
        notes: BackendNotes,
        profile: str,
    ) -> tuple[_PivotProfileConfig | None, BackendSolveResult | None]:
        pivot_config = _PIVOT_PROFILE_CONFIGS.get(profile)
        if pivot_config is None:
            supported = ", ".join(sorted(_PIVOT_PROFILE_CONFIGS))
            return (
                None,
                self._failure(
                    notes=notes,
                    category="input",
                    code="E_NUM_SOLVE_FAILED",
                    reason="unsupported_pivot_profile",
                    message=f"unsupported pivot_profile '{profile}'; expected one of {supported}",
                ),
            )
        return (pivot_config, None)

    def _validate_inputs(
        self,
        A: SparseComplexMatrix,
        vector: NDArray[np.complex128],
        notes: BackendNotes,
    ) -> BackendSolveResult | None:
        if vector.ndim != 1:
            return self._failure(
                notes=notes,
                category="input",
                code="E_NUM_SOLVE_FAILED",
                reason="rhs_vector_rank_invalid",
                message="rhs vector must be rank-1",
            )
        if not isspmatrix(A):
            return self._failure(
                notes=notes,
                category="input",
                code="E_NUM_SOLVE_FAILED",
                reason="matrix_not_sparse",
                message="matrix input must be a SciPy sparse matrix",
            )
        if A.shape[0] != A.shape[1] or A.shape[0] != vector.shape[0]:
            return self._failure(
                notes=notes,
                category="input",
                code="E_NUM_SOLVE_FAILED",
                reason="dimension_mismatch",
                message="matrix must be square and match rhs vector length",
            )
        return None

    def _run_solver(
        self,
        A: csc_matrix,
        vector: NDArray[np.complex128],
        pivot_config: _PivotProfileConfig,
        notes: BackendNotes,
    ) -> tuple[_RawSolveArtifacts | None, BackendSolveResult | None]:
        try:
            lu = _scipy_splu(
                A,
                permc_spec=pivot_config.permc_spec,
                diag_pivot_thresh=pivot_config.diag_pivot_thresh,
            )
            raw_x = lu.solve(vector)
        except Exception as exc:  # pragma: no cover - deterministic mapping path
            category, reason = _classify_backend_exception(exc)
            code = "E_NUM_SINGULAR_MATRIX" if reason == "singular_matrix" else "E_NUM_SOLVE_FAILED"
            return (
                None,
                self._failure(
                    notes=notes,
                    category=category,
                    code=code,
                    reason=reason,
                    message=str(exc),
                ),
            )
        return (
            _RawSolveArtifacts(
                raw_x=raw_x,
                permutation_row=tuple(int(item) for item in np.asarray(lu.perm_r).tolist()),
                permutation_col=tuple(int(item) for item in np.asarray(lu.perm_c).tolist()),
                # SuperLU does not expose pivot sequence directly; retain deterministic row pivot map.
                pivot_order=tuple(int(item) for item in np.asarray(lu.perm_r).tolist()),
            ),
            None,
        )

    def _recover_solution(
        self,
        raw_x: object,
        undo_column_scaling: NDArray[np.float64] | None,
    ) -> object:
        if undo_column_scaling is None:
            return raw_x
        scaled_solution = np.asarray(raw_x, dtype=np.complex128)
        return scaled_solution * undo_column_scaling.astype(np.complex128, copy=False)

    def _normalize_solution(
        self,
        raw_x: object,
        expected_size: int,
        notes: BackendNotes,
    ) -> tuple[NDArray[np.complex128] | None, BackendSolveResult | None]:
        x = np.asarray(raw_x, dtype=np.complex128)
        if x.ndim == 0:
            x = x.reshape((1,))
        if x.ndim != 1 or x.shape[0] != expected_size:
            return (
                None,
                self._failure(
                    notes=notes,
                    category="backend",
                    code="E_NUM_SOLVE_FAILED",
                    reason="solution_shape_invalid",
                    message="backend solution payload had unexpected shape",
                ),
            )
        if not np.isfinite(x.real).all() or not np.isfinite(x.imag).all():
            return (
                None,
                self._failure(
                    notes=notes,
                    category="numeric",
                    code="E_NUM_SOLVE_FAILED",
                    reason="solution_non_finite",
                    message="backend returned non-finite solution values",
                ),
            )
        return (x, None)

    def _failure(
        self,
        *,
        notes: BackendNotes,
        category: str,
        code: str,
        reason: str,
        message: str,
    ) -> BackendSolveResult:
        metadata = BackendMetadata(
            backend_id=self.backend_id,
            backend_name=self.backend_name,
            python_version=platform.python_version(),
            scipy_version=scipy.__version__,
            permutation_row=None,
            permutation_col=None,
            pivot_order=None,
            success=False,
            failure_category=category,
            failure_code=code,
            failure_reason=reason,
            failure_message=message,
            notes=notes,
        )
        return BackendSolveResult(x=None, metadata=metadata)


def _sparse_format_name(matrix: object) -> str:
    if not isspmatrix(matrix):
        return "not_sparse"
    return str(cast(spmatrix, matrix).getformat())


def _apply_row_col_max_abs_scaling(
    A: csc_matrix,
    b: NDArray[np.complex128],
) -> tuple[csc_matrix, NDArray[np.complex128], NDArray[np.float64]]:
    abs_matrix = np.abs(A)
    row_max = np.asarray(abs_matrix.max(axis=1).toarray(), dtype=np.float64).reshape(-1)
    row_scale = _safe_reciprocal_scale(row_max)
    row_scaled_matrix = cast(
        csc_matrix,
        diags(row_scale, offsets=0, shape=A.shape, format="csc", dtype=np.float64) @ A,
    )
    row_scaled_vector = b * row_scale.astype(np.complex128, copy=False)

    col_max = np.asarray(np.abs(row_scaled_matrix).max(axis=0).toarray(), dtype=np.float64).reshape(
        -1
    )
    col_scale = _safe_reciprocal_scale(col_max)
    fully_scaled_matrix = cast(
        csc_matrix,
        row_scaled_matrix
        @ diags(col_scale, offsets=0, shape=A.shape, format="csc", dtype=np.float64),
    )
    return (
        fully_scaled_matrix,
        row_scaled_vector,
        col_scale,
    )


def _safe_reciprocal_scale(max_abs_values: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = np.ones(max_abs_values.shape[0], dtype=np.float64)
    positive = np.isfinite(max_abs_values) & (max_abs_values > 0.0)
    if not np.any(positive):
        return scale

    positive_indices = np.flatnonzero(positive)
    inverse = 1.0 / max_abs_values[positive]
    finite = np.isfinite(inverse) & (inverse > 0.0)
    if not np.any(finite):
        return scale

    scale[positive_indices[finite]] = inverse[finite]
    return scale


def _classify_backend_exception(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ValueError):
        return ("input", "invalid_argument")
    if isinstance(exc, RuntimeError):
        # SuperLU factorization failures are surfaced as RuntimeError; keep a stable
        # singular classification independent of backend message text.
        return ("numeric", "singular_matrix")
    return ("backend", "unexpected_exception")

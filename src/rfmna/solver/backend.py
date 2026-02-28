from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import scipy  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, isspmatrix, spmatrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import MatrixRankWarning  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve as _scipy_spsolve

type SparseComplexMatrix = Any


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
    backend_name: str = "SciPy sparse spsolve"

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
            pivot_profile=solve_options.pivot_profile,
            scaling_enabled=solve_options.scaling_enabled,
        )
        input_failure = self._validate_inputs(A, vector, notes)
        if input_failure is not None:
            return input_failure

        assert isspmatrix(A)
        normalized_A = A if isinstance(A, csc_matrix) else A.tocsc()
        normalized_notes = BackendNotes(
            input_format=notes.input_format,
            normalized_format=_sparse_format_name(normalized_A),
            n_unknowns=notes.n_unknowns,
            nnz=notes.nnz,
            pivot_profile=notes.pivot_profile,
            scaling_enabled=notes.scaling_enabled,
        )
        raw_x, solver_failure = self._run_solver(normalized_A, vector, normalized_notes)
        if solver_failure is not None:
            return solver_failure
        x, solution_failure = self._normalize_solution(raw_x, vector.shape[0], normalized_notes)
        if solution_failure is not None:
            return solution_failure
        assert x is not None

        metadata = BackendMetadata(
            backend_id=self.backend_id,
            backend_name=self.backend_name,
            python_version=platform.python_version(),
            scipy_version=scipy.__version__,
            permutation_row=None,
            permutation_col=None,
            pivot_order=None,
            success=True,
            failure_category=None,
            failure_code=None,
            failure_reason=None,
            failure_message=None,
            notes=normalized_notes,
        )
        return BackendSolveResult(x=x, metadata=metadata)

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
        notes: BackendNotes,
    ) -> tuple[object, BackendSolveResult | None]:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", MatrixRankWarning)
                raw_x = _scipy_spsolve(A, vector)
            if any(issubclass(item.category, MatrixRankWarning) for item in caught_warnings):
                return (
                    None,
                    self._failure(
                        notes=notes,
                        category="numeric",
                        code="E_NUM_SINGULAR_MATRIX",
                        reason="singular_matrix",
                        message="sparse solve reported singular matrix",
                    ),
                )
        except Exception as exc:  # pragma: no cover - deterministic mapping path
            category, reason = _classify_backend_exception(exc)
            return (
                None,
                self._failure(
                    notes=notes,
                    category=category,
                    code="E_NUM_SOLVE_FAILED",
                    reason=reason,
                    message=str(exc),
                ),
            )
        return (raw_x, None)

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


def _classify_backend_exception(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ValueError):
        return ("input", "invalid_argument")
    if isinstance(exc, RuntimeError):
        return ("backend", "runtime_error")
    return ("backend", "unexpected_exception")

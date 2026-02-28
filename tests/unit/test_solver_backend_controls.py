from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import BackendSolveOptions, SciPySparseBackend
from rfmna.solver import backend as backend_module

pytestmark = pytest.mark.unit
EXPECTED_CALL_COUNT = 2


class _FakeLu:
    def __init__(
        self, *, perm_r: NDArray[np.int64], perm_c: NDArray[np.int64], value: complex
    ) -> None:
        self.perm_r = perm_r
        self.perm_c = perm_c
        self._value = np.complex128(value)

    def solve(self, vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.full(vector.shape[0], fill_value=self._value, dtype=np.complex128)


def test_backend_controls_materially_drive_superlu_configuration_and_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[NDArray[np.complex128], str, float, NDArray[np.complex128]]] = []

    def fake_splu(
        matrix: csr_matrix,
        *,
        permc_spec: str,
        diag_pivot_thresh: float,
    ) -> _FakeLu:
        dense = np.asarray(matrix.toarray(), dtype=np.complex128)
        rhs = np.asarray(current_rhs[0], dtype=np.complex128).copy()
        calls.append((dense, permc_spec, float(diag_pivot_thresh), rhs))
        size = int(matrix.shape[0])
        perm = np.arange(size, dtype=np.int64)
        if permc_spec == "MMD_AT_PLUS_A":
            perm = perm[::-1]
        return _FakeLu(perm_r=perm, perm_c=perm[::-1], value=1.0 + 0.0j)

    current_rhs: list[NDArray[np.complex128]] = []

    original_run_solver = SciPySparseBackend._run_solver

    def wrapped_run_solver(
        self: SciPySparseBackend,
        A: csr_matrix,
        vector: NDArray[np.complex128],
        pivot_config: object,
        notes: object,
    ) -> tuple[object | None, object | None]:
        current_rhs[:] = [np.asarray(vector, dtype=np.complex128)]
        return original_run_solver(self, A, vector, pivot_config, notes)

    monkeypatch.setattr(backend_module, "_scipy_splu", fake_splu)
    monkeypatch.setattr(SciPySparseBackend, "_run_solver", wrapped_run_solver)

    backend = SciPySparseBackend()
    matrix = csr_matrix(
        np.asarray(
            [
                [1.0e-6 + 0.0j, 5.0 + 0.0j],
                [2.0e3 + 0.0j, 1.0 + 0.0j],
            ],
            dtype=np.complex128,
        )
    )
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    default_result = backend.solve(
        matrix,
        rhs,
        options=BackendSolveOptions(pivot_profile="default", scaling_enabled=False),
    )
    alt_result = backend.solve(
        matrix,
        rhs,
        options=BackendSolveOptions(pivot_profile="alt", scaling_enabled=True),
    )

    assert len(calls) == EXPECTED_CALL_COUNT
    assert calls[0][1] == "COLAMD"
    assert calls[0][2] == pytest.approx(1.0)
    assert calls[1][1] == "MMD_AT_PLUS_A"
    assert calls[1][2] == pytest.approx(0.01)
    assert np.allclose(calls[0][0], matrix.toarray())
    assert not np.allclose(calls[1][0], matrix.toarray())
    assert not np.allclose(calls[1][3], rhs)

    assert default_result.metadata.notes.effective_pivot_profile == "default"
    assert default_result.metadata.notes.effective_scaling_enabled is False
    assert default_result.metadata.notes.scaling_mode == "none"
    assert alt_result.metadata.notes.effective_pivot_profile == "alt"
    assert alt_result.metadata.notes.effective_scaling_enabled is True
    assert alt_result.metadata.notes.scaling_mode == "row_col_max_abs_v1"
    assert alt_result.metadata.permutation_row == (1, 0)
    assert alt_result.metadata.permutation_col == (0, 1)


def test_backend_rejects_unknown_pivot_profile_with_deterministic_error() -> None:
    backend = SciPySparseBackend()
    matrix = csr_matrix(np.eye(2, dtype=np.complex128))
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    result = backend.solve(
        matrix,
        rhs,
        options=BackendSolveOptions(pivot_profile="not_supported", scaling_enabled=False),
    )

    assert result.x is None
    assert result.metadata.success is False
    assert result.metadata.failure_code == "E_NUM_SOLVE_FAILED"
    assert result.metadata.failure_reason == "unsupported_pivot_profile"
    assert result.metadata.notes.pivot_profile == "not_supported"
    assert result.metadata.notes.effective_pivot_profile == "unknown"


def test_backend_runtime_error_classification_is_not_message_dependent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_splu(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("backend factorization failed")

    monkeypatch.setattr(backend_module, "_scipy_splu", fake_splu)
    backend = SciPySparseBackend()
    matrix = csr_matrix(np.eye(2, dtype=np.complex128))
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    result = backend.solve(
        matrix,
        rhs,
        options=BackendSolveOptions(pivot_profile="default", scaling_enabled=False),
    )

    assert result.x is None
    assert result.metadata.success is False
    assert result.metadata.failure_code == "E_NUM_SINGULAR_MATRIX"
    assert result.metadata.failure_reason == "singular_matrix"


def test_backend_metadata_repeatability_with_real_controls() -> None:
    backend = SciPySparseBackend()
    matrix = csr_matrix(
        np.asarray(
            [
                [4.0 + 0.0j, -1.0 + 2.0j, 0.0 + 0.0j],
                [3.0 - 1.0j, 2.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 1.0 - 1.0j, 5.0 + 0.0j],
            ],
            dtype=np.complex128,
        )
    )
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 1.0j, -0.5 + 0.0j], dtype=np.complex128)
    options = BackendSolveOptions(pivot_profile="alt", scaling_enabled=True)

    first = backend.solve(matrix, rhs, options=options)
    second = backend.solve(matrix, rhs, options=options)

    assert first.metadata.success is True
    assert second.metadata.success is True
    assert first.x is not None and second.x is not None
    np.testing.assert_allclose(first.x, second.x, rtol=1e-12, atol=1e-12)
    assert first.metadata.permutation_row == second.metadata.permutation_row
    assert first.metadata.permutation_col == second.metadata.permutation_col
    assert first.metadata.pivot_order == second.metadata.pivot_order
    assert first.metadata.notes.effective_pivot_profile == "alt"
    assert second.metadata.notes.effective_pivot_profile == "alt"
    assert first.metadata.notes.effective_scaling_enabled is True
    assert second.metadata.notes.effective_scaling_enabled is True

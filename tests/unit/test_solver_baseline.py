from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import (  # type: ignore[import-untyped]
    coo_matrix,
    csc_matrix,
    csr_matrix,
    issparse,
    spmatrix,
)

from rfmna.solver import (
    DEGRADED_MAX,
    EPSILON,
    PASS_MAX,
    SciPySparseBackend,
    classify_status,
    compute_residual_metrics,
    solve_linear_system,
)
from rfmna.solver import backend as backend_module

pytestmark = pytest.mark.unit


def _as_complex(values: list[complex]) -> NDArray[np.complex128]:
    return np.asarray(values, dtype=np.complex128)


def test_nominal_complex_sparse_solve_pass_status() -> None:
    expected_x = _as_complex([1.0 + 2.0j, -0.5 + 0.25j])
    A = csr_matrix(
        np.asarray(
            [
                [3.0 + 1.0j, 1.0 - 1.0j],
                [2.0 + 0.0j, 4.0 + 2.0j],
            ],
            dtype=np.complex128,
        )
    )
    b = A @ expected_x

    result = solve_linear_system(A, b)

    assert result.x is not None
    np.testing.assert_allclose(result.x, expected_x, atol=1e-12, rtol=1e-12)
    assert result.status == "pass"
    assert result.residual.res_rel <= PASS_MAX


def test_residual_formula_and_matrix_inf_norm_and_epsilon_path() -> None:
    A_dense = np.asarray(
        [
            [1.0 + 0.0j, -2.0 + 0.0j],
            [3.0 + 4.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    A = coo_matrix(A_dense)
    x = _as_complex([1.0 - 1.0j, 2.0 + 0.5j])
    b = _as_complex([0.5 + 0.5j, -1.0 + 2.0j])
    metrics = compute_residual_metrics(A, b, x)

    residual = (A_dense @ x) - b
    expected_l2 = float(np.linalg.norm(residual, ord=2))
    expected_linf = float(np.max(np.abs(residual)))
    matrix_inf_norm = float(np.max(np.sum(np.abs(A_dense), axis=1)))
    elementwise_abs_max = float(np.max(np.abs(A_dense)))
    assert matrix_inf_norm != elementwise_abs_max
    denominator = matrix_inf_norm * float(np.max(np.abs(x))) + float(np.max(np.abs(b))) + EPSILON
    expected_rel = expected_linf / denominator

    assert metrics.res_l2 == pytest.approx(expected_l2)
    assert metrics.res_linf == pytest.approx(expected_linf)
    assert metrics.res_rel == pytest.approx(expected_rel)

    zero = np.zeros((1, 1), dtype=np.complex128)
    zero_metrics = compute_residual_metrics(
        csr_matrix(zero),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )
    assert zero_metrics.res_rel == 0.0
    assert math.isfinite(zero_metrics.res_rel)


def test_status_threshold_bands() -> None:
    assert classify_status(PASS_MAX) == "pass"
    assert classify_status(float(np.nextafter(PASS_MAX, np.inf))) == "degraded"
    assert classify_status(DEGRADED_MAX) == "degraded"
    assert classify_status(float(np.nextafter(DEGRADED_MAX, np.inf))) == "fail"


def test_backend_metadata_schema_presence_and_types() -> None:
    A = csr_matrix(np.asarray([[2.0 + 0.0j]], dtype=np.complex128))
    b = _as_complex([4.0 + 0.0j])
    result = solve_linear_system(A, b)
    metadata = result.backend
    payload = asdict(metadata)

    assert tuple(payload) == (
        "backend_id",
        "backend_name",
        "python_version",
        "scipy_version",
        "permutation_row",
        "permutation_col",
        "pivot_order",
        "success",
        "failure_category",
        "failure_code",
        "failure_reason",
        "failure_message",
        "notes",
    )
    assert isinstance(metadata.backend_id, str)
    assert isinstance(metadata.backend_name, str)
    assert isinstance(metadata.notes.n_unknowns, int)
    assert isinstance(metadata.notes.nnz, int)
    assert metadata.permutation_row == (0,)
    assert metadata.permutation_col == (0,)
    assert metadata.pivot_order == (0,)
    assert metadata.notes.effective_pivot_profile == "default"
    assert metadata.notes.effective_scaling_enabled is False
    assert metadata.notes.scaling_mode == "none"


def test_backend_failure_policy_a_returns_fail_with_placeholders() -> None:
    A = csr_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128))
    b = _as_complex([1.0 + 0.0j])

    result = solve_linear_system(A, b)

    assert result.status == "fail"
    assert result.x is None
    assert math.isnan(result.residual.res_l2)
    assert math.isnan(result.residual.res_linf)
    assert math.isnan(result.residual.res_rel)
    assert result.backend.success is False
    assert result.failure_category == "numeric"
    assert result.failure_code == "E_NUM_SINGULAR_MATRIX"
    assert result.failure_reason == "singular_matrix"
    assert isinstance(result.failure_message, str)


@pytest.mark.parametrize(
    "matrix_builder",
    [csr_matrix, csc_matrix, coo_matrix],
)
def test_sparse_inputs_and_sparse_only_backend_call(
    matrix_builder: Callable[[NDArray[np.complex128]], spmatrix],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    class _FakeLu:
        def __init__(self, size: int) -> None:
            self.perm_r = np.arange(size, dtype=np.int64)
            self.perm_c = np.arange(size, dtype=np.int64)

        def solve(self, vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return np.asarray(vector / 2.0, dtype=np.complex128)

    def fake_splu(
        matrix: csr_matrix | csc_matrix | coo_matrix,
        *,
        permc_spec: str,
        diag_pivot_thresh: float,
    ) -> _FakeLu:
        called["value"] = True
        assert issparse(matrix)
        assert permc_spec == "COLAMD"
        assert diag_pivot_thresh == pytest.approx(1.0)
        return _FakeLu(size=matrix.shape[0])

    monkeypatch.setattr(backend_module, "_scipy_splu", fake_splu)

    A = matrix_builder(np.asarray([[2.0 + 0.0j]], dtype=np.complex128))
    b = _as_complex([4.0 + 0.0j])
    result = solve_linear_system(A, b, backend=SciPySparseBackend())

    assert called["value"] is True
    assert result.x is not None
    np.testing.assert_allclose(result.x, _as_complex([2.0 + 0.0j]))


def test_repeatability_and_no_input_mutation() -> None:
    A = coo_matrix(
        np.asarray(
            [
                [5.0 + 0.0j, -1.0 + 0.0j],
                [0.0 + 0.0j, 2.0 + 3.0j],
            ],
            dtype=np.complex128,
        )
    )
    A_before = A.copy()
    b = _as_complex([4.0 + 1.0j, 2.0 - 3.0j])
    b_before = b.copy()

    first = solve_linear_system(A, b)
    second = solve_linear_system(A, b)

    assert first.status == second.status
    assert first.x is not None
    assert second.x is not None
    np.testing.assert_allclose(first.x, second.x)
    assert first.residual.res_l2 == pytest.approx(second.residual.res_l2)
    assert first.residual.res_linf == pytest.approx(second.residual.res_linf)
    assert first.residual.res_rel == pytest.approx(second.residual.res_rel)
    assert (A_before != A).nnz == 0
    np.testing.assert_array_equal(b, b_before)

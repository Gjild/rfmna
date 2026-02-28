from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import (
    DEGRADED_MAX,
    EPSILON,
    PASS_MAX,
    classify_status,
    compute_residual_metrics,
    load_solver_threshold_config,
)

pytestmark = pytest.mark.conformance

EXPECTED_EPSILON = 1.0e-30
EXPECTED_PASS_MAX = 1.0e-9
EXPECTED_DEGRADED_MAX = 1.0e-6
EXPECTED_FAIL_MIN_EXCLUSIVE = 1.0e-6


def test_residual_formula_uses_contract_epsilon_and_matrix_infinity_norm() -> None:
    A_dense = np.asarray(
        [
            [2.0 + 0.0j, -1.0 + 1.0j],
            [0.5 + 0.5j, 3.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    A = csr_matrix(A_dense)
    x = np.asarray([1.0 + 0.0j, -2.0 + 1.0j], dtype=np.complex128)
    b = np.asarray([0.0 + 0.0j, 1.0 + 1.0j], dtype=np.complex128)

    metrics = compute_residual_metrics(A, b, x)
    residual = (A_dense @ x) - b
    expected_linf = float(np.max(np.abs(residual)))
    expected_l2 = float(np.linalg.norm(residual, ord=2))
    matrix_inf_norm = float(np.max(np.sum(np.abs(A_dense), axis=1)))
    denominator = (
        matrix_inf_norm * float(np.max(np.abs(x))) + float(np.max(np.abs(b))) + EXPECTED_EPSILON
    )
    expected_rel = expected_linf / denominator

    assert EPSILON == EXPECTED_EPSILON
    assert metrics.res_l2 == pytest.approx(expected_l2)
    assert metrics.res_linf == pytest.approx(expected_linf)
    assert metrics.res_rel == pytest.approx(expected_rel)


def test_status_band_mapping_is_exact() -> None:
    defaults = load_solver_threshold_config()
    assert PASS_MAX == EXPECTED_PASS_MAX
    assert DEGRADED_MAX == EXPECTED_DEGRADED_MAX
    assert defaults.residual.fail_min_exclusive == EXPECTED_FAIL_MIN_EXCLUSIVE
    assert defaults.residual.fail_min_exclusive == defaults.residual.degraded_max
    assert classify_status(EXPECTED_PASS_MAX) == "pass"
    assert classify_status(float(np.nextafter(EXPECTED_PASS_MAX, np.inf))) == "degraded"
    assert classify_status(EXPECTED_DEGRADED_MAX) == "degraded"
    assert classify_status(float(np.nextafter(EXPECTED_DEGRADED_MAX, np.inf))) == "fail"


def test_denominator_zero_edge_is_defined() -> None:
    A = csr_matrix(np.zeros((1, 1), dtype=np.complex128))
    b = np.asarray([0.0 + 0.0j], dtype=np.complex128)
    x = np.asarray([0.0 + 0.0j], dtype=np.complex128)

    metrics = compute_residual_metrics(A, b, x)

    assert metrics.res_rel == 0.0
    assert np.isfinite(metrics.res_rel)

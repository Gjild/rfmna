from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import BackendSolveOptions, SciPySparseBackend, solve_linear_system
from rfmna.sweep_engine import SweepLayout, run_sweep

pytestmark = pytest.mark.conformance
EXPECTED_SWEEP_POINTS = 2

_CONTROL_FIXTURE_DENSE = np.asarray(
    [
        [
            2.6384438297663473e74 + 0.0j,
            2.86524699969777e74 + 3.0473171225094524e74j,
            0.0j,
            0.0j,
            0.0j,
            0.0j,
            0.0j,
        ],
        [
            0.0j,
            3.0836891654823143e-108 + 0.0j,
            0.0j,
            0.0j,
            0.0j,
            0.0j,
            -9.75096862874179e-109 - 6.237821402620139e-108j,
        ],
        [
            0.0j,
            0.0j,
            9.769176667392338e-100 + 0.0j,
            0.0j,
            2.584451326040648e-99 + 4.4922468020931236e-100j,
            0.0j,
            0.0j,
        ],
        [
            -2.2488239972286424e52 - 3.238077357096843e52j,
            4.710756390976206e52 + 5.82661770797455e52j,
            0.0j,
            1.0833170443504462e53 + 5.06313725425131e52j,
            0.0j,
            0.0j,
            0.0j,
        ],
        [
            0.0j,
            3.829380669833552e84 - 1.0388590676939656e85j,
            0.0j,
            0.0j,
            1.3444394228751206e85 + 0.0j,
            3.697899405268454e84 - 5.895062816138611e84j,
            5.524779066594446e84 + 1.4490430946115029e85j,
        ],
        [
            -7.948368175655053e31 - 1.9830383064662093e31j,
            -9.805323809570723e30 + 4.126512300406292e31j,
            0.0j,
            0.0j,
            0.0j,
            6.560458474933576e31 - 3.2548614248363053e31j,
            0.0j,
        ],
        [
            2.5500161608954387e-36 - 6.161324346773996e-36j,
            7.54685130656736e-36 + 1.4875716749913805e-36j,
            0.0j,
            0.0j,
            5.343800599882129e-36 - 7.256832192232989e-36j,
            0.0j,
            5.152501190408741e-36 + 0.0j,
        ],
    ],
    dtype=np.complex128,
)
_PERMUTATION_DENSE = np.asarray(
    [
        [4.0 + 0.0j, 1.0 - 2.0j, 0.0 + 0.0j, 0.5 + 0.0j],
        [0.2 + 0.1j, 3.0 + 0.0j, 1.0 + 0.3j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 - 0.4j, 2.0 + 0.0j, 0.2 + 0.0j],
        [0.0 + 0.5j, 0.0 + 0.0j, 0.1 + 0.0j, 1.5 + 0.0j],
    ],
    dtype=np.complex128,
)


@dataclass(frozen=True, slots=True)
class _ConstEstimator:
    value: float | None

    def estimate(self, A: object) -> float | None:
        del A
        return self.value


def test_controlled_fixture_recovers_on_alt_stage_with_deterministic_metadata() -> None:
    A = csr_matrix(_CONTROL_FIXTURE_DENSE)
    b = np.ones(7, dtype=np.complex128)

    result = solve_linear_system(
        A,
        b,
        node_voltage_count=7,
        condition_estimator=_ConstEstimator(0.5),
    )

    assert result.status == "pass"
    assert result.backend.success is True
    assert [row.stage for row in result.attempt_trace[:3]] == ["baseline", "alt_pivot", "scaling"]
    assert result.attempt_trace[0].stage_state == "run"
    assert result.attempt_trace[0].success is False
    assert result.attempt_trace[0].backend_failure_reason == "singular_matrix"
    assert result.attempt_trace[1].stage_state == "run"
    assert result.attempt_trace[1].success is True
    assert result.attempt_trace[1].pivot_profile == "alt"
    assert result.attempt_trace[2].stage_state == "skipped"
    assert result.attempt_trace[2].skip_reason == "prior_stage_succeeded"
    assert result.backend.notes.effective_pivot_profile == "alt"
    assert result.backend.notes.effective_scaling_enabled is False
    assert result.backend.notes.pivot_mode == "superlu_mmd_at_plus_a_diag0p01_v1"
    assert result.backend.notes.scaling_mode == "none"


def test_permutation_equivalent_inputs_remain_tolerance_stable_under_alt_scaling_controls() -> None:
    b = np.asarray([1.0 + 0.0j, 2.0 + 0.5j, -0.5 + 1.0j, 0.7 - 0.2j], dtype=np.complex128)
    perm = np.asarray([2, 0, 3, 1], dtype=np.int64)
    A_left = csr_matrix(_PERMUTATION_DENSE)
    A_right = csr_matrix(_PERMUTATION_DENSE[np.ix_(perm, perm)])
    b_right = b[perm]
    options = BackendSolveOptions(pivot_profile="alt", scaling_enabled=True)

    left = solve_linear_system(A_left, b, node_voltage_count=4)
    right_backend = SciPySparseBackend().solve(A_right, b_right, options=options)
    left_backend = SciPySparseBackend().solve(A_left, b, options=options)

    assert left.status == "pass"
    assert left_backend.x is not None and right_backend.x is not None
    mapped = np.empty_like(right_backend.x)
    mapped[perm] = right_backend.x
    np.testing.assert_allclose(mapped, left_backend.x, rtol=1e-10, atol=1e-10)
    assert left_backend.metadata.notes.effective_pivot_profile == "alt"
    assert right_backend.metadata.notes.effective_pivot_profile == "alt"
    assert left_backend.metadata.notes.effective_scaling_enabled is True
    assert right_backend.metadata.notes.effective_scaling_enabled is True


def test_sweep_fail_sentinel_policy_preserved_when_backend_controls_are_exercised() -> None:
    freq = np.asarray([1.0, 2.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=7, n_aux=0)

    def assemble_point(index: int, frequency_hz: float) -> tuple[csr_matrix, np.ndarray]:
        del frequency_hz
        if index == 0:
            return csr_matrix(_CONTROL_FIXTURE_DENSE), np.ones(7, dtype=np.complex128)
        return csr_matrix(np.zeros((7, 7), dtype=np.complex128)), np.ones(7, dtype=np.complex128)

    result = run_sweep(freq, layout, assemble_point)

    assert result.n_points == EXPECTED_SWEEP_POINTS
    assert result.status.tolist() == ["pass", "fail"]
    fail_index = 1
    assert math.isnan(result.V_nodes[fail_index, 0].real)
    assert math.isnan(result.V_nodes[fail_index, 0].imag)
    assert math.isnan(result.res_l2[fail_index])
    assert math.isnan(result.res_linf[fail_index])
    assert math.isnan(result.res_rel[fail_index])
    assert math.isnan(result.cond_ind[fail_index])
    assert result.diagnostics_by_point[fail_index]

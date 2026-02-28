from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import FallbackRunConfig, solve_linear_system

pytestmark = [
    pytest.mark.regression,
    pytest.mark.filterwarnings("ignore:overflow encountered in dot:RuntimeWarning"),
]

_SCALING_RECOVERY_DENSE = np.asarray(
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


@dataclass(frozen=True, slots=True)
class _ConstEstimator:
    value: float

    def estimate(self, A: object) -> float:
        del A
        return self.value


def test_scaling_stage_recovery_after_baseline_failure_is_stable() -> None:
    matrix = csr_matrix(_SCALING_RECOVERY_DENSE)
    rhs = np.ones(7, dtype=np.complex128)
    run_config = FallbackRunConfig(enable_alt_pivot=False, enable_scaling=True, enable_gmin=False)

    first = solve_linear_system(
        matrix,
        rhs,
        node_voltage_count=7,
        run_config=run_config,
        condition_estimator=_ConstEstimator(0.5),
    )
    second = solve_linear_system(
        matrix,
        rhs,
        node_voltage_count=7,
        run_config=run_config,
        condition_estimator=_ConstEstimator(0.5),
    )

    assert first.status == "pass"
    assert second.status == "pass"
    assert first.x is not None and second.x is not None
    np.testing.assert_allclose(first.x, second.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first.residual.res_rel, second.residual.res_rel, rtol=0.0, atol=0.0)

    first_trace = first.attempt_trace
    second_trace = second.attempt_trace
    assert [row.stage for row in first_trace[:3]] == ["baseline", "alt_pivot", "scaling"]
    assert first_trace[0].stage_state == "run"
    assert first_trace[0].success is False
    assert first_trace[0].backend_failure_reason == "singular_matrix"
    assert first_trace[1].stage_state == "skipped"
    assert first_trace[1].skip_reason == "stage_disabled"
    assert first_trace[2].stage_state == "run"
    assert first_trace[2].success is True
    assert first_trace[2].scaling_enabled is True
    assert first_trace[2].pivot_profile == "default"
    assert first_trace[-1].stage == "final_fail"
    assert first_trace[-1].stage_state == "skipped"

    assert first.backend.notes.effective_pivot_profile == "default"
    assert first.backend.notes.effective_scaling_enabled is True
    assert first.backend.notes.scaling_mode == "row_col_max_abs_v1"
    assert [row.stage_state for row in first_trace] == [row.stage_state for row in second_trace]
    assert [row.skip_reason for row in first_trace] == [row.skip_reason for row in second_trace]

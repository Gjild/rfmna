from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics import PortBoundary
from rfmna.solver import SolveResult, load_solver_threshold_config, solve_linear_system
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.conformance


def test_rf_warning_propagation_parity_and_context_across_metrics() -> None:
    thresholds = load_solver_threshold_config()
    warn_value = (thresholds.conditioning.warn_max + thresholds.conditioning.fail_max) * 0.5
    frequencies = np.asarray([1.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=0)
    request = SweepRFRequest(
        ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
        metrics=("y", "z", "s", "zin", "zout"),
    )

    class _ConstEstimator:
        def estimate(self, A: object) -> float | None:
            del A
            return warn_value

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        return solve_linear_system(
            matrix, rhs, node_voltage_count=1, condition_estimator=_ConstEstimator()
        )

    result = run_sweep(frequencies, layout, assemble, solve_point=solve_point, rf_request=request)
    assert result.rf_payloads is not None

    expected_warning_code = thresholds.conditioning.ill_conditioned_warning_code
    for metric_name in result.rf_payloads.metric_names:
        payload = result.rf_payloads.get(metric_name)
        assert payload is not None
        warning_diags = [
            diag for diag in payload.diagnostics_by_point[0] if diag.severity == "warning"
        ]
        assert warning_diags
        assert {diag.code for diag in warning_diags} == {expected_warning_code}
        assert all(diag.frequency_index == 0 for diag in warning_diags)
        assert all(diag.frequency_hz == 1.0 for diag in warning_diags)

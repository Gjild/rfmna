from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics import PortBoundary, YParameterResult
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.cross_check


def test_cross_check_one_port_resistor_y_matches_analytic_reference() -> None:
    resistance_ohm = 50.0
    expected_y11 = 1.0 / resistance_ohm

    frequencies = np.asarray((1.0e3, 1.0e6), dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=0)
    rf_request = SweepRFRequest(
        ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
        metrics=("y",),
    )

    def assemble_point(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        matrix = np.asarray([[expected_y11 + 0.0j]], dtype=np.complex128)
        rhs = np.zeros(1, dtype=np.complex128)
        return csc_matrix(matrix), rhs

    result = run_sweep(frequencies, layout, assemble_point, rf_request=rf_request)
    assert result.status.tolist() == ["pass", "pass"]
    assert result.rf_payloads is not None

    y_payload = result.rf_payloads.get("y")
    assert isinstance(y_payload, YParameterResult)
    assert np.allclose(y_payload.y[:, 0, 0], expected_y11 + 0.0j, rtol=0.0, atol=1.0e-12)

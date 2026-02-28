from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics import PortBoundary, extract_y_parameters

pytestmark = pytest.mark.conformance

# Conformance ID mapping (port_wave_conventions_v4_0_0.md):
# - RFY-001 -> test_rfy_001_one_port_voltage_excitation_matches_analytic_y11
#   (Section 2.1: I = YV, column-wise extraction via imposed voltage boundaries)
# - RFY-002 -> test_rfy_002_two_port_columns_match_analytic_y_block
#   (Sections 1 + 2.1: port orientation/sign and deterministic column construction)


def test_rfy_001_one_port_voltage_excitation_matches_analytic_y11() -> None:
    conductance = 0.02 + 0.0j
    frequencies = np.asarray([1.0, 10.0, 100.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[conductance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_y_parameters(frequencies, ports, assemble)
    assert result.y.shape == (3, 1, 1)
    assert np.allclose(result.y[:, 0, 0], np.asarray([conductance, conductance, conductance]))
    assert list(result.status.astype(str)) == ["pass", "pass", "pass"]
    assert result.diagnostics_by_point == ((), (), ())


def test_rfy_002_two_port_columns_match_analytic_y_block() -> None:
    analytic_y = np.asarray(
        [[0.15 + 0.0j, -0.05 + 0.0j], [-0.05 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128
    )
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(analytic_y),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_y_parameters(frequencies, ports, assemble)
    assert result.port_ids == ("P1", "P2")
    assert result.y.shape == (2, 2, 2)
    assert np.allclose(result.y[0, :, :], analytic_y)
    assert np.allclose(result.y[1, :, :], analytic_y)
    assert list(result.status.astype(str)) == ["pass", "pass"]
    assert result.diagnostics_by_point == ((), ())

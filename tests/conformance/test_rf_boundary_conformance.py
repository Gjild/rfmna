from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics.boundary import (
    PortBoundary,
    apply_current_boundaries,
    apply_voltage_boundaries,
)

pytestmark = pytest.mark.conformance

# Conformance ID mapping (port_wave_conventions_v4_0_0.md):
# - RFBC-001 -> test_rfbc_001_voltage_orientation_rows_follow_vp_definition (Section 1, V_p = V(p+) - V(p-))
# - RFBC-002 -> test_rfbc_002_current_sign_positive_into_dut (Section 1, I_p positive into DUT)


def test_rfbc_001_voltage_orientation_rows_follow_vp_definition() -> None:
    matrix = csc_matrix(
        np.asarray(
            [
                [2.0 + 0.0j, -1.0 + 0.0j],
                [-1.0 + 0.0j, 2.0 + 0.0j],
            ],
            dtype=np.complex128,
        )
    )
    rhs = np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    forward = apply_voltage_boundaries(
        matrix,
        rhs,
        (PortBoundary(port_id="P", p_plus_index=0, p_minus_index=1),),
        imposed_port_voltages=(("P", 1.0 + 0.0j),),
    )
    reverse = apply_voltage_boundaries(
        matrix,
        rhs,
        (PortBoundary(port_id="P", p_plus_index=1, p_minus_index=0),),
        imposed_port_voltages=(("P", 1.0 + 0.0j),),
    )

    assert forward.diagnostics == ()
    assert reverse.diagnostics == ()
    assert forward.matrix is not None
    assert reverse.matrix is not None
    forward_dense = np.asarray(forward.matrix.toarray(), dtype=np.complex128)
    reverse_dense = np.asarray(reverse.matrix.toarray(), dtype=np.complex128)

    assert forward_dense[2, 0] == 1.0 + 0.0j
    assert forward_dense[2, 1] == -1.0 + 0.0j
    assert forward_dense[0, 2] == 1.0 + 0.0j
    assert forward_dense[1, 2] == -1.0 + 0.0j

    assert reverse_dense[2, 0] == -1.0 + 0.0j
    assert reverse_dense[2, 1] == 1.0 + 0.0j
    assert reverse_dense[0, 2] == -1.0 + 0.0j
    assert reverse_dense[1, 2] == 1.0 + 0.0j


def test_rfbc_002_current_sign_positive_into_dut() -> None:
    matrix = csc_matrix(
        np.asarray([[3.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)
    )
    rhs = np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    forward = apply_current_boundaries(
        matrix,
        rhs,
        (PortBoundary(port_id="P", p_plus_index=0, p_minus_index=1),),
        imposed_port_currents=(("P", 2.0 + 0.0j),),
    )
    reverse = apply_current_boundaries(
        matrix,
        rhs,
        (PortBoundary(port_id="P", p_plus_index=1, p_minus_index=0),),
        imposed_port_currents=(("P", 2.0 + 0.0j),),
    )

    assert forward.diagnostics == ()
    assert reverse.diagnostics == ()
    assert forward.rhs is not None
    assert reverse.rhs is not None
    assert np.allclose(forward.rhs, np.asarray([2.0 + 0.0j, -2.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(reverse.rhs, np.asarray([-2.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128))

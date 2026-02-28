from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.diagnostics import canonical_witness_json, diagnostic_sort_key
from rfmna.rf_metrics import PortBoundary, extract_z_parameters

pytestmark = pytest.mark.conformance

# Conformance ID mapping (port_wave_conventions_v4_0_0.md):
# - RFZ-001 -> test_rfz_001_one_port_current_excitation_matches_analytic_z11
#   (Section 2.2: V = ZI under current excitation)
# - RFZ-002 -> test_rfz_002_two_port_columns_match_analytic_z_block
#   (Sections 1 + 2.2: orientation/sign conventions and deterministic column-wise extraction)
# - RFZ-003 -> test_rfz_003_y_to_z_singular_gate_emits_explicit_code
#   (Well-posedness gate: singular Y->Z conversion emits E_NUM_ZBLOCK_SINGULAR)
# - RFZ-004 -> test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code
#   (Well-posedness gate: ill-conditioned Y->Z conversion emits E_NUM_ZBLOCK_ILL_CONDITIONED)
# - RFZ-005 -> test_rfz_005_diagnostics_are_sorted_and_witness_stable
#   (Determinism: diagnostics sorted canonically with stable witness payloads)


def _inverse_2x2(matrix: np.ndarray) -> np.ndarray:
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    determinant = (a * d) - (b * c)
    return np.asarray(
        [[d / determinant, -b / determinant], [-c / determinant, a / determinant]],
        dtype=np.complex128,
    )


def test_rfz_001_one_port_current_excitation_matches_analytic_z11() -> None:
    conductance = 0.2 + 0.0j
    frequencies = np.asarray([1.0, 10.0, 100.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[conductance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble)
    assert result.z.shape == (3, 1, 1)
    assert np.allclose(
        result.z[:, 0, 0], np.asarray([1.0 / conductance, 1.0 / conductance, 1.0 / conductance])
    )
    assert list(result.status.astype(str)) == ["pass", "pass", "pass"]
    assert result.diagnostics_by_point == ((), (), ())


def test_rfz_002_two_port_columns_match_analytic_z_block() -> None:
    y_block = np.asarray(
        [[0.5 + 0.0j, -0.1 + 0.0j], [-0.1 + 0.0j, 0.4 + 0.0j]], dtype=np.complex128
    )
    expected_z = _inverse_2x2(y_block)
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_block),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble)
    assert result.port_ids == ("P1", "P2")
    assert result.z.shape == (2, 2, 2)
    assert np.allclose(result.z[0, :, :], expected_z)
    assert np.allclose(result.z[1, :, :], expected_z)
    assert list(result.status.astype(str)) == ["pass", "pass"]
    assert result.diagnostics_by_point == ((), ())


def test_rfz_003_y_to_z_singular_gate_emits_explicit_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    singular_y = np.asarray(
        [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(singular_y),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble, extraction_mode="y_to_z")
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.z[0, :, :].real).all()
    assert np.isnan(result.z[0, :, :].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_NUM_ZBLOCK_SINGULAR"]


def test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    ill_y = np.asarray(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0e-12 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(ill_y),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble, extraction_mode="y_to_z")
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.z[0, :, :].real).all()
    assert np.isnan(result.z[0, :, :].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == [
        "E_NUM_ZBLOCK_ILL_CONDITIONED"
    ]


def test_rfz_005_diagnostics_are_sorted_and_witness_stable() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=2, p_minus_index=2),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    baseline = extract_z_parameters(frequencies, ports, assemble)
    current = extract_z_parameters(frequencies, ports, assemble)
    point_diags = baseline.diagnostics_by_point[0]

    assert list(baseline.status.astype(str)) == ["fail"]
    assert point_diags
    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]

from __future__ import annotations

import numpy as np
import pytest

from rfmna.diagnostics import canonical_witness_json, diagnostic_sort_key
from rfmna.rf_metrics import convert_y_to_s, convert_z_to_s
from rfmna.rf_metrics.y_params import YParameterResult
from rfmna.rf_metrics.z_params import ZParameterResult

pytestmark = pytest.mark.conformance

# Conformance ID mapping (port_wave_conventions_v4_0_0.md):
# - RFS-001 -> test_rfs_001_z_formula_matches_section_3_1
#   (Section 3.1: S=(Z-Z0)(Z+Z0)^-1 with scalar Z0)
# - RFS-002 -> test_rfs_002_y_formula_matches_section_3_1
#   (Section 3.1: S=(I-Z0Y)(I+Z0Y)^-1 with diagonal per-port Z0)
# - RFS-003 -> test_rfs_003_complex_z0_rejected_with_explicit_model_code
#   (Section 3: complex Z0 rejected with E_MODEL_PORT_Z0_COMPLEX)
# - RFS-004 -> test_rfs_004_nonpositive_z0_rejected_with_explicit_model_code
#   (Section 3: non-positive Z0 rejected with E_MODEL_PORT_Z0_NONPOSITIVE)
# - RFS-005 -> test_rfs_005_singular_conversion_emits_explicit_code
#   (Section 3.1: singular conversion emits E_NUM_S_CONVERSION_SINGULAR)
# - RFS-006 -> test_rfs_006_diagnostics_ordering_and_witnesses_are_deterministic
#   (Section 4 + diagnostics taxonomy: deterministic diagnostics ordering/witness payload stability)


def test_rfs_001_z_formula_matches_section_3_1() -> None:
    frequencies = np.asarray([1.0, 10.0], dtype=np.float64)
    z_block = np.asarray(
        [
            [[75.0 + 0.0j]],
            [[100.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    z_result = ZParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1",),
        z=z_block,
        status=np.asarray(["pass", "pass"]),
        diagnostics_by_point=((), ()),
        extraction_mode="direct",
    )

    result = convert_z_to_s(z_result, z0_ohm=50.0)
    expected = np.asarray(
        [
            (75.0 - 50.0) / (75.0 + 50.0),
            (100.0 - 50.0) / (100.0 + 50.0),
        ],
        dtype=np.float64,
    )

    assert result.s.shape == (2, 1, 1)
    assert np.allclose(result.s[:, 0, 0], expected)
    assert list(result.status.astype(str)) == ["pass", "pass"]
    assert result.diagnostics_by_point == ((), ())


def test_rfs_002_y_formula_matches_section_3_1() -> None:
    frequencies = np.asarray([2.0], dtype=np.float64)
    y_block = np.asarray(
        [[[0.02 + 0.0j, -0.004 + 0.0j], [-0.004 + 0.0j, 0.03 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = YParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1", "P2"),
        y=y_block,
        status=np.asarray(["pass"]),
        diagnostics_by_point=((),),
    )
    z0 = np.asarray([50.0, 75.0], dtype=np.float64)

    result = convert_y_to_s(y_result, z0_ohm=z0)
    z0_diag = np.diag(z0.astype(np.complex128))
    identity = np.eye(2, dtype=np.complex128)
    expected = (identity - (z0_diag @ y_block[0])) @ np.linalg.inv(
        identity + (z0_diag @ y_block[0])
    )

    assert result.port_ids == ("P1", "P2")
    assert result.s.shape == (1, 2, 2)
    assert np.allclose(result.s[0], expected)
    assert list(result.status.astype(str)) == ["pass"]
    assert result.diagnostics_by_point == ((),)


def test_rfs_003_complex_z0_rejected_with_explicit_model_code() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    z_block = np.asarray(
        [
            [[75.0 + 0.0j]],
            [[100.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    z_result = ZParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1",),
        z=z_block,
        status=np.asarray(["pass", "pass"]),
        diagnostics_by_point=((), ()),
        extraction_mode="direct",
    )

    result = convert_z_to_s(z_result, z0_ohm=50.0 + 1.0j)
    assert list(result.status.astype(str)) == ["fail", "fail"]
    assert np.isnan(result.s[:, :, :].real).all()
    assert np.isnan(result.s[:, :, :].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_MODEL_PORT_Z0_COMPLEX"]
    assert [diag.code for diag in result.diagnostics_by_point[1]] == ["E_MODEL_PORT_Z0_COMPLEX"]


def test_rfs_004_nonpositive_z0_rejected_with_explicit_model_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    y_block = np.asarray(
        [[[0.02 + 0.0j, -0.004 + 0.0j], [-0.004 + 0.0j, 0.03 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = YParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1", "P2"),
        y=y_block,
        status=np.asarray(["pass"]),
        diagnostics_by_point=((),),
    )

    result = convert_y_to_s(y_result, z0_ohm=np.asarray([50.0, 0.0], dtype=np.float64))
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_MODEL_PORT_Z0_NONPOSITIVE"]


def test_rfs_005_singular_conversion_emits_explicit_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_block = np.asarray([[[-50.0 + 0.0j]]], dtype=np.complex128)
    z_result = ZParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1",),
        z=z_block,
        status=np.asarray(["pass"]),
        diagnostics_by_point=((),),
        extraction_mode="direct",
    )

    result = convert_z_to_s(z_result, z0_ohm=50.0)
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_NUM_S_CONVERSION_SINGULAR"]


def test_rfs_006_diagnostics_ordering_and_witnesses_are_deterministic() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_block = np.asarray(
        [[[75.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 90.0 + 0.0j]]],
        dtype=np.complex128,
    )
    z_result = ZParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1", "P2"),
        z=z_block,
        status=np.asarray(["pass"]),
        diagnostics_by_point=((),),
        extraction_mode="direct",
    )

    baseline = convert_z_to_s(z_result, z0_ohm=np.asarray([50.0 + 1.0j, 0.0], dtype=np.complex128))
    current = convert_z_to_s(z_result, z0_ohm=np.asarray([50.0 + 1.0j, 0.0], dtype=np.complex128))

    assert list(baseline.status.astype(str)) == ["fail"]
    point_diags = baseline.diagnostics_by_point[0]
    assert [diag.code for diag in point_diags] == [
        "E_MODEL_PORT_Z0_COMPLEX",
        "E_MODEL_PORT_Z0_NONPOSITIVE",
    ]
    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]

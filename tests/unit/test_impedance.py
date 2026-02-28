from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.rf_metrics.impedance as impedance_module
from rfmna.diagnostics import canonical_witness_json, diagnostic_sort_key
from rfmna.rf_metrics import PortBoundary, extract_zin_zout
from rfmna.rf_metrics.z_params import ZParameterResult

pytestmark = pytest.mark.unit

_PERMUTATION_REPEATS = 30


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_trivial_fixture_resistor_yields_expected_zin_and_zout() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    conductance = 0.25 + 0.0j

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[conductance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_zin_zout(frequencies, ports, assemble)
    expected = np.asarray(
        [1.0 / conductance, 1.0 / conductance, 1.0 / conductance], dtype=np.complex128
    )

    assert np.allclose(result.zin, expected)
    assert np.allclose(result.zout, expected)
    assert list(result.status.astype(str)) == ["pass", "pass", "pass"]
    assert result.diagnostics_by_point == ((), (), ())


def test_short_fixture_yields_near_zero_impedance() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    conductance = 1.0e12 + 0.0j

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[conductance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_zin_zout(frequencies, ports, assemble)
    assert np.allclose(
        result.zin, np.asarray([1.0e-12 + 0.0j], dtype=np.complex128), atol=1.0e-20, rtol=0.0
    )
    assert np.allclose(
        result.zout, np.asarray([1.0e-12 + 0.0j], dtype=np.complex128), atol=1.0e-20, rtol=0.0
    )
    assert list(result.status.astype(str)) == ["pass"]
    assert result.diagnostics_by_point == ((),)


def test_open_fixture_emits_fail_sentinel_and_explicit_singular_diagnostic() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_zin_zout(frequencies, ports, assemble)
    assert np.isnan(result.zin.real).all()
    assert np.isnan(result.zin.imag).all()
    assert np.isnan(result.zout.real).all()
    assert np.isnan(result.zout.imag).all()
    assert list(result.status.astype(str)) == ["fail", "fail"]
    assert "E_NUM_SINGULAR_MATRIX" in [diag.code for diag in result.diagnostics_by_point[0]]
    assert "E_NUM_SINGULAR_MATRIX" in [diag.code for diag in result.diagnostics_by_point[1]]


def test_singular_boundary_fixture_emits_explicit_boundary_diagnostic() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=1, p_minus_index=None),
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(
                np.asarray(
                    [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128
                )
            ),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_zin_zout(frequencies, ports, assemble)
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.zin.real).all()
    assert np.isnan(result.zin.imag).all()
    assert np.isnan(result.zout.real).all()
    assert np.isnan(result.zout.imag).all()
    assert "E_TOPO_RF_BOUNDARY_INCONSISTENT" in [
        diag.code for diag in result.diagnostics_by_point[0]
    ]


def test_orientation_conventions_are_preserved_under_terminal_swap() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    conductance = 0.5 + 0.0j
    normal_ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    swapped_ports = (PortBoundary(port_id="P1", p_plus_index=None, p_minus_index=0),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[conductance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    normal = extract_zin_zout(frequencies, normal_ports, assemble)
    swapped = extract_zin_zout(frequencies, swapped_ports, assemble)
    assert np.allclose(normal.zin, np.asarray([2.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(swapped.zin, np.asarray([2.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(normal.zout, swapped.zout)


def test_port_permutation_invariance_and_canonical_selection() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    base_ports = [
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    ]
    y_matrix = np.asarray(
        [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_matrix),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    baseline = extract_zin_zout(frequencies, tuple(base_ports), assemble)
    assert baseline.port_ids == ("P1", "P2")
    assert baseline.input_port_id == "P1"
    assert baseline.output_port_id == "P2"

    baseline_zin = np.asarray(baseline.zin, dtype=np.complex128).copy()
    baseline_zout = np.asarray(baseline.zout, dtype=np.complex128).copy()
    baseline_diag = tuple(_diag_json(point) for point in baseline.diagnostics_by_point)
    rng = Random(0)

    for _ in range(_PERMUTATION_REPEATS):
        ports = list(base_ports)
        rng.shuffle(ports)
        current = extract_zin_zout(frequencies, tuple(ports), assemble)
        assert current.port_ids == ("P1", "P2")
        assert current.input_port_id == "P1"
        assert current.output_port_id == "P2"
        assert np.allclose(current.zin, baseline_zin)
        assert np.allclose(current.zout, baseline_zout)
        assert tuple(_diag_json(point) for point in current.diagnostics_by_point) == baseline_diag


def test_undefined_impedance_emits_explicit_diagnostic_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def fake_extract_z_parameters(*args: object, **kwargs: object) -> ZParameterResult:
        del args, kwargs
        return ZParameterResult(
            frequencies_hz=np.asarray([1.0], dtype=np.float64),
            port_ids=("P1",),
            z=np.asarray(
                [[[np.complex128(complex(float("nan"), float("nan")))]]], dtype=np.complex128
            ),
            status=np.asarray(["pass"]),
            diagnostics_by_point=((),),
            extraction_mode="direct",
        )

    monkeypatch.setattr(impedance_module, "extract_z_parameters", fake_extract_z_parameters)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_zin_zout(frequencies, ports, assemble)
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.zin.real).all()
    assert np.isnan(result.zin.imag).all()
    assert np.isnan(result.zout.real).all()
    assert np.isnan(result.zout.imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_NUM_IMPEDANCE_UNDEFINED"]


def test_diagnostics_sort_and_witness_stability_for_selection_errors() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    input_port_id = "P9"
    output_port_id = "P8"

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    baseline = extract_zin_zout(
        frequencies,
        ports,
        assemble,
        input_port_id=input_port_id,
        output_port_id=output_port_id,
    )
    current = extract_zin_zout(
        frequencies,
        ports,
        assemble,
        input_port_id=input_port_id,
        output_port_id=output_port_id,
    )

    point_diags = baseline.diagnostics_by_point[0]
    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert [diag.code for diag in point_diags] == [
        "E_TOPO_RF_BOUNDARY_INCONSISTENT",
        "E_TOPO_RF_BOUNDARY_INCONSISTENT",
    ]
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]

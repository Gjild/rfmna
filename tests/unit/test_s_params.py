from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csc_matrix as scipy_csc_matrix  # type: ignore[import-untyped]

import rfmna.rf_metrics.s_params as s_params_module
from rfmna.diagnostics import (
    DiagnosticEvent,
    Severity,
    SolverStage,
    canonical_witness_json,
    diagnostic_sort_key,
)
from rfmna.rf_metrics import convert_y_to_s, convert_z_to_s
from rfmna.rf_metrics.y_params import YParameterResult
from rfmna.rf_metrics.z_params import ZParameterResult

pytestmark = pytest.mark.unit


def _z_result(
    *,
    frequencies_hz: np.ndarray,
    port_ids: tuple[str, ...],
    z: np.ndarray,
    status: np.ndarray | None = None,
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...] | None = None,
) -> ZParameterResult:
    n_points = int(frequencies_hz.shape[0])
    return ZParameterResult(
        frequencies_hz=frequencies_hz,
        port_ids=port_ids,
        z=z,
        status=np.asarray(["pass"] * n_points if status is None else status),
        diagnostics_by_point=tuple(() for _ in range(n_points))
        if diagnostics_by_point is None
        else diagnostics_by_point,
        extraction_mode="direct",
    )


def _y_result(
    *,
    frequencies_hz: np.ndarray,
    port_ids: tuple[str, ...],
    y: np.ndarray,
    status: np.ndarray | None = None,
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...] | None = None,
) -> YParameterResult:
    n_points = int(frequencies_hz.shape[0])
    return YParameterResult(
        frequencies_hz=frequencies_hz,
        port_ids=port_ids,
        y=y,
        status=np.asarray(["pass"] * n_points if status is None else status),
        diagnostics_by_point=tuple(() for _ in range(n_points))
        if diagnostics_by_point is None
        else diagnostics_by_point,
    )


def test_convert_z_to_s_one_port_matches_closed_form() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    z_matrix = np.asarray(
        [
            [[75.0 + 0.0j]],
            [[100.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    z_result = _z_result(frequencies_hz=frequencies, port_ids=("P1",), z=z_matrix)

    result = convert_z_to_s(z_result, z0_ohm=50.0)
    expected = np.asarray(
        [(75.0 - 50.0) / (75.0 + 50.0), (100.0 - 50.0) / (100.0 + 50.0)], dtype=np.float64
    )

    assert result.conversion_source == "from_z"
    assert result.s.shape == (2, 1, 1)
    assert np.allclose(result.s[:, 0, 0], expected)
    assert list(result.status.astype(str)) == ["pass", "pass"]
    assert result.diagnostics_by_point == ((), ())


def test_convert_y_to_s_two_port_matches_matrix_formula() -> None:
    frequencies = np.asarray([3.0], dtype=np.float64)
    y_matrix = np.asarray(
        [[[0.02 + 0.0j, -0.005 + 0.0j], [-0.005 + 0.0j, 0.03 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = _y_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), y=y_matrix)
    z0 = np.asarray([50.0, 75.0], dtype=np.float64)

    result = convert_y_to_s(y_result, z0_ohm=z0)
    z0_diag = np.diag(z0.astype(np.complex128))
    identity = np.eye(2, dtype=np.complex128)
    expected = (identity - (z0_diag @ y_matrix[0])) @ np.linalg.inv(
        identity + (z0_diag @ y_matrix[0])
    )

    assert result.conversion_source == "from_y"
    assert result.s.shape == (1, 2, 2)
    assert np.allclose(result.s[0], expected)
    assert list(result.status.astype(str)) == ["pass"]
    assert result.diagnostics_by_point == ((),)


def test_convert_y_to_s_builds_sparse_denominator_from_y_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([3.0], dtype=np.float64)
    y_matrix = np.asarray(
        [[[0.02 + 0.0j, -0.005 + 0.0j], [-0.005 + 0.0j, 0.03 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = _y_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), y=y_matrix)
    z0 = np.asarray([50.0, 75.0], dtype=np.float64)
    captured_csc_inputs: list[np.ndarray] = []
    original_csc_matrix = s_params_module.csc_matrix

    def _spy_csc_matrix(*args: object, **kwargs: object) -> object:
        if args:
            captured_csc_inputs.append(np.asarray(args[0], dtype=np.complex128).copy())
        return scipy_csc_matrix(*args, **kwargs)

    monkeypatch.setattr(s_params_module, "csc_matrix", _spy_csc_matrix)
    result = convert_y_to_s(y_result, z0_ohm=z0)

    assert list(result.status.astype(str)) == ["pass"]
    assert len(captured_csc_inputs) == 1
    assert np.allclose(captured_csc_inputs[0], y_matrix[0])

    monkeypatch.setattr(s_params_module, "csc_matrix", original_csc_matrix)


def test_complex_z0_emits_explicit_model_code_and_retains_all_points() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    z_matrix = np.asarray([[[75.0 + 0.0j]], [[80.0 + 0.0j]]], dtype=np.complex128)
    z_result = _z_result(frequencies_hz=frequencies, port_ids=("P1",), z=z_matrix)

    result = convert_z_to_s(z_result, z0_ohm=50.0 + 1.0j)

    assert result.s.shape == (2, 1, 1)
    assert np.isnan(result.s[:, :, :].real).all()
    assert np.isnan(result.s[:, :, :].imag).all()
    assert list(result.status.astype(str)) == ["fail", "fail"]
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_MODEL_PORT_Z0_COMPLEX"]
    assert [diag.code for diag in result.diagnostics_by_point[1]] == ["E_MODEL_PORT_Z0_COMPLEX"]
    assert result.diagnostics_by_point[0][0].frequency_index == 0
    assert result.diagnostics_by_point[1][0].frequency_index == 1


def test_nonpositive_z0_vector_emits_explicit_model_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    y_matrix = np.asarray(
        [[[0.02 + 0.0j, -0.001 + 0.0j], [-0.001 + 0.0j, 0.025 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = _y_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), y=y_matrix)

    result = convert_y_to_s(y_result, z0_ohm=np.asarray([50.0, 0.0], dtype=np.float64))

    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    point_diags = result.diagnostics_by_point[0]
    assert [diag.code for diag in point_diags] == ["E_MODEL_PORT_Z0_NONPOSITIVE"]
    assert point_diags[0].port_context is not None
    assert point_diags[0].port_context.port_id == "P2"


def test_nonfinite_z0_vector_emits_explicit_model_code() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    y_matrix = np.asarray(
        [[[0.02 + 0.0j, -0.001 + 0.0j], [-0.001 + 0.0j, 0.025 + 0.0j]]],
        dtype=np.complex128,
    )
    y_result = _y_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), y=y_matrix)

    result = convert_y_to_s(y_result, z0_ohm=np.asarray([50.0, float("inf")], dtype=np.float64))

    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    point_diags = result.diagnostics_by_point[0]
    assert [diag.code for diag in point_diags] == ["E_MODEL_PORT_Z0_NONPOSITIVE"]
    assert point_diags[0].port_context is not None
    assert point_diags[0].port_context.port_id == "P2"


def test_singular_conversion_emits_explicit_code_without_regularization() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_matrix = np.asarray([[[-50.0 + 0.0j]]], dtype=np.complex128)
    z_result = _z_result(frequencies_hz=frequencies, port_ids=("P1",), z=z_matrix)

    result = convert_z_to_s(z_result, z0_ohm=50.0)

    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_NUM_S_CONVERSION_SINGULAR"]


def test_conversion_does_not_mutate_y_or_z_inputs() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_matrix = np.asarray([[[75.0 + 0.0j]]], dtype=np.complex128)
    y_matrix = np.asarray(
        [[[0.02 + 0.0j, -0.005 + 0.0j], [-0.005 + 0.0j, 0.03 + 0.0j]]],
        dtype=np.complex128,
    )
    z_original = deepcopy(z_matrix)
    y_original = deepcopy(y_matrix)

    z_result = _z_result(frequencies_hz=frequencies, port_ids=("P1",), z=z_matrix)
    y_result = _y_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), y=y_matrix)

    _ = convert_z_to_s(z_result, z0_ohm=50.0)
    _ = convert_y_to_s(y_result, z0_ohm=np.asarray([50.0, 50.0], dtype=np.float64))

    assert np.array_equal(z_matrix, z_original)
    assert np.array_equal(y_matrix, y_original)


def test_diagnostics_sorting_and_witness_serialization_are_deterministic() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_matrix = np.asarray(
        [[[75.0 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 90.0 + 0.0j]]],
        dtype=np.complex128,
    )
    z_result = _z_result(frequencies_hz=frequencies, port_ids=("P1", "P2"), z=z_matrix)
    z0 = np.asarray([50.0 + 1.0j, 0.0], dtype=np.complex128)

    baseline = convert_z_to_s(z_result, z0_ohm=z0)
    current = convert_z_to_s(z_result, z0_ohm=z0)
    point_diags = baseline.diagnostics_by_point[0]

    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]


def test_upstream_diagnostics_are_retained_for_failed_input_points() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    z_matrix = np.asarray([[[75.0 + 0.0j]]], dtype=np.complex128)
    upstream_diag = DiagnosticEvent(
        code="E_NUM_SOLVE_FAILED",
        severity=Severity.ERROR,
        message="upstream failure",
        suggested_action="inspect upstream extraction",
        solver_stage=SolverStage.SOLVE,
        element_id="rf_z_params",
        frequency_hz=1.0,
        frequency_index=0,
        witness={"source": "unit"},
    )
    z_result = _z_result(
        frequencies_hz=frequencies,
        port_ids=("P1",),
        z=z_matrix,
        status=np.asarray(["fail"]),
        diagnostics_by_point=((upstream_diag,),),
    )

    result = convert_z_to_s(z_result, z0_ohm=50.0)
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.s[0].real).all()
    assert np.isnan(result.s[0].imag).all()
    assert result.diagnostics_by_point[0][0].code == "E_NUM_SOLVE_FAILED"

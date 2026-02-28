from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.rf_metrics.z_params as z_params_module
from rfmna.diagnostics import canonical_witness_json, diagnostic_sort_key
from rfmna.rf_metrics import PortBoundary, extract_z_parameters
from rfmna.rf_metrics.y_params import YParameterResult
from rfmna.solver import SolveResult, load_solver_threshold_config, solve_linear_system

pytestmark = pytest.mark.unit

_PERMUTATION_REPEATS = 30
_POINT_FAIL_INDEX = 1
_EXPECTED_DIAG_COUNT = 3


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


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


def test_direct_one_port_matches_analytic_impedance() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    admittance = 0.25 + 0.0j

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[admittance]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble)
    assert result.z.shape == (2, 1, 1)
    assert np.allclose(result.z[:, 0, 0], np.asarray([1.0 / admittance, 1.0 / admittance]))
    assert list(result.status.astype(str)) == ["pass", "pass"]


def test_direct_two_port_matches_analytic_impedance_and_is_permutation_stable() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    ports = [
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    ]
    y_matrix = np.asarray(
        [[0.5 + 0.0j, -0.1 + 0.0j], [-0.1 + 0.0j, 0.4 + 0.0j]], dtype=np.complex128
    )
    expected_z = _inverse_2x2(y_matrix)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_matrix),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    baseline = extract_z_parameters(frequencies, tuple(ports), assemble)
    assert baseline.port_ids == ("P1", "P2")
    assert np.allclose(baseline.z[0, :, :], expected_z)
    assert np.allclose(baseline.z[1, :, :], expected_z)
    assert np.allclose(baseline.z[2, :, :], expected_z)

    baseline_json = _diag_json(baseline.diagnostics_by_point[0])
    rng = Random(0)
    for _ in range(_PERMUTATION_REPEATS):
        permuted_ports = list(ports)
        rng.shuffle(permuted_ports)
        current = extract_z_parameters(frequencies, tuple(permuted_ports), assemble)
        assert current.port_ids == ("P1", "P2")
        assert np.allclose(current.z, baseline.z)
        assert _diag_json(current.diagnostics_by_point[0]) == baseline_json


def test_failed_point_is_retained_with_matrix_nan_sentinel_and_diagnostic() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(point_index: int, _: float) -> tuple[csc_matrix, np.ndarray]:
        if point_index == _POINT_FAIL_INDEX:
            raise RuntimeError("intentional assembly failure")
        return (
            csc_matrix(np.asarray([[0.5 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble)
    assert result.z.shape == (3, 1, 1)
    assert np.isclose(result.z[0, 0, 0], 2.0 + 0.0j)
    assert np.isnan(result.z[1, :, :].real).all()
    assert np.isnan(result.z[1, :, :].imag).all()
    assert np.isclose(result.z[2, 0, 0], 2.0 + 0.0j)
    assert list(result.status.astype(str)) == ["pass", "fail", "pass"]
    assert [diag.code for diag in result.diagnostics_by_point[1]] == ["E_NUM_SOLVE_FAILED"]


def test_y_to_z_conversion_runs_only_when_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    called = {"count": 0}

    def fake_extract_y_parameters(*args: object, **kwargs: object) -> YParameterResult:
        called["count"] += 1
        return YParameterResult(
            frequencies_hz=np.asarray([1.0], dtype=np.float64),
            port_ids=("P1",),
            y=np.asarray([[[0.5 + 0.0j]]], dtype=np.complex128),
            status=np.asarray(["pass"]),
            diagnostics_by_point=((),),
        )

    monkeypatch.setattr(z_params_module, "extract_y_parameters", fake_extract_y_parameters)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.5 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    direct = extract_z_parameters(frequencies, ports, assemble, extraction_mode="direct")
    assert called["count"] == 0
    assert np.isclose(direct.z[0, 0, 0], 2.0 + 0.0j)

    converted = extract_z_parameters(frequencies, ports, assemble, extraction_mode="y_to_z")
    assert called["count"] == 1
    assert np.isclose(converted.z[0, 0, 0], 2.0 + 0.0j)


def test_y_to_z_singular_gate_emits_explicit_code_and_no_regularization() -> None:
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


def test_y_to_z_ill_conditioned_gate_emits_explicit_code() -> None:
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


def test_y_to_z_well_conditioned_matches_direct() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    )
    y_matrix = np.asarray(
        [[0.6 + 0.0j, -0.2 + 0.0j], [-0.2 + 0.0j, 0.5 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_matrix),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    direct = extract_z_parameters(frequencies, ports, assemble, extraction_mode="direct")
    via_y = extract_z_parameters(frequencies, ports, assemble, extraction_mode="y_to_z")
    assert np.allclose(via_y.z, direct.z)
    assert list(via_y.status.astype(str)) == list(direct.status.astype(str))


def test_diagnostics_are_canonically_sorted_with_stable_witnesses() -> None:
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
    assert len(point_diags) == _EXPECTED_DIAG_COUNT
    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]


def test_default_direct_solver_infers_node_voltage_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    calls: list[dict[str, object]] = []
    real_solve = solve_linear_system

    def spy_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        calls.append(dict(kwargs))
        return real_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr(z_params_module, "solve_linear_system", spy_solve)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble, extraction_mode="direct")
    assert list(result.status.astype(str)) == ["pass"]
    assert calls
    assert all(call.get("node_voltage_count") == 1 for call in calls)


def test_y_to_z_conversion_uses_no_gmin_regularization_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    y_matrix = np.asarray(
        [[0.6 + 0.0j, -0.2 + 0.0j], [-0.2 + 0.0j, 0.5 + 0.0j]], dtype=np.complex128
    )
    calls: list[dict[str, object]] = []
    real_solve = solve_linear_system

    def spy_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        calls.append(dict(kwargs))
        return real_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr(z_params_module, "solve_linear_system", spy_solve)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_matrix),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_z_parameters(frequencies, ports, assemble, extraction_mode="y_to_z")
    assert list(result.status.astype(str)) == ["pass"]

    mna_calls = [call for call in calls if "node_voltage_count" in call]
    conversion_calls = [call for call in calls if "run_config" in call]
    assert mna_calls
    assert all(call.get("node_voltage_count") == len(ports) for call in mna_calls)
    assert conversion_calls
    assert all(call["run_config"].enable_gmin is False for call in conversion_calls)


def test_y_to_z_conversion_uses_no_gmin_default_solver_path() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    y_matrix = np.asarray(
        [[0.6 + 0.0j, -0.2 + 0.0j], [-0.2 + 0.0j, 0.5 + 0.0j]], dtype=np.complex128
    )

    class _ConstEstimator:
        def estimate(self, A: object) -> float | None:
            del A
            return None

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(y_matrix),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        return solve_linear_system(
            matrix, rhs, node_voltage_count=2, condition_estimator=_ConstEstimator()
        )

    result = extract_z_parameters(
        frequencies, ports, assemble, solve_point=solve_point, extraction_mode="y_to_z"
    )
    point_diags = result.diagnostics_by_point[0]
    warning_codes = [diag.code for diag in point_diags if diag.severity == "warning"]
    assert list(result.status.astype(str)) == ["pass"]
    assert warning_codes == []


def test_solver_warnings_propagate_for_direct_extraction_columns() -> None:
    thresholds = load_solver_threshold_config()
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    class _ConstEstimator:
        def estimate(self, A: object) -> float | None:
            del A
            return None

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.5 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        return solve_linear_system(
            matrix, rhs, node_voltage_count=1, condition_estimator=_ConstEstimator()
        )

    result = extract_z_parameters(
        frequencies, ports, assemble, solve_point=solve_point, extraction_mode="direct"
    )
    point_diags = result.diagnostics_by_point[0]
    warning_codes = [diag.code for diag in point_diags if diag.severity == "warning"]
    assert list(result.status.astype(str)) == ["pass"]
    assert warning_codes == [thresholds.conditioning.unavailable_warning_code]
    assert point_diags[0].frequency_index == 0


def test_y_to_z_conversion_not_rescued_by_custom_mna_solver_gmin() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    singular_y = np.asarray(
        [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(singular_y),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        return solve_linear_system(matrix, rhs, node_voltage_count=2)

    result = extract_z_parameters(
        frequencies, ports, assemble, solve_point=solve_point, extraction_mode="y_to_z"
    )
    assert list(result.status.astype(str)) == ["fail"]
    assert [diag.code for diag in result.diagnostics_by_point[0]] == ["E_NUM_ZBLOCK_SINGULAR"]


def test_explicit_node_voltage_count_covers_internal_nodes_not_in_ports() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    singular_with_internal_node = np.asarray(
        [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(singular_with_internal_node),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    default_result = extract_z_parameters(frequencies, ports, assemble, extraction_mode="direct")
    explicit_result = extract_z_parameters(
        frequencies,
        ports,
        assemble,
        node_voltage_count=2,
        extraction_mode="direct",
    )

    assert list(default_result.status.astype(str)) == ["fail"]
    assert list(explicit_result.status.astype(str)) == ["pass"]

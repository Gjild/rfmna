from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.diagnostics import canonical_witness_json, diagnostic_sort_key
from rfmna.rf_metrics import PortBoundary, extract_y_parameters
from rfmna.solver import SolveResult, load_solver_threshold_config, solve_linear_system

pytestmark = pytest.mark.unit

_PERMUTATION_REPEATS = 30
_EXPECTED_DIAGNOSTIC_COUNT = 3
_POINT_ONE_FREQ_HZ = 2.0
_SOLVER_FAIL_CALL_INDEX = 2


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_two_port_analytic_y_and_canonical_column_order_under_permutations() -> None:
    admittance = np.asarray(
        [[0.15 + 0.0j, -0.05 + 0.0j], [-0.05 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128
    )
    rhs = np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    base_ports = [
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    ]

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (csc_matrix(admittance), rhs)

    baseline = extract_y_parameters(frequencies, tuple(base_ports), assemble)
    assert baseline.port_ids == ("P1", "P2")
    assert baseline.y.shape == (3, 2, 2)
    assert np.allclose(baseline.y[0, :, :], admittance)
    assert np.allclose(baseline.y[1, :, :], admittance)
    assert np.allclose(baseline.y[2, :, :], admittance)
    assert list(baseline.status.astype(str)) == ["pass", "pass", "pass"]

    baseline_y = np.asarray(baseline.y, dtype=np.complex128).copy()
    baseline_diag = tuple(_diag_json(point) for point in baseline.diagnostics_by_point)
    rng = Random(0)

    for _ in range(_PERMUTATION_REPEATS):
        ports = list(base_ports)
        rng.shuffle(ports)
        current = extract_y_parameters(frequencies, tuple(ports), assemble)
        assert current.port_ids == ("P1", "P2")
        assert np.allclose(current.y, baseline_y)
        assert tuple(_diag_json(point) for point in current.diagnostics_by_point) == baseline_diag


def test_failed_point_is_retained_with_full_matrix_nan_sentinel_and_frequency_alignment() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(point_index: int, _: float) -> tuple[csc_matrix, np.ndarray]:
        if point_index == 1:
            raise RuntimeError("intentional assembly failure")
        return (
            csc_matrix(np.asarray([[0.5 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_y_parameters(frequencies, ports, assemble)
    assert result.y.shape == (3, 1, 1)
    assert np.isclose(result.y[0, 0, 0], 0.5 + 0.0j)
    assert np.isnan(result.y[1, :, :].real).all()
    assert np.isnan(result.y[1, :, :].imag).all()
    assert np.isclose(result.y[2, 0, 0], 0.5 + 0.0j)
    assert list(result.status.astype(str)) == ["pass", "fail", "pass"]

    assert result.diagnostics_by_point[0] == ()
    assert result.diagnostics_by_point[2] == ()
    point_one = result.diagnostics_by_point[1]
    assert [diag.code for diag in point_one] == ["E_NUM_SOLVE_FAILED"]
    assert point_one[0].frequency_index == 1
    assert point_one[0].frequency_hz == _POINT_ONE_FREQ_HZ


def test_single_column_failure_marks_entire_point_matrix_as_nan() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )
    solve_call_count = {"count": 0}

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(
                np.asarray(
                    [[0.1 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.2 + 0.0j]], dtype=np.complex128
                )
            ),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        solve_call_count["count"] += 1
        if solve_call_count["count"] == _SOLVER_FAIL_CALL_INDEX:
            raise RuntimeError("intentional solver failure")
        return solve_linear_system(matrix, rhs)

    result = extract_y_parameters(frequencies, ports, assemble, solve_point=solve_point)
    assert list(result.status.astype(str)) == ["fail"]
    assert np.isnan(result.y[0, :, :].real).all()
    assert np.isnan(result.y[0, :, :].imag).all()
    assert result.diagnostics_by_point[0]
    assert result.diagnostics_by_point[0][0].code == "E_NUM_SOLVE_FAILED"


def test_diagnostics_use_canonical_sort_and_stable_witness_serialization() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=5, p_minus_index=5),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    baseline = extract_y_parameters(frequencies, ports, assemble)
    current = extract_y_parameters(frequencies, ports, assemble)

    assert list(baseline.status.astype(str)) == ["fail"]
    point_diags = baseline.diagnostics_by_point[0]
    assert len(point_diags) == _EXPECTED_DIAGNOSTIC_COUNT
    assert point_diags == tuple(sorted(point_diags, key=diagnostic_sort_key))
    assert all(diag.frequency_index == 0 for diag in point_diags)
    assert all(diag.frequency_hz == 1.0 for diag in point_diags)
    assert [canonical_witness_json(diag.witness) for diag in point_diags] == [
        canonical_witness_json(diag.witness) for diag in current.diagnostics_by_point[0]
    ]


def test_default_solver_infers_node_voltage_count_for_mna_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    calls: list[dict[str, object]] = []
    real_solve = solve_linear_system

    def spy_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        calls.append(dict(kwargs))
        return real_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr("rfmna.rf_metrics.y_params.solve_linear_system", spy_solve)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = extract_y_parameters(frequencies, ports, assemble)
    assert list(result.status.astype(str)) == ["pass"]
    assert calls
    assert all(call.get("node_voltage_count") == 1 for call in calls)


def test_solver_warnings_are_propagated_with_point_and_frequency_context() -> None:
    thresholds = load_solver_threshold_config()
    frequencies = np.asarray([1.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)
    warn_value = (thresholds.conditioning.warn_max + thresholds.conditioning.fail_max) * 0.5

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

    result = extract_y_parameters(frequencies, ports, assemble, solve_point=solve_point)
    point_diags = result.diagnostics_by_point[0]
    assert list(result.status.astype(str)) == ["pass"]
    assert [diag.code for diag in point_diags] == [
        thresholds.conditioning.ill_conditioned_warning_code
    ]
    assert point_diags[0].frequency_index == 0
    assert point_diags[0].frequency_hz == 1.0
    assert point_diags[0].solver_stage == "solve"

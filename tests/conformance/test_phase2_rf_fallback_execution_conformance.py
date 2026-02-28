from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.rf_metrics.y_params as y_params_module
import rfmna.rf_metrics.z_params as z_params_module
import rfmna.sweep_engine.run as sweep_run_module
from rfmna.rf_metrics import (
    PortBoundary,
    extract_y_parameters,
    extract_z_parameters,
    extract_zin_zout,
)
from rfmna.solver import SolveResult, solve_linear_system
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.conformance
_EXPECTED_POINTS = 2


def _zero_assemble_point(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
    return (
        csc_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128)),
        np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )


def _has_node_count_unavailable_skip(trace: tuple[object, ...]) -> bool:
    return any(
        getattr(row, "stage", None) == "gmin"
        and getattr(row, "stage_state", None) == "skipped"
        and getattr(row, "skip_reason", None) == "node_voltage_count_unavailable"
        for row in trace
    )


def test_sweep_rf_payload_paths_keep_mna_gmin_eligible_and_conversion_no_gmin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=0)
    request = SweepRFRequest(
        ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
        metrics=("y", "z", "s", "zin", "zout"),
    )

    calls: list[tuple[dict[str, object], tuple[object, ...]]] = []
    real_solve = solve_linear_system

    def spy_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        result = real_solve(matrix, rhs, **kwargs)
        calls.append((dict(kwargs), tuple(result.attempt_trace)))
        return result

    monkeypatch.setattr(sweep_run_module, "solve_linear_system", spy_solve)

    result = run_sweep(frequencies, layout, _zero_assemble_point, rf_request=request)
    assert result.n_points == _EXPECTED_POINTS
    assert list(result.status.astype(str)) == ["pass", "pass"]
    assert result.rf_payloads is not None
    for metric in request.metrics:
        payload = result.rf_payloads.get(metric)
        assert payload is not None
        assert list(payload.status.astype(str)) == ["pass", "pass"]

    mna_calls = [(kwargs, trace) for kwargs, trace in calls if "node_voltage_count" in kwargs]
    conversion_calls = [(kwargs, trace) for kwargs, trace in calls if "run_config" in kwargs]
    assert mna_calls
    assert all(kwargs["node_voltage_count"] == layout.n_nodes for kwargs, _ in mna_calls)
    assert all(not _has_node_count_unavailable_skip(trace) for _, trace in mna_calls)

    assert conversion_calls
    assert all(
        getattr(kwargs["run_config"], "enable_gmin", None) is False
        for kwargs, _ in conversion_calls
    )


def test_standalone_rf_extractors_keep_mna_gmin_eligible_and_point_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    calls: list[tuple[str, dict[str, object], tuple[object, ...]]] = []
    real_y_solve = y_params_module.solve_linear_system
    real_z_solve = z_params_module.solve_linear_system

    def spy_y_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        result = real_y_solve(matrix, rhs, **kwargs)
        calls.append(("y", dict(kwargs), tuple(result.attempt_trace)))
        return result

    def spy_z_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        result = real_z_solve(matrix, rhs, **kwargs)
        calls.append(("z", dict(kwargs), tuple(result.attempt_trace)))
        return result

    monkeypatch.setattr(y_params_module, "solve_linear_system", spy_y_solve)
    monkeypatch.setattr(z_params_module, "solve_linear_system", spy_z_solve)

    y_result = extract_y_parameters(frequencies, ports, _zero_assemble_point)
    z_result = extract_z_parameters(frequencies, ports, _zero_assemble_point)
    impedance_result = extract_zin_zout(frequencies, ports, _zero_assemble_point)

    assert list(y_result.status.astype(str)) == ["pass", "pass"]
    assert list(z_result.status.astype(str)) == ["pass", "pass"]
    assert list(impedance_result.status.astype(str)) == ["pass", "pass"]
    assert len(y_result.diagnostics_by_point) == _EXPECTED_POINTS
    assert len(z_result.diagnostics_by_point) == _EXPECTED_POINTS
    assert len(impedance_result.diagnostics_by_point) == _EXPECTED_POINTS

    assert calls
    assert {"y", "z"} <= {source for source, _, _ in calls}
    mna_calls = [(kwargs, trace) for _, kwargs, trace in calls if "node_voltage_count" in kwargs]
    assert mna_calls
    assert all(kwargs["node_voltage_count"] == 1 for kwargs, _ in mna_calls)
    assert all(not _has_node_count_unavailable_skip(trace) for _, trace in mna_calls)

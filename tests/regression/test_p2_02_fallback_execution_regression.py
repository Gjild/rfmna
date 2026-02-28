from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.sweep_engine.run as sweep_run_module
from rfmna.rf_metrics import (
    PortBoundary,
    extract_y_parameters,
    extract_z_parameters,
    extract_zin_zout,
)
from rfmna.solver import FallbackRunConfig, SolveResult, solve_linear_system
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.regression
_EXPECTED_POINTS = 2


def test_sweep_and_rf_paths_keep_mna_gmin_eligible_and_conversion_gmin_disabled(
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

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = run_sweep(frequencies, layout, assemble, rf_request=request)
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
    assert all(kwargs["node_voltage_count"] == 1 for kwargs, _ in mna_calls)
    for _, trace in mna_calls:
        assert not any(
            row.stage == "gmin"
            and row.stage_state == "skipped"
            and row.skip_reason == "node_voltage_count_unavailable"
            for row in trace
        )

    assert conversion_calls
    assert all(
        isinstance(kwargs["run_config"], FallbackRunConfig) for kwargs, _ in conversion_calls
    )
    assert all(kwargs["run_config"].enable_gmin is False for kwargs, _ in conversion_calls)


def test_standalone_rf_extractors_preserve_point_order_and_avoid_node_count_unavailable() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    y_result = extract_y_parameters(frequencies, ports, assemble)
    z_result = extract_z_parameters(frequencies, ports, assemble)
    zz_result = extract_zin_zout(frequencies, ports, assemble)

    assert list(y_result.status.astype(str)) == ["pass", "pass"]
    assert list(z_result.status.astype(str)) == ["pass", "pass"]
    assert list(zz_result.status.astype(str)) == ["pass", "pass"]
    assert len(y_result.diagnostics_by_point) == _EXPECTED_POINTS
    assert len(z_result.diagnostics_by_point) == _EXPECTED_POINTS
    assert len(zz_result.diagnostics_by_point) == _EXPECTED_POINTS

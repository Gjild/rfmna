from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.sweep_engine.run as sweep_run_module
from rfmna.rf_metrics import PortBoundary
from rfmna.solver import FallbackRunConfig, SolveResult, solve_linear_system
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.unit

_EXPECTED_POINTS = 2


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _pass_assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
    y_block = np.asarray([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128)
    return (
        csc_matrix(y_block),
        np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
    )


def _assemble_with_fail(point_index: int, _: float) -> tuple[csc_matrix, np.ndarray]:
    if point_index == 1:
        raise RuntimeError("intentional point failure")
    return _pass_assemble(point_index, 0.0)


def test_non_rf_callers_remain_compatible_and_rf_payloads_default_to_none() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)

    result = run_sweep(frequencies, layout, _pass_assemble)
    assert result.n_points == _EXPECTED_POINTS
    assert result.V_nodes.shape == (_EXPECTED_POINTS, 2)
    assert result.I_aux.shape == (_EXPECTED_POINTS, 0)
    assert result.rf_payloads is None


def test_rf_payloads_attach_only_requested_metrics_with_canonical_order() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        ),
        metrics=("zout", "y", "zin"),
    )

    result = run_sweep(frequencies, layout, _pass_assemble, rf_request=request)
    assert result.rf_payloads is not None
    assert result.rf_payloads.metric_names == ("y", "zin", "zout")
    assert result.rf_payloads.get("y") is not None
    assert result.rf_payloads.get("zin") is not None
    assert result.rf_payloads.get("zout") is not None
    assert result.rf_payloads.get("z") is None
    assert result.rf_payloads.get("s") is None


def test_base_sweep_arrays_and_fail_semantics_unchanged_with_rf_request() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("y", "z", "s", "zin", "zout"),
    )

    baseline = run_sweep(frequencies, layout, _assemble_with_fail)
    with_rf = run_sweep(frequencies, layout, _assemble_with_fail, rf_request=request)

    np.testing.assert_allclose(with_rf.V_nodes, baseline.V_nodes, equal_nan=True)
    np.testing.assert_allclose(with_rf.I_aux, baseline.I_aux, equal_nan=True)
    np.testing.assert_allclose(with_rf.res_l2, baseline.res_l2, equal_nan=True)
    np.testing.assert_allclose(with_rf.res_linf, baseline.res_linf, equal_nan=True)
    np.testing.assert_allclose(with_rf.res_rel, baseline.res_rel, equal_nan=True)
    np.testing.assert_allclose(with_rf.cond_ind, baseline.cond_ind, equal_nan=True)
    assert with_rf.status.tolist() == baseline.status.tolist()
    assert with_rf.diagnostics_by_point == baseline.diagnostics_by_point


def test_rf_payloads_align_indices_and_apply_sentinel_policy_for_failed_points() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("y", "z", "s", "zin", "zout"),
    )
    fail_index = 1

    result = run_sweep(frequencies, layout, _assemble_with_fail, rf_request=request)
    assert result.rf_payloads is not None
    assert result.status.tolist() == ["pass", "fail", "pass"]

    y_payload = result.rf_payloads.get("y")
    assert y_payload is not None
    assert y_payload.y.shape == (3, 2, 2)
    assert list(y_payload.status.astype(str)) == ["pass", "fail", "pass"]
    assert np.isnan(y_payload.y[fail_index].real).all()
    assert np.isnan(y_payload.y[fail_index].imag).all()
    assert y_payload.diagnostics_by_point[fail_index]

    z_payload = result.rf_payloads.get("z")
    assert z_payload is not None
    assert z_payload.z.shape == (3, 2, 2)
    assert list(z_payload.status.astype(str)) == ["pass", "fail", "pass"]
    assert np.isnan(z_payload.z[fail_index].real).all()
    assert np.isnan(z_payload.z[fail_index].imag).all()
    assert z_payload.diagnostics_by_point[fail_index]

    s_payload = result.rf_payloads.get("s")
    assert s_payload is not None
    assert s_payload.s.shape == (3, 2, 2)
    assert list(s_payload.status.astype(str)) == ["pass", "fail", "pass"]
    assert np.isnan(s_payload.s[fail_index].real).all()
    assert np.isnan(s_payload.s[fail_index].imag).all()
    assert s_payload.diagnostics_by_point[fail_index]

    zin_payload = result.rf_payloads.get("zin")
    assert zin_payload is not None
    assert zin_payload.metric_name == "zin"
    assert zin_payload.values.shape == (3,)
    assert list(zin_payload.status.astype(str)) == ["pass", "fail", "pass"]
    assert np.isnan(zin_payload.values[fail_index].real)
    assert np.isnan(zin_payload.values[fail_index].imag)
    assert zin_payload.diagnostics_by_point[fail_index]

    zout_payload = result.rf_payloads.get("zout")
    assert zout_payload is not None
    assert zout_payload.metric_name == "zout"
    assert zout_payload.values.shape == (3,)
    assert list(zout_payload.status.astype(str)) == ["pass", "fail", "pass"]
    assert np.isnan(zout_payload.values[fail_index].real)
    assert np.isnan(zout_payload.values[fail_index].imag)
    assert zout_payload.diagnostics_by_point[fail_index]


def test_rf_payload_ordering_and_values_are_deterministic_under_request_permutations() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)
    request_one = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        ),
        metrics=("zout", "s", "y", "zin", "z"),
    )
    request_two = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("zin", "z", "y", "s", "zout"),
    )

    first = run_sweep(frequencies, layout, _assemble_with_fail, rf_request=request_one)
    second = run_sweep(frequencies, layout, _assemble_with_fail, rf_request=request_two)

    assert first.rf_payloads is not None
    assert second.rf_payloads is not None
    assert first.rf_payloads.metric_names == ("y", "z", "s", "zin", "zout")
    assert second.rf_payloads.metric_names == ("y", "z", "s", "zin", "zout")
    assert first.rf_payloads.metric_names == second.rf_payloads.metric_names

    for metric_name in first.rf_payloads.metric_names:
        left = first.rf_payloads.get(metric_name)
        right = second.rf_payloads.get(metric_name)
        assert left is not None
        assert right is not None
        if metric_name in {"zin", "zout"}:
            np.testing.assert_allclose(left.values, right.values, equal_nan=True)
        elif metric_name == "y":
            np.testing.assert_allclose(left.y, right.y, equal_nan=True)
        elif metric_name == "z":
            np.testing.assert_allclose(left.z, right.z, equal_nan=True)
        else:
            np.testing.assert_allclose(left.s, right.s, equal_nan=True)
        assert tuple(_diag_json(point) for point in left.diagnostics_by_point) == tuple(
            _diag_json(point) for point in right.diagnostics_by_point
        )


def test_default_solver_paths_propagate_node_count_and_disable_conversion_gmin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=0)
    request = SweepRFRequest(
        ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
        metrics=("y", "z", "s", "zin", "zout"),
    )
    calls: list[dict[str, object]] = []
    real_solve = solve_linear_system

    def spy_solve(matrix: object, rhs: np.ndarray, **kwargs: object) -> SolveResult:
        calls.append(dict(kwargs))
        return real_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr(sweep_run_module, "solve_linear_system", spy_solve)

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[0.0 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    result = run_sweep(frequencies, layout, assemble, rf_request=request)
    assert list(result.status.astype(str)) == ["pass"]
    assert result.rf_payloads is not None
    assert calls

    mna_calls = [call for call in calls if "node_voltage_count" in call]
    conversion_calls = [call for call in calls if "run_config" in call]
    assert mna_calls
    assert all(call.get("node_voltage_count") == layout.n_nodes for call in mna_calls)
    assert conversion_calls
    assert all(isinstance(call["run_config"], FallbackRunConfig) for call in conversion_calls)
    assert all(call["run_config"].enable_gmin is False for call in conversion_calls)


def test_custom_mna_solve_point_does_not_override_conversion_no_gmin_policy() -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=0)
    request = SweepRFRequest(
        ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
        metrics=("s",),
    )

    def assemble(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        return (
            csc_matrix(np.asarray([[-0.02 + 0.0j]], dtype=np.complex128)),
            np.asarray([0.0 + 0.0j], dtype=np.complex128),
        )

    def solve_point(matrix: object, rhs: np.ndarray) -> SolveResult:
        return solve_linear_system(matrix, rhs, node_voltage_count=1)

    result = run_sweep(frequencies, layout, assemble, solve_point=solve_point, rf_request=request)
    assert result.rf_payloads is not None
    s_payload = result.rf_payloads.get("s")
    assert s_payload is not None
    assert list(s_payload.status.astype(str)) == ["fail"]
    assert np.isnan(s_payload.s[0].real).all()
    assert np.isnan(s_payload.s[0].imag).all()
    assert [diag.code for diag in s_payload.diagnostics_by_point[0]] == [
        "E_NUM_S_CONVERSION_SINGULAR"
    ]

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics import PortBoundary
from rfmna.solver import (
    AttemptTraceRecord,
    BackendMetadata,
    BackendNotes,
    ResidualMetrics,
    SolveResult,
)
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.conformance


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


def _load_fixture(name: str) -> dict[str, object]:
    raw = (_fixtures_dir() / name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"fixture '{name}' must be a JSON object")
    return cast(dict[str, object], payload)


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _metadata() -> BackendMetadata:
    return BackendMetadata(
        backend_id="conformance",
        backend_name="conformance",
        python_version=None,
        scipy_version=None,
        permutation_row=None,
        permutation_col=None,
        pivot_order=None,
        success=True,
        failure_category=None,
        failure_code=None,
        failure_reason=None,
        failure_message=None,
        notes=BackendNotes(
            input_format="csr",
            normalized_format="csr",
            n_unknowns=0,
            nnz=0,
            pivot_profile="default",
            scaling_enabled=False,
        ),
    )


def _solve_result(status: str, x: np.ndarray | None) -> SolveResult:
    return SolveResult(
        x=x,
        residual=ResidualMetrics(res_l2=0.0, res_linf=0.0, res_rel=0.0),
        status=status,  # type: ignore[arg-type]
        backend=_metadata(),
        failure_category=None if status != "fail" else "numeric",
        failure_code="E_NUM_SOLVE_FAILED" if status == "fail" else None,
        failure_reason="fixture_planned" if status == "fail" else None,
        failure_message="fixture planned fail" if status == "fail" else None,
        cond_ind=1.0,
        warnings=(),
        attempt_trace=(
            AttemptTraceRecord(
                attempt_index=0,
                stage="baseline",
                stage_state="run",
                scaling_enabled=False,
                pivot_profile="default",
                gmin_value=None,
                success=status != "fail",
                backend_failure_category=None,
                backend_failure_reason=None,
                backend_failure_code=None,
                backend_failure_message=None,
                res_l2=0.0,
                res_linf=0.0,
                res_rel=0.0,
                status=status if status in {"pass", "degraded", "fail"} else None,
                skip_reason=None,
            ),
        ),
    )


def test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment() -> None:
    fixture = _load_fixture("rf_pass_degraded_fail_sweep_v1.json")
    frequencies = np.asarray(fixture["frequencies_hz"], dtype=np.float64)
    layout_obj = cast(dict[str, object], fixture["layout"])
    layout = SweepLayout(
        n_nodes=int(layout_obj["n_nodes"]),
        n_aux=int(layout_obj["n_aux"]),
    )
    statuses = tuple(str(value) for value in cast(list[object], fixture["point_statuses"]))
    fail_index = int(fixture["fail_point_index"])

    def assemble_point(index: int, frequency_hz: float) -> tuple[csr_matrix, np.ndarray]:
        del index, frequency_hz
        return csr_matrix(np.eye(layout.n_nodes + layout.n_aux, dtype=np.complex128)), np.zeros(
            layout.n_nodes + layout.n_aux,
            dtype=np.complex128,
        )

    def solve_point(A: csr_matrix, b: np.ndarray) -> SolveResult:
        del A, b
        idx = solve_point.calls
        solve_point.calls += 1
        status = statuses[idx]
        x = (
            None
            if status == "fail"
            else np.asarray([idx + 1.0 + 0.0j, idx + 5.0 + 0.0j], dtype=np.complex128)
        )
        return _solve_result(status, x)

    solve_point.calls = 0  # type: ignore[attr-defined]
    result = run_sweep(frequencies, layout, assemble_point, solve_point=solve_point)

    assert result.n_points == len(frequencies)
    assert result.status.tolist() == list(statuses)
    assert result.V_nodes.shape == (len(frequencies), layout.n_nodes)
    assert result.I_aux.shape == (len(frequencies), layout.n_aux)

    assert math.isnan(result.V_nodes[fail_index, 0].real)
    assert math.isnan(result.V_nodes[fail_index, 0].imag)
    assert math.isnan(result.I_aux[fail_index, 0].real)
    assert math.isnan(result.I_aux[fail_index, 0].imag)
    assert math.isnan(result.res_l2[fail_index])
    assert math.isnan(result.res_linf[fail_index])
    assert math.isnan(result.res_rel[fail_index])
    assert math.isnan(result.cond_ind[fail_index])

    for point_index, point_diags in enumerate(result.diagnostics_by_point):
        for diag in point_diags:
            assert diag.point_index == point_index
            assert diag.frequency_index == point_index


def test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic() -> None:
    fixture = _load_fixture("rf_payload_sentinel_ordering_v1.json")
    frequencies = np.asarray(fixture["frequencies_hz"], dtype=np.float64)
    layout_obj = cast(dict[str, object], fixture["layout"])
    layout = SweepLayout(
        n_nodes=int(layout_obj["n_nodes"]),
        n_aux=int(layout_obj["n_aux"]),
    )
    fail_assemble_index = int(fixture["fail_assemble_index"])
    expected_status = [str(value) for value in cast(list[object], fixture["expected_status"])]
    expected_metric_names = tuple(
        str(value) for value in cast(list[object], fixture["expected_canonical_metric_names"])
    )

    ports_payload = cast(list[dict[str, object]], fixture["ports"])
    ports = tuple(
        PortBoundary(
            port_id=str(port["port_id"]),
            p_plus_index=cast(int | None, port["p_plus_index"]),
            p_minus_index=cast(int | None, port["p_minus_index"]),
        )
        for port in ports_payload
    )

    permutations = cast(list[list[object]], fixture["metrics_permutations"])
    first_metrics = tuple(str(value) for value in permutations[0])
    second_metrics = tuple(str(value) for value in permutations[1])

    def assemble_point(point_index: int, frequency_hz: float) -> tuple[csc_matrix, np.ndarray]:
        del frequency_hz
        if point_index == fail_assemble_index:
            raise RuntimeError("fixture assembly fail")
        y_block = np.asarray(
            [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128
        )
        return (
            csc_matrix(y_block),
            np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        )

    first = run_sweep(
        frequencies,
        layout,
        assemble_point,
        rf_request=SweepRFRequest(ports=ports, metrics=first_metrics),
    )
    second = run_sweep(
        frequencies,
        layout,
        assemble_point,
        rf_request=SweepRFRequest(ports=tuple(reversed(ports)), metrics=second_metrics),
    )

    assert first.rf_payloads is not None
    assert second.rf_payloads is not None
    assert first.status.tolist() == expected_status
    assert second.status.tolist() == expected_status
    assert first.rf_payloads.metric_names == expected_metric_names
    assert second.rf_payloads.metric_names == expected_metric_names

    for metric_name in expected_metric_names:
        left = first.rf_payloads.get(metric_name)
        right = second.rf_payloads.get(metric_name)
        assert left is not None
        assert right is not None
        if metric_name in {"zin", "zout"}:
            np.testing.assert_allclose(left.values, right.values, equal_nan=True)
            assert np.isnan(left.values[fail_assemble_index].real)
            assert np.isnan(left.values[fail_assemble_index].imag)
        elif metric_name == "y":
            np.testing.assert_allclose(left.y, right.y, equal_nan=True)
            assert np.isnan(left.y[fail_assemble_index].real).all()
            assert np.isnan(left.y[fail_assemble_index].imag).all()
        elif metric_name == "z":
            np.testing.assert_allclose(left.z, right.z, equal_nan=True)
            assert np.isnan(left.z[fail_assemble_index].real).all()
            assert np.isnan(left.z[fail_assemble_index].imag).all()
        else:
            np.testing.assert_allclose(left.s, right.s, equal_nan=True)
            assert np.isnan(left.s[fail_assemble_index].real).all()
            assert np.isnan(left.s[fail_assemble_index].imag).all()

        assert list(left.status.astype(str)) == expected_status
        assert list(right.status.astype(str)) == expected_status
        assert left.diagnostics_by_point[fail_assemble_index]
        assert right.diagnostics_by_point[fail_assemble_index]
        assert tuple(_diag_json(point) for point in left.diagnostics_by_point) == tuple(
            _diag_json(point) for point in right.diagnostics_by_point
        )

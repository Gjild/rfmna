from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import (
    AttemptTraceRecord,
    BackendMetadata,
    BackendNotes,
    ResidualMetrics,
    SolveResult,
    SolveWarning,
)
from rfmna.sweep_engine import SweepLayout, run_sweep

pytestmark = pytest.mark.conformance

EXPECTED_ORDERED_DIAGNOSTICS = 2


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


def _solve_result(
    status: str,
    x: np.ndarray | None,
    *,
    warnings: tuple[SolveWarning, ...] = (),
) -> SolveResult:
    return SolveResult(
        x=x,
        residual=ResidualMetrics(res_l2=0.0, res_linf=0.0, res_rel=0.0),
        status=status,  # type: ignore[arg-type]
        backend=_metadata(),
        failure_category=None if status != "fail" else "numeric",
        failure_code="E_NUM_SOLVE_FAILED" if status == "fail" else None,
        failure_reason="planned" if status == "fail" else None,
        failure_message="planned fail" if status == "fail" else None,
        cond_ind=1.0,
        warnings=warnings,
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


def test_fail_point_sentinel_policy_and_full_point_presence() -> None:
    freq = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=1, n_aux=1)

    def assemble_point(index: int, frequency_hz: float) -> tuple[csr_matrix, np.ndarray]:
        del index, frequency_hz
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    statuses = ("pass", "fail", "degraded")

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
    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)

    assert result.n_points == len(freq)
    assert result.status.tolist() == ["pass", "fail", "degraded"]
    assert result.V_nodes.shape == (3, 1)
    assert result.I_aux.shape == (3, 1)

    fail_index = 1
    assert math.isnan(result.V_nodes[fail_index, 0].real)
    assert math.isnan(result.V_nodes[fail_index, 0].imag)
    assert math.isnan(result.I_aux[fail_index, 0].real)
    assert math.isnan(result.I_aux[fail_index, 0].imag)
    assert math.isnan(result.res_l2[fail_index])
    assert math.isnan(result.res_linf[fail_index])
    assert math.isnan(result.res_rel[fail_index])
    assert math.isnan(result.cond_ind[fail_index])
    assert result.diagnostics_by_point[fail_index]

    for index, status in enumerate(result.status.tolist()):
        if status == "fail":
            assert len(result.diagnostics_by_point[index]) >= 1
            for diag in result.diagnostics_by_point[index]:
                assert diag.element_id
                assert diag.solver_stage in {"assemble", "solve", "postprocess"}
                assert diag.frequency_index == index


def test_fail_point_diagnostics_sort_severity_before_stage() -> None:
    freq = np.asarray([1.0], dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)

    def assemble_point(index: int, frequency_hz: float) -> tuple[csr_matrix, np.ndarray]:
        del index, frequency_hz
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    def solve_point(A: csr_matrix, b: np.ndarray) -> SolveResult:
        del A, b
        return _solve_result(
            "pass",
            np.asarray([1.0 + 0.0j], dtype=np.complex128),  # wrong shape -> postprocess error
            warnings=(SolveWarning(code="W_NUM_ILL_CONDITIONED", message="warn"),),
        )

    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)
    ordered = result.diagnostics_by_point[0]
    assert len(ordered) == EXPECTED_ORDERED_DIAGNOSTICS
    assert ordered[0].severity == "error"
    assert ordered[0].solver_stage == "postprocess"
    assert ordered[1].severity == "warning"
    assert ordered[1].solver_stage == "solve"

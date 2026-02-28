from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage, diagnostic_sort_key
from rfmna.solver import (
    AttemptTraceRecord,
    BackendMetadata,
    BackendNotes,
    ResidualMetrics,
    SolveResult,
    SolveWarning,
)
from rfmna.sweep_engine import SweepDiagnostic, SweepLayout, run_sweep, sweep_diagnostic_sort_key

pytestmark = pytest.mark.unit

EXPECTED_ORDERED_DIAGNOSTICS = 2


def _backend_metadata() -> BackendMetadata:
    return BackendMetadata(
        backend_id="test_backend",
        backend_name="test_backend",
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


def _to_diagnostic_event(diag: SweepDiagnostic) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=diag.code,
        severity=Severity.ERROR if diag.severity == "error" else Severity.WARNING,
        message=diag.message,
        suggested_action=diag.suggested_action,
        solver_stage=SolverStage(diag.solver_stage),
        element_id=diag.element_id,
        frequency_hz=diag.frequency_hz,
        frequency_index=diag.frequency_index,
        sweep_index=diag.sweep_index,
        witness=diag.witness,
    )


def _solve_result(  # noqa: PLR0913
    *,
    status: str,
    x: NDArray[np.complex128] | None,
    res_l2: float,
    res_linf: float,
    res_rel: float,
    cond_ind: float,
    warnings: tuple[SolveWarning, ...] = (),
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> SolveResult:
    return SolveResult(
        x=x,
        residual=ResidualMetrics(res_l2=res_l2, res_linf=res_linf, res_rel=res_rel),
        status=status,  # type: ignore[arg-type]
        backend=_backend_metadata(),
        failure_category=None if status != "fail" else "numeric",
        failure_code=failure_code,
        failure_reason=None if status != "fail" else "test_failure",
        failure_message=failure_message,
        cond_ind=cond_ind,
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
                backend_failure_code=failure_code,
                backend_failure_message=failure_message,
                res_l2=res_l2,
                res_linf=res_linf,
                res_rel=res_rel,
                status=status if status in {"pass", "degraded", "fail"} else None,
                skip_reason=None,
            ),
        ),
    )


def test_all_pass_sweep_shapes_order_and_mapping() -> None:
    layout = SweepLayout(n_nodes=2, n_aux=1)
    freq = np.asarray([1.0, 2.0, 5.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(3, dtype=np.complex128)), np.zeros(3, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        value = float(solve_point.calls)
        solve_point.calls += 1
        x = np.asarray([value + 1j, value + 2j, value + 3j], dtype=np.complex128)
        return _solve_result(
            status="pass",
            x=x,
            res_l2=value + 0.1,
            res_linf=value + 0.2,
            res_rel=value + 0.3,
            cond_ind=value + 0.4,
        )

    solve_point.calls = 0  # type: ignore[attr-defined]
    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)

    assert result.n_points == len(freq)
    assert result.V_nodes.shape == (3, 2)
    assert result.I_aux.shape == (3, 1)
    assert result.res_l2.shape == (3,)
    assert result.status.tolist() == ["pass", "pass", "pass"]
    np.testing.assert_allclose(result.V_nodes[0], np.asarray([0.0 + 1j, 0.0 + 2j]))
    np.testing.assert_allclose(result.I_aux[2], np.asarray([2.0 + 3j]))


def test_single_fail_sentinelization_and_neighbor_preservation() -> None:
    layout = SweepLayout(n_nodes=1, n_aux=1)
    freq = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        i = solve_point.calls
        solve_point.calls += 1
        if i == 1:
            return _solve_result(
                status="fail",
                x=np.asarray([9.0 + 0.0j, 9.0 + 0.0j], dtype=np.complex128),
                res_l2=9.0,
                res_linf=9.0,
                res_rel=9.0,
                cond_ind=0.1,
                failure_code="E_NUM_SOLVE_FAILED",
                failure_message="planned fail",
            )
        x = np.asarray([complex(i + 1, 0.0), complex(i + 10, 0.0)], dtype=np.complex128)
        status = "pass" if i == 0 else "degraded"
        return _solve_result(
            status=status,
            x=x,
            res_l2=0.1 + i,
            res_linf=0.2 + i,
            res_rel=0.3 + i,
            cond_ind=0.9 - i,
        )

    solve_point.calls = 0  # type: ignore[attr-defined]
    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)

    assert result.status.tolist() == ["pass", "fail", "degraded"]
    assert np.isfinite(result.V_nodes[0, 0].real)
    assert math.isnan(result.V_nodes[1, 0].real)
    assert math.isnan(result.I_aux[1, 0].real)
    assert math.isnan(result.res_l2[1])
    assert math.isnan(result.res_linf[1])
    assert math.isnan(result.res_rel[1])
    assert math.isnan(result.cond_ind[1])
    assert result.diagnostics_by_point[1]
    assert np.isfinite(result.V_nodes[2, 0].real)


def test_mixed_statuses_only_fail_rows_are_sentinelized() -> None:
    layout = SweepLayout(n_nodes=2, n_aux=0)
    freq = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    statuses = ("pass", "degraded", "fail", "pass")

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        i = solve_point.calls
        solve_point.calls += 1
        status = statuses[i]
        return _solve_result(
            status=status,
            x=np.asarray([1.0 + i, 2.0 + i], dtype=np.complex128),
            res_l2=0.1,
            res_linf=0.2,
            res_rel=0.3,
            cond_ind=0.4,
            failure_code="E_NUM_SOLVE_FAILED" if status == "fail" else None,
            failure_message="bad point" if status == "fail" else None,
        )

    solve_point.calls = 0  # type: ignore[attr-defined]
    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)

    assert result.status.tolist() == ["pass", "degraded", "fail", "pass"]
    assert math.isnan(result.V_nodes[2, 0].real)
    assert np.isfinite(result.V_nodes[1, 0].real)
    assert np.isfinite(result.V_nodes[3, 0].real)


def test_continue_after_fail_executes_later_points() -> None:
    layout = SweepLayout(n_nodes=1, n_aux=0)
    freq = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    executed: list[int] = []

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        executed.append(index)
        del frequency_hz
        if index == 0:
            raise RuntimeError("assemble point failure")
        return csr_matrix(np.eye(1, dtype=np.complex128)), np.zeros(1, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        return _solve_result(
            status="pass",
            x=np.asarray([1.0 + 0.0j], dtype=np.complex128),
            res_l2=0.1,
            res_linf=0.2,
            res_rel=0.3,
            cond_ind=0.4,
        )

    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)
    assert executed == [0, 1, 2]
    assert result.status.tolist() == ["fail", "pass", "pass"]


def test_assembler_and_orchestration_exception_mapping() -> None:
    layout = SweepLayout(n_nodes=2, n_aux=0)
    freq = np.asarray([1.0, 2.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del frequency_hz
        if index == 0:
            raise ValueError("assembly exploded")
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        return _solve_result(
            status="pass",
            x=np.asarray(
                [1.0 + 0.0j], dtype=np.complex128
            ),  # wrong length triggers postprocess fail
            res_l2=0.1,
            res_linf=0.2,
            res_rel=0.3,
            cond_ind=0.4,
        )

    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)
    assert result.status.tolist() == ["fail", "fail"]
    assert result.diagnostics_by_point[0][0].code == "E_NUM_SOLVE_FAILED"
    assert result.diagnostics_by_point[0][0].solver_stage == "assemble"
    assert result.diagnostics_by_point[1][0].code == "E_NUM_SOLVE_FAILED"
    assert result.diagnostics_by_point[1][0].solver_stage == "postprocess"


def test_deterministic_reproducibility() -> None:
    layout = SweepLayout(n_nodes=1, n_aux=0)
    freq = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(1, dtype=np.complex128)), np.zeros(1, dtype=np.complex128)

    @dataclass(frozen=True, slots=True)
    class ScriptedSolve:
        outcomes: tuple[str, ...]

        def __call__(self, A: csr_matrix, b: NDArray[np.complex128]) -> SolveResult:
            del A, b
            idx = ScriptedSolve.calls
            ScriptedSolve.calls += 1
            status = self.outcomes[idx]
            return _solve_result(
                status=status,
                x=np.asarray([complex(idx + 1, 0.0)], dtype=np.complex128),
                res_l2=float(idx),
                res_linf=float(idx),
                res_rel=float(idx),
                cond_ind=float(idx),
                failure_code="E_NUM_SOLVE_FAILED" if status == "fail" else None,
                failure_message="x" if status == "fail" else None,
            )

    ScriptedSolve.calls = 0  # type: ignore[attr-defined]
    first = run_sweep(
        freq, layout, assemble_point, solve_point=ScriptedSolve(("pass", "fail", "degraded"))
    )
    ScriptedSolve.calls = 0  # type: ignore[attr-defined]
    second = run_sweep(
        freq, layout, assemble_point, solve_point=ScriptedSolve(("pass", "fail", "degraded"))
    )

    np.testing.assert_allclose(first.V_nodes, second.V_nodes, equal_nan=True)
    np.testing.assert_allclose(first.res_rel, second.res_rel, equal_nan=True)
    assert first.status.tolist() == second.status.tolist()
    assert first.diagnostics_by_point == second.diagnostics_by_point


def test_empty_sweep_returns_empty_shaped_outputs() -> None:
    layout = SweepLayout(n_nodes=2, n_aux=1)
    freq = np.asarray([], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        raise AssertionError(
            f"assemble_point should not be called for empty sweep: {index}, {frequency_hz}"
        )

    result = run_sweep(freq, layout, assemble_point)
    assert result.n_points == 0
    assert result.V_nodes.shape == (0, 2)
    assert result.I_aux.shape == (0, 1)
    assert result.res_l2.shape == (0,)
    assert result.diagnostics_by_point == ()


def test_input_purity() -> None:
    layout = SweepLayout(n_nodes=1, n_aux=0)
    freq = np.asarray([1.0, 2.0], dtype=np.float64)
    freq_before = freq.copy()

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(1, dtype=np.complex128)), np.zeros(1, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        return _solve_result(
            status="pass",
            x=np.asarray([1.0 + 0.0j], dtype=np.complex128),
            res_l2=0.0,
            res_linf=0.0,
            res_rel=0.0,
            cond_ind=1.0,
        )

    _ = run_sweep(freq, layout, assemble_point, solve_point=solve_point)
    np.testing.assert_array_equal(freq, freq_before)


def test_fail_diagnostics_include_point_and_frequency_context() -> None:
    layout = SweepLayout(n_nodes=1, n_aux=0)
    freq = np.asarray([10.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(1, dtype=np.complex128)), np.zeros(1, dtype=np.complex128)

    def solve_point(
        A: csr_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        return _solve_result(
            status="fail",
            x=None,
            res_l2=math.nan,
            res_linf=math.nan,
            res_rel=math.nan,
            cond_ind=math.nan,
            failure_code="E_NUM_SOLVE_FAILED",
            failure_message="planned fail",
        )

    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)
    diag = result.diagnostics_by_point[0][0]
    assert diag.point_index == 0
    assert diag.frequency_hz == pytest.approx(10.0)
    assert diag.frequency_index == 0
    assert diag.element_id == "solver"
    assert diag.solver_stage == "solve"


def test_fail_point_diagnostics_sorted_severity_before_stage() -> None:
    layout = SweepLayout(n_nodes=2, n_aux=0)
    freq = np.asarray([1.0], dtype=np.float64)

    def assemble_point(
        index: int, frequency_hz: float
    ) -> tuple[csr_matrix, NDArray[np.complex128]]:
        del index, frequency_hz
        return csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2, dtype=np.complex128)

    def solve_point(A: csr_matrix, b: NDArray[np.complex128]) -> SolveResult:
        del A, b
        return _solve_result(
            status="pass",
            x=np.asarray([1.0 + 0.0j], dtype=np.complex128),  # wrong size -> postprocess error
            res_l2=0.0,
            res_linf=0.0,
            res_rel=0.0,
            cond_ind=1.0,
            warnings=(SolveWarning(code="W_NUM_ILL_CONDITIONED", message="warn"),),
        )

    result = run_sweep(freq, layout, assemble_point, solve_point=solve_point)

    assert result.status.tolist() == ["fail"]
    ordered = result.diagnostics_by_point[0]
    assert len(ordered) == EXPECTED_ORDERED_DIAGNOSTICS
    assert ordered[0].severity == "error"
    assert ordered[0].solver_stage == "postprocess"
    assert ordered[0].code == "E_NUM_SOLVE_FAILED"
    assert ordered[1].severity == "warning"
    assert ordered[1].solver_stage == "solve"
    assert ordered[1].code == "W_NUM_ILL_CONDITIONED"


def test_sweep_diagnostic_sort_key_matches_canonical_diagnostics_sorting() -> None:
    diagnostics = (
        SweepDiagnostic(
            code="W_NUM_ILL_CONDITIONED",
            severity="warning",
            message="warn later stage",
            suggested_action="act",
            solver_stage="solve",
            point_index=0,
            frequency_hz=1.0,
            element_id="solver",
            witness={"k": 2},
        ),
        SweepDiagnostic(
            code="W_ALPHA",
            severity="warning",
            message="warn earlier stage",
            suggested_action="act",
            solver_stage="parse",
            point_index=0,
            frequency_hz=1.0,
            element_id="sweep_engine",
            witness={"k": 1},
        ),
        SweepDiagnostic(
            code="E_NUM_SOLVE_FAILED",
            severity="error",
            message="error wins",
            suggested_action="act",
            solver_stage="postprocess",
            point_index=0,
            frequency_hz=1.0,
            element_id="solver",
            witness={"k": 3},
        ),
    )

    expected = tuple(
        sorted(
            diagnostics,
            key=lambda diag: diagnostic_sort_key(_to_diagnostic_event(diag)),
        )
    )
    current = tuple(sorted(diagnostics, key=sweep_diagnostic_sort_key))

    assert current == expected


def test_sweep_diagnostic_sort_key_is_permutation_stable() -> None:
    baseline_input = (
        SweepDiagnostic(
            code="W_B",
            severity="warning",
            message="warn b",
            suggested_action="act",
            solver_stage="solve",
            point_index=0,
            frequency_hz=1.0,
            element_id="solver",
            witness={"x": 2},
        ),
        SweepDiagnostic(
            code="E_A",
            severity="error",
            message="error a",
            suggested_action="act",
            solver_stage="postprocess",
            point_index=0,
            frequency_hz=1.0,
            element_id="solver",
            witness={"x": 1},
        ),
        SweepDiagnostic(
            code="W_A",
            severity="warning",
            message="warn a",
            suggested_action="act",
            solver_stage="parse",
            point_index=0,
            frequency_hz=1.0,
            element_id="sweep_engine",
            witness={"x": 0},
        ),
    )
    expected = tuple(sorted(baseline_input, key=sweep_diagnostic_sort_key))
    rng = Random(0)

    for _ in range(20):
        permuted = list(baseline_input)
        rng.shuffle(permuted)
        assert tuple(sorted(permuted, key=sweep_diagnostic_sort_key)) == expected

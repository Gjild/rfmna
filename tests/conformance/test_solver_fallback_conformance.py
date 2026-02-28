from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import (
    BackendMetadata,
    BackendNotes,
    BackendSolveOptions,
    BackendSolveResult,
    SolverBackend,
    load_solver_threshold_config,
    solve_linear_system,
)

pytestmark = pytest.mark.conformance
ALT_SUCCESS_CALL_COUNT = 2


@dataclass(frozen=True, slots=True)
class _Response:
    success: bool


@dataclass(frozen=True, slots=True)
class _ConstEstimator:
    value: float | None

    def estimate(self, A: object) -> float | None:
        del A
        return self.value


class _Backend(SolverBackend):
    def __init__(self, responses: tuple[_Response, ...]) -> None:
        self._responses = responses
        self.call_count = 0

    def solve(
        self,
        A: object,
        b: np.ndarray,
        *,
        options: BackendSolveOptions | None = None,
    ) -> BackendSolveResult:
        response = (
            self._responses[self.call_count]
            if self.call_count < len(self._responses)
            else _Response(False)
        )
        self.call_count += 1
        sparse = csr_matrix(A)
        selected_options = options if options is not None else BackendSolveOptions()
        metadata = BackendMetadata(
            backend_id="conformance",
            backend_name="conformance",
            python_version=None,
            scipy_version=None,
            permutation_row=None,
            permutation_col=None,
            pivot_order=None,
            success=response.success,
            failure_category=None if response.success else "numeric",
            failure_code=None if response.success else "E_NUM_SOLVE_FAILED",
            failure_reason=None if response.success else "planned_failure",
            failure_message=None if response.success else "planned failure",
            notes=BackendNotes(
                input_format="csr",
                normalized_format="csr",
                n_unknowns=int(b.shape[0]),
                nnz=int(sparse.nnz),
                pivot_profile=selected_options.pivot_profile,
                scaling_enabled=selected_options.scaling_enabled,
            ),
        )
        if response.success:
            return BackendSolveResult(x=np.asarray(b, dtype=np.complex128), metadata=metadata)
        return BackendSolveResult(x=None, metadata=metadata)


def test_threshold_artifact_declares_normative_fallback_order() -> None:
    defaults = load_solver_threshold_config()
    assert defaults.fallback_order == ("baseline", "alt_pivot", "scaling", "gmin", "final_fail")


def test_normative_ladder_order_and_no_hidden_retries_and_final_status() -> None:
    defaults = load_solver_threshold_config()
    A = csr_matrix(np.eye(2, dtype=np.complex128))
    b = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    backend = _Backend((_Response(False), _Response(True)))

    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(0.5),
    )

    expected = (
        ["baseline", "alt_pivot", "scaling"]
        + (["gmin"] * len(defaults.gmin_values))
        + ["final_fail"]
    )
    assert [row.stage for row in result.attempt_trace] == expected
    assert backend.call_count == ALT_SUCCESS_CALL_COUNT
    assert result.status == "pass"
    assert result.attempt_trace[1].success is True


def test_per_point_reset_and_gmin_restart() -> None:
    defaults = load_solver_threshold_config()
    A = csr_matrix(np.eye(1, dtype=np.complex128))
    b = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    attempts = 3 + len(defaults.gmin_values)

    left = solve_linear_system(
        A,
        b,
        backend=_Backend(tuple(_Response(False) for _ in range(attempts))),
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(0.2),
    )
    right = solve_linear_system(
        A,
        b,
        backend=_Backend(tuple(_Response(False) for _ in range(attempts))),
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(0.2),
    )

    left_gmin = [row.gmin_value for row in left.attempt_trace if row.stage == "gmin"]
    right_gmin = [row.gmin_value for row in right.attempt_trace if row.stage == "gmin"]
    assert left.attempt_trace[0].attempt_index == 0
    assert right.attempt_trace[0].attempt_index == 0
    assert left_gmin == list(defaults.gmin_values)
    assert right_gmin == list(defaults.gmin_values)


def test_unavailable_condition_indicator_emits_warning() -> None:
    defaults = load_solver_threshold_config()
    A = csr_matrix(np.eye(1, dtype=np.complex128))
    b = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    result = solve_linear_system(
        A,
        b,
        backend=_Backend((_Response(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(None),
    )

    assert math.isnan(result.cond_ind)
    assert len(result.warnings) == 1
    assert result.warnings[0].code == defaults.conditioning.unavailable_warning_code


def test_ill_conditioned_band_emits_warning_without_fail() -> None:
    defaults = load_solver_threshold_config()
    A = csr_matrix(np.eye(1, dtype=np.complex128))
    b = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    warn_value = (defaults.conditioning.warn_max + defaults.conditioning.fail_max) * 0.5
    assert warn_value > defaults.conditioning.fail_max
    assert warn_value <= defaults.conditioning.warn_max

    result = solve_linear_system(
        A,
        b,
        backend=_Backend((_Response(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(warn_value),
    )

    assert result.status == "pass"
    assert len(result.warnings) == 1
    assert result.warnings[0].code == defaults.conditioning.ill_conditioned_warning_code


def test_fail_condition_band_forces_numeric_failure_classification() -> None:
    defaults = load_solver_threshold_config()
    A = csr_matrix(np.eye(1, dtype=np.complex128))
    b = np.asarray([1.0 + 0.0j], dtype=np.complex128)

    result = solve_linear_system(
        A,
        b,
        backend=_Backend((_Response(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstEstimator(defaults.conditioning.fail_max),
    )

    assert result.status == "fail"
    assert result.failure_code == "E_NUM_SOLVE_FAILED"
    assert result.failure_reason == "condition_indicator_below_fail_threshold"
    assert result.warnings == ()

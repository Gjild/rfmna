from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.solver import (
    AttemptTraceRecord,
    BackendMetadata,
    BackendNotes,
    FallbackRunConfig,
    ResidualMetrics,
    SolveResult,
    load_solver_threshold_config,
)
from rfmna.solver.fallback import ConditionEstimator, run_fallback_ladder
from rfmna.sweep_engine import SweepLayout, run_sweep

pytestmark = pytest.mark.property

_THRESHOLDS = load_solver_threshold_config()
type _SweepStatus = Literal["pass", "degraded", "fail"]


@st.composite
def _status_sequences_with_fail(draw: st.DrawFn) -> tuple[_SweepStatus, ...]:
    length = draw(st.integers(min_value=1, max_value=8))
    fail_index = draw(st.integers(min_value=0, max_value=length - 1))
    statuses: list[_SweepStatus] = [
        cast(_SweepStatus, draw(st.sampled_from(("pass", "degraded")))) for _ in range(length)
    ]
    statuses[fail_index] = "fail"
    return tuple(statuses)


@st.composite
def _condition_indicator_inputs(draw: st.DrawFn) -> float | None:
    case = draw(st.sampled_from(("none", "nan", "fail", "warn", "pass")))
    if case == "none":
        return None
    if case == "nan":
        return float("nan")
    if case == "fail":
        scale = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        return float(_THRESHOLDS.conditioning.fail_max * scale)
    if case == "warn":
        span = _THRESHOLDS.conditioning.warn_max - _THRESHOLDS.conditioning.fail_max
        scale = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        return float(_THRESHOLDS.conditioning.fail_max + (span * scale))

    margin = draw(st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False))
    return float(
        _THRESHOLDS.conditioning.warn_max
        + (margin * max(1.0, abs(_THRESHOLDS.conditioning.warn_max)))
    )


def _backend_metadata() -> BackendMetadata:
    return BackendMetadata(
        backend_id="property_test_backend",
        backend_name="property_test_backend",
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
            input_format="csc",
            normalized_format="csc",
            n_unknowns=0,
            nnz=0,
            pivot_profile="default",
            scaling_enabled=False,
        ),
    )


def _solve_result_for_status(
    *,
    status: _SweepStatus,
    x: NDArray[np.complex128],
    point_index: int,
) -> SolveResult:
    return SolveResult(
        x=x,
        residual=ResidualMetrics(res_l2=0.1 + point_index, res_linf=0.2 + point_index, res_rel=0.0),
        status=status,
        backend=_backend_metadata(),
        failure_category=None if status != "fail" else "numeric",
        failure_code=None if status != "fail" else "E_NUM_SOLVE_FAILED",
        failure_reason=None if status != "fail" else "property_forced_failure",
        failure_message=None if status != "fail" else "property_forced_failure",
        cond_ind=0.5,
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
                res_l2=0.1 + point_index,
                res_linf=0.2 + point_index,
                res_rel=0.0,
                status=status,
                skip_reason=None,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ConstantConditionEstimator(ConditionEstimator):
    value: float | None

    def estimate(self, A: csc_matrix) -> float | None:
        del A
        return self.value


@given(statuses=_status_sequences_with_fail())
def test_sweep_fail_sentinel_and_point_presence_invariant(
    statuses: tuple[_SweepStatus, ...],
) -> None:
    layout = SweepLayout(n_nodes=2, n_aux=1)
    frequencies = np.asarray([float(index + 1) for index in range(len(statuses))], dtype=np.float64)

    def assemble_point(
        point_index: int,
        frequency_hz: float,
    ) -> tuple[csc_matrix, NDArray[np.complex128]]:
        del point_index, frequency_hz
        return csc_matrix(np.eye(3, dtype=np.complex128)), np.zeros(3, dtype=np.complex128)

    def solve_point(
        A: csc_matrix,
        b: NDArray[np.complex128],
    ) -> SolveResult:
        del A, b
        point_index = solve_point.calls
        status = statuses[point_index]
        solve_point.calls += 1
        base_value = np.complex128(complex(point_index + 1, point_index + 2))
        return _solve_result_for_status(
            status=status,
            x=np.asarray([base_value, base_value + 1.0, base_value + 2.0], dtype=np.complex128),
            point_index=point_index,
        )

    solve_point.calls = 0  # type: ignore[attr-defined]
    result = run_sweep(frequencies, layout, assemble_point, solve_point=solve_point)

    assert result.n_points == len(statuses)
    assert result.status.tolist() == list(statuses)

    for point_index, status in enumerate(statuses):
        if status == "fail":
            assert np.isnan(result.V_nodes[point_index].real).all()
            assert np.isnan(result.V_nodes[point_index].imag).all()
            assert np.isnan(result.I_aux[point_index].real).all()
            assert np.isnan(result.I_aux[point_index].imag).all()
            assert math.isnan(float(result.res_l2[point_index]))
            assert math.isnan(float(result.res_linf[point_index]))
            assert math.isnan(float(result.res_rel[point_index]))
            assert math.isnan(float(result.cond_ind[point_index]))
            assert result.diagnostics_by_point[point_index]
            continue

        assert np.isfinite(result.V_nodes[point_index].real).all()
        assert np.isfinite(result.V_nodes[point_index].imag).all()
        assert np.isfinite(result.I_aux[point_index].real).all()
        assert np.isfinite(result.I_aux[point_index].imag).all()
        assert np.isfinite(result.res_l2[point_index])
        assert np.isfinite(result.res_linf[point_index])
        assert np.isfinite(result.res_rel[point_index])
        assert np.isfinite(result.cond_ind[point_index])


@given(condition_indicator=_condition_indicator_inputs())
def test_condition_indicator_band_classification_property(
    condition_indicator: float | None,
) -> None:
    matrix = csc_matrix(np.eye(2, dtype=np.complex128))
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    result = run_fallback_ladder(
        matrix,
        rhs,
        node_voltage_count=2,
        run_config=FallbackRunConfig(
            enable_alt_pivot=False, enable_scaling=False, enable_gmin=False
        ),
        condition_estimator=_ConstantConditionEstimator(condition_indicator),
    )

    unavailable_warning = _THRESHOLDS.conditioning.unavailable_warning_code
    ill_warning = _THRESHOLDS.conditioning.ill_conditioned_warning_code
    warning_codes = tuple(warning.code for warning in result.warnings)

    if condition_indicator is None or not np.isfinite(condition_indicator):
        assert result.status == "pass"
        assert np.isnan(result.cond_ind)
        assert warning_codes == (unavailable_warning,)
        return

    assert result.cond_ind == condition_indicator
    if condition_indicator <= _THRESHOLDS.conditioning.fail_max:
        assert result.status == "fail"
        assert result.failure_reason == "condition_indicator_below_fail_threshold"
        assert warning_codes == ()
        return

    if condition_indicator <= _THRESHOLDS.conditioning.warn_max:
        assert result.status == "pass"
        assert warning_codes == (ill_warning,)
        return

    assert result.status == "pass"
    assert warning_codes == ()

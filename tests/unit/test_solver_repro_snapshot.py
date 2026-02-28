from __future__ import annotations

import pytest

from rfmna.solver import AttemptTraceRecord, FallbackRunConfig
from rfmna.solver.repro_snapshot import build_solver_config_snapshot

pytestmark = pytest.mark.unit
_EXPECTED_TOTAL_SOLVE_CALLS = 2
_EXPECTED_TOTAL_ATTEMPT_RECORDS = 10
_EXPECTED_MAX_ATTEMPTS_PER_CALL = 5
_EXPECTED_CALLS_WITH_RETRY = 2


def _trace_row(
    *,
    attempt_index: int,
    stage: str,
    stage_state: str,
    success: bool,
    skip_reason: str | None = None,
) -> AttemptTraceRecord:
    return AttemptTraceRecord(
        attempt_index=attempt_index,
        stage=stage,  # type: ignore[arg-type]
        stage_state=stage_state,  # type: ignore[arg-type]
        scaling_enabled=False,
        pivot_profile="default",
        gmin_value=None,
        success=success,
        backend_failure_category=None,
        backend_failure_reason=None,
        backend_failure_code=None,
        backend_failure_message=None,
        res_l2=0.0,
        res_linf=0.0,
        res_rel=0.0,
        status="pass" if success else None,
        skip_reason=skip_reason,
    )


def test_snapshot_defaults_emit_required_keys_with_zeroed_attempt_summary() -> None:
    snapshot = build_solver_config_snapshot()
    assert snapshot["schema"] == "solver_repro_snapshot_v1"

    retry_controls = snapshot["retry_controls"]
    assert isinstance(retry_controls, dict)
    assert retry_controls["enable_alt_pivot"] is True
    assert retry_controls["enable_scaling"] is True
    assert retry_controls["enable_gmin"] is True
    assert retry_controls["fallback_order"] == [
        "baseline",
        "alt_pivot",
        "scaling",
        "gmin",
        "final_fail",
    ]

    conversion_controls = snapshot["conversion_math_controls"]
    assert isinstance(conversion_controls, dict)
    assert conversion_controls["enable_gmin_regularization"] is False

    summary = snapshot["attempt_trace_summary"]
    assert isinstance(summary, dict)
    assert summary["total_solve_calls"] == 0
    assert summary["total_attempt_records"] == 0
    assert summary["max_attempts_per_call"] == 0
    assert summary["calls_with_retry"] == 0
    assert summary["calls_with_fail"] == 0
    assert summary["calls_with_stage_skips"] == 0
    assert summary["skip_reason_counts"] == {}


def test_snapshot_attempt_trace_summary_is_deterministic_and_stage_complete() -> None:
    attempt_traces = (
        (
            _trace_row(
                attempt_index=0,
                stage="baseline",
                stage_state="run",
                success=False,
            ),
            _trace_row(
                attempt_index=1,
                stage="alt_pivot",
                stage_state="run",
                success=True,
            ),
            _trace_row(
                attempt_index=2,
                stage="scaling",
                stage_state="skipped",
                success=False,
                skip_reason="prior_stage_succeeded",
            ),
            _trace_row(
                attempt_index=3,
                stage="gmin",
                stage_state="skipped",
                success=False,
                skip_reason="prior_stage_succeeded",
            ),
            _trace_row(
                attempt_index=4,
                stage="final_fail",
                stage_state="skipped",
                success=False,
                skip_reason="prior_stage_succeeded",
            ),
        ),
        (
            _trace_row(
                attempt_index=0,
                stage="baseline",
                stage_state="run",
                success=False,
            ),
            _trace_row(
                attempt_index=1,
                stage="alt_pivot",
                stage_state="run",
                success=False,
            ),
            _trace_row(
                attempt_index=2,
                stage="scaling",
                stage_state="run",
                success=False,
            ),
            _trace_row(
                attempt_index=3,
                stage="gmin",
                stage_state="run",
                success=False,
            ),
            _trace_row(
                attempt_index=4,
                stage="final_fail",
                stage_state="run",
                success=False,
            ),
        ),
    )

    snapshot = build_solver_config_snapshot(attempt_traces=attempt_traces)
    summary = snapshot["attempt_trace_summary"]
    assert isinstance(summary, dict)
    assert summary["total_solve_calls"] == _EXPECTED_TOTAL_SOLVE_CALLS
    assert summary["total_attempt_records"] == _EXPECTED_TOTAL_ATTEMPT_RECORDS
    assert summary["max_attempts_per_call"] == _EXPECTED_MAX_ATTEMPTS_PER_CALL
    assert summary["calls_with_retry"] == _EXPECTED_CALLS_WITH_RETRY
    assert summary["calls_with_fail"] == 1
    assert summary["calls_with_stage_skips"] == 1
    assert summary["stage_run_counts"] == {
        "baseline": 2,
        "alt_pivot": 2,
        "scaling": 1,
        "gmin": 1,
        "final_fail": 1,
    }
    assert summary["stage_skip_counts"] == {
        "baseline": 0,
        "alt_pivot": 0,
        "scaling": 1,
        "gmin": 1,
        "final_fail": 1,
    }
    assert summary["skip_reason_counts"] == {"prior_stage_succeeded": 3}


def test_snapshot_serializes_inactive_controls_explicitly() -> None:
    snapshot = build_solver_config_snapshot(
        run_config=FallbackRunConfig(
            enable_alt_pivot=False,
            enable_scaling=False,
            enable_gmin=False,
        )
    )
    retry_controls = snapshot["retry_controls"]
    assert isinstance(retry_controls, dict)
    assert retry_controls["enable_alt_pivot"] is False
    assert retry_controls["enable_scaling"] is False
    assert retry_controls["enable_gmin"] is False

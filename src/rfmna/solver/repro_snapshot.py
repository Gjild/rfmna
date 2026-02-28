from __future__ import annotations

from collections.abc import Mapping, Sequence

from .fallback import (
    AttemptTraceRecord,
    FallbackRunConfig,
    SolverThresholdConfig,
    load_solver_threshold_config,
)

_SCHEMA_ID = "solver_repro_snapshot_v1"
_STAGE_ORDER: tuple[str, ...] = ("baseline", "alt_pivot", "scaling", "gmin", "final_fail")


def build_solver_config_snapshot(
    *,
    run_config: FallbackRunConfig | None = None,
    thresholds: SolverThresholdConfig | None = None,
    attempt_traces: Sequence[Sequence[AttemptTraceRecord]] = (),
) -> Mapping[str, object]:
    config = run_config if run_config is not None else FallbackRunConfig()
    threshold_config = (
        thresholds
        if thresholds is not None
        else load_solver_threshold_config(config.thresholds_artifact_path)
    )
    gmin_values = (
        config.gmin_values_override
        if config.gmin_values_override is not None
        else threshold_config.gmin_values
    )

    stage_run_counts = {stage: 0 for stage in _STAGE_ORDER}
    stage_skip_counts = {stage: 0 for stage in _STAGE_ORDER}
    skip_reason_counts: dict[str, int] = {}
    total_attempt_records = 0
    max_attempts_per_call = 0
    calls_with_retry = 0
    calls_with_fail = 0
    calls_with_stage_skips = 0

    for trace in attempt_traces:
        total_attempt_records += len(trace)
        max_attempts_per_call = max(max_attempts_per_call, len(trace))
        call_has_retry = False
        call_has_skip = False
        call_failed = False

        for row in trace:
            if row.stage_state == "run":
                stage_run_counts[row.stage] = stage_run_counts[row.stage] + 1
                if row.stage not in {"baseline", "final_fail"}:
                    call_has_retry = True
                if row.stage == "final_fail":
                    call_failed = True
            else:
                stage_skip_counts[row.stage] = stage_skip_counts[row.stage] + 1
                call_has_skip = True

            if row.skip_reason is not None:
                skip_reason_counts[row.skip_reason] = skip_reason_counts.get(row.skip_reason, 0) + 1

        if call_has_retry:
            calls_with_retry += 1
        if call_failed:
            calls_with_fail += 1
        if call_has_skip:
            calls_with_stage_skips += 1

    return {
        "schema": _SCHEMA_ID,
        "retry_controls": {
            "enable_alt_pivot": config.enable_alt_pivot,
            "enable_scaling": config.enable_scaling,
            "enable_gmin": config.enable_gmin,
            "fallback_order": list(threshold_config.fallback_order),
            "gmin_values": [float(value) for value in gmin_values],
            "scaling_enabled_by_default": threshold_config.scaling_enabled_by_default,
            "thresholds_source": (
                "default_artifact"
                if config.thresholds_artifact_path is None
                else "override_artifact"
            ),
        },
        "conversion_math_controls": {
            "enable_gmin_regularization": False,
        },
        "attempt_trace_summary": {
            "total_solve_calls": len(attempt_traces),
            "total_attempt_records": total_attempt_records,
            "max_attempts_per_call": max_attempts_per_call,
            "calls_with_retry": calls_with_retry,
            "calls_with_fail": calls_with_fail,
            "calls_with_stage_skips": calls_with_stage_skips,
            "stage_run_counts": stage_run_counts,
            "stage_skip_counts": stage_skip_counts,
            "skip_reason_counts": {
                key: skip_reason_counts[key] for key in sorted(skip_reason_counts)
            },
        },
    }

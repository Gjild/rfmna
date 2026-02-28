from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast, runtime_checkable

import numpy as np
import typer
from numpy.typing import NDArray

from rfmna.diagnostics import (
    DiagnosticEvent,
    Severity,
    SolverStage,
    build_diagnostic_event,
    sort_diagnostics,
)
from rfmna.parser import PreflightInput, preflight_check
from rfmna.rf_metrics import PortBoundary, SParameterResult, YParameterResult, ZParameterResult
from rfmna.solver import (
    AttemptTraceRecord,
    FallbackRunConfig,
    SolveResult,
)
from rfmna.solver.backend import SparseComplexMatrix
from rfmna.solver.repro_snapshot import build_solver_config_snapshot
from rfmna.solver.solve import solve_linear_system
from rfmna.sweep_engine import (
    AssemblePointFn,
    RFMetricName,
    SweepLayout,
    SweepResult,
    SweepRFRequest,
    SweepRFScalarResult,
    run_sweep,
    sweep_diagnostic_sort_key,
)
from rfmna.viz_io import attach_manifest_to_run_payload, build_manifest

app = typer.Typer(help="RF MNA solver CLI")

_RF_METRIC_ORDER: dict[RFMetricName, int] = {
    "y": 0,
    "z": 1,
    "s": 2,
    "zin": 3,
    "zout": 4,
}
_RF_METRIC_CHOICES: tuple[str, ...] = tuple(_RF_METRIC_ORDER)
_CLI_RF_METRIC_INVALID = "E_CLI_RF_METRIC_INVALID"
_CLI_RF_OPTIONS_INVALID = "E_CLI_RF_OPTIONS_INVALID"
_CLI_CHECK_LOADER_FAILED = "E_CLI_CHECK_LOADER_FAILED"
_CLI_CHECK_INTERNAL = "E_CLI_CHECK_INTERNAL"
_CHECK_OUTPUT_SCHEMA_ID: Final[str] = "docs/spec/schemas/check_output_v1.json"
_CHECK_OUTPUT_SCHEMA_VERSION: Final[int] = 1
_RF_OPTION = typer.Option(
    None,
    "--rf",
    help="Repeatable RF metric selector: y|z|s|zin|zout",
)
_SOLVER_CONFIG_SNAPSHOT_STATE: dict[str, dict[str, object] | None] = {"value": None}


@dataclass(frozen=True, slots=True)
class CliDesignBundle:
    preflight_input: PreflightInput
    frequencies_hz: Sequence[float] | NDArray[np.float64]
    sweep_layout: SweepLayout
    assemble_point: AssemblePointFn
    rf_ports: tuple[PortBoundary, ...] = ()


@runtime_checkable
class DesignLoader(Protocol):
    def __call__(self, design: str) -> CliDesignBundle: ...


@runtime_checkable
class PreflightRunner(Protocol):
    def __call__(self, input_data: PreflightInput) -> tuple[DiagnosticEvent, ...]: ...


@runtime_checkable
class SweepRunner(Protocol):
    def __call__(
        self,
        frequencies_hz: Sequence[float] | NDArray[np.float64],
        sweep_layout: SweepLayout,
        assemble_point: AssemblePointFn,
    ) -> SweepResult: ...


def _load_design_bundle(design: str) -> CliDesignBundle:
    raise typer.BadParameter(
        f"Design loader is not configured for '{design}'. Provide a valid design integration."
    )


def _execute_preflight(input_data: PreflightInput) -> tuple[DiagnosticEvent, ...]:
    return preflight_check(input_data)


def _execute_sweep(
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    sweep_layout: SweepLayout,
    assemble_point: AssemblePointFn,
) -> SweepResult:
    return _execute_sweep_with_repro_capture(
        frequencies_hz,
        sweep_layout,
        assemble_point,
        rf_request=None,
    )


def _execute_sweep_with_rf(
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    sweep_layout: SweepLayout,
    assemble_point: AssemblePointFn,
    *,
    rf_request: SweepRFRequest | None,
) -> SweepResult:
    if rf_request is None:
        return _execute_sweep(frequencies_hz, sweep_layout, assemble_point)
    return _execute_sweep_with_repro_capture(
        frequencies_hz,
        sweep_layout,
        assemble_point,
        rf_request=rf_request,
    )


def _execute_sweep_with_repro_capture(
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    sweep_layout: SweepLayout,
    assemble_point: AssemblePointFn,
    *,
    rf_request: SweepRFRequest | None,
) -> SweepResult:
    traces: list[tuple[AttemptTraceRecord, ...]] = []
    conversion_run_config = FallbackRunConfig(enable_gmin=False)

    def mna_solver(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        result = solve_linear_system(A, b, node_voltage_count=sweep_layout.n_nodes)
        traces.append(result.attempt_trace)
        return result

    def conversion_solver(A: SparseComplexMatrix, b: NDArray[np.complex128]) -> SolveResult:
        result = solve_linear_system(A, b, run_config=conversion_run_config)
        traces.append(result.attempt_trace)
        return result

    result = run_sweep(
        frequencies_hz,
        sweep_layout,
        assemble_point,
        solve_point=mna_solver,
        conversion_solve_point=conversion_solver,
        rf_request=rf_request,
    )
    _set_last_solver_config_snapshot(build_solver_config_snapshot(attempt_traces=traces))
    return result


@app.command()
def check(
    design: str,
    format: Literal["text", "json"] = typer.Option(
        "text",
        "--format",
        help="Check output format: text|json",
        show_default=True,
    ),
) -> None:
    """Preflight checks for a design file."""
    diagnostics: tuple[DiagnosticEvent, ...]
    try:
        bundle = _load_design_bundle(design)
    except typer.BadParameter as exc:
        diagnostics = (_check_loader_failure_diagnostic(design=design, exc=exc),)
    except Exception as exc:  # pragma: no cover - defensive boundary path
        diagnostics = (_check_internal_failure_diagnostic(design=design, exc=exc),)
    else:
        try:
            diagnostics = tuple(sort_diagnostics(_execute_preflight(bundle.preflight_input)))
        except Exception as exc:  # pragma: no cover - defensive boundary path
            diagnostics = (_check_internal_failure_diagnostic(design=design, exc=exc),)

    _emit_check_output(design=design, diagnostics=diagnostics, output_format=format)
    raise typer.Exit(code=_derive_check_exit_code(diagnostics))


@app.command()
def run(
    design: str,
    analysis: str = "ac",
    rf: list[str] | None = _RF_OPTION,
) -> None:
    """Run simulation."""
    if analysis != "ac":
        raise typer.BadParameter("Only AC analysis is supported in v4.")

    try:
        _reset_last_solver_config_snapshot()
        bundle = _load_design_bundle(design)
        preflight_diagnostics = tuple(sort_diagnostics(_execute_preflight(bundle.preflight_input)))
        _print_preflight_diagnostics(preflight_diagnostics)
        if _has_error_diagnostics(preflight_diagnostics):
            typer.echo("run blocked: preflight contains error diagnostics")
            raise typer.Exit(code=2)

        rf_request, rf_option_diagnostics = _resolve_rf_request(
            bundle=bundle,
            raw_metrics=tuple(rf or ()),
        )
        _print_preflight_diagnostics(rf_option_diagnostics)
        if _has_error_diagnostics(rf_option_diagnostics):
            raise typer.Exit(code=2)

        sweep_result = _execute_sweep_with_rf(
            bundle.frequencies_hz,
            bundle.sweep_layout,
            bundle.assemble_point,
            rf_request=rf_request,
        )
    except typer.BadParameter:
        raise
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary path
        typer.echo(f"internal error: {exc}")
        raise typer.Exit(code=2) from exc

    frequencies = np.asarray(bundle.frequencies_hz, dtype=np.float64)
    manifest = build_manifest(
        input_payload={
            "design": design,
            "analysis": analysis,
            "frequencies_hz": [float(value) for value in frequencies.tolist()],
        },
        resolved_params_payload={},
        solver_config_snapshot={
            **_consume_last_solver_config_snapshot(),
            "analysis": analysis,
        },
        frequency_grid_metadata={"n_points": sweep_result.n_points},
    )
    run_payload = attach_manifest_to_run_payload(sweep_result, manifest)
    _print_run_summary(run_payload.run_payload, frequencies)
    _print_sweep_diagnostics(run_payload.run_payload)
    _print_rf_lines(run_payload.run_payload)
    raise typer.Exit(code=_derive_run_exit_code(run_payload.run_payload.status))


def _reset_last_solver_config_snapshot() -> None:
    _SOLVER_CONFIG_SNAPSHOT_STATE["value"] = None


def _set_last_solver_config_snapshot(snapshot: Mapping[str, object]) -> None:
    _SOLVER_CONFIG_SNAPSHOT_STATE["value"] = dict(snapshot)


def _consume_last_solver_config_snapshot() -> dict[str, object]:
    snapshot = _SOLVER_CONFIG_SNAPSHOT_STATE["value"]
    if snapshot is None:
        return dict(build_solver_config_snapshot())
    return dict(snapshot)


def _resolve_rf_request(
    *,
    bundle: CliDesignBundle,
    raw_metrics: Sequence[str],
) -> tuple[SweepRFRequest | None, tuple[DiagnosticEvent, ...]]:
    if not raw_metrics:
        return (None, ())

    canonical_metrics, parse_diagnostics = _canonical_rf_metrics(raw_metrics)
    if parse_diagnostics:
        return (None, parse_diagnostics)

    if not bundle.rf_ports:
        return (
            None,
            (
                _cli_option_error(
                    code=_CLI_RF_OPTIONS_INVALID,
                    message="--rf requires design loader RF port boundaries",
                    suggested_action="provide deterministic rf_ports in the design loader bundle",
                    witness={
                        "issue": "rf_ports_missing",
                        "metrics": list(canonical_metrics),
                    },
                ),
            ),
        )

    try:
        request = SweepRFRequest(
            ports=bundle.rf_ports,
            metrics=canonical_metrics,
        )
    except ValueError as exc:
        return (
            None,
            (
                _cli_option_error(
                    code=_CLI_RF_OPTIONS_INVALID,
                    message=str(exc),
                    suggested_action="provide compatible --rf options and canonical RF port mapping",
                    witness={
                        "issue": "rf_request_invalid",
                        "metrics": list(canonical_metrics),
                        "n_rf_ports": len(bundle.rf_ports),
                    },
                ),
            ),
        )
    return (request, ())


def _canonical_rf_metrics(
    raw_metrics: Sequence[str],
) -> tuple[tuple[RFMetricName, ...], tuple[DiagnosticEvent, ...]]:
    unique_valid: dict[RFMetricName, None] = {}
    invalid_values: set[str] = set()
    for metric in raw_metrics:
        normalized = metric.strip().lower()
        if normalized in _RF_METRIC_ORDER:
            unique_valid[cast(RFMetricName, normalized)] = None
            continue
        invalid_values.add(metric)

    if invalid_values:
        diagnostics = [
            _cli_option_error(
                code=_CLI_RF_METRIC_INVALID,
                message=f"unsupported --rf value '{invalid_value}'",
                suggested_action=f"choose --rf from: {','.join(_RF_METRIC_CHOICES)}",
                witness={
                    "allowed_values": list(_RF_METRIC_CHOICES),
                    "option": "--rf",
                    "provided_value": invalid_value,
                },
            )
            for invalid_value in sorted(invalid_values)
        ]
        return ((), tuple(sort_diagnostics(diagnostics)))

    ordered = tuple(sorted(unique_valid, key=lambda metric_name: _RF_METRIC_ORDER[metric_name]))
    return (ordered, ())


def _cli_option_error(
    *,
    code: str,
    message: str,
    suggested_action: str,
    witness: object,
) -> DiagnosticEvent:
    return build_diagnostic_event(
        code=code,
        message=message,
        suggested_action=suggested_action,
        solver_stage=SolverStage.PARSE,
        element_id="cli",
        witness=witness,
    )


def _has_error_diagnostics(diagnostics: Sequence[DiagnosticEvent]) -> bool:
    return any(_severity_is_error(event.severity) for event in diagnostics)


def _severity_is_error(severity: object) -> bool:
    if isinstance(severity, Severity):
        return severity is Severity.ERROR
    if isinstance(severity, str):
        return severity.lower() == "error"
    return False


def _derive_run_exit_code(statuses: NDArray[np.str_]) -> int:
    normalized = [str(value) for value in statuses.tolist()]
    if any(value == "fail" for value in normalized):
        return 2
    if any(value == "degraded" for value in normalized):
        return 1
    return 0


def _derive_check_exit_code(diagnostics: Sequence[DiagnosticEvent]) -> int:
    if _has_error_diagnostics(diagnostics):
        return 2
    return 0


def _emit_check_output(
    *,
    design: str,
    diagnostics: Sequence[DiagnosticEvent],
    output_format: Literal["text", "json"],
) -> None:
    if output_format == "json":
        typer.echo(_build_check_json_output(design=design, diagnostics=diagnostics))
        return
    _print_preflight_diagnostics(diagnostics)


def _build_check_json_output(*, design: str, diagnostics: Sequence[DiagnosticEvent]) -> str:
    payload: dict[str, object] = {
        "schema": _CHECK_OUTPUT_SCHEMA_ID,
        "schema_version": _CHECK_OUTPUT_SCHEMA_VERSION,
        "design": design,
        "status": "fail" if _has_error_diagnostics(diagnostics) else "pass",
        "exit_code": _derive_check_exit_code(diagnostics),
        "diagnostics": [event.model_dump(mode="json", exclude_none=True) for event in diagnostics],
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _check_loader_failure_diagnostic(*, design: str, exc: Exception) -> DiagnosticEvent:
    return build_diagnostic_event(
        code=_CLI_CHECK_LOADER_FAILED,
        message=f"check loader boundary failed: {_exception_message(exc)}",
        element_id="cli.check",
        witness={
            "design": design,
            "error_type": type(exc).__name__,
            "source": "design_loader",
        },
    )


def _check_internal_failure_diagnostic(*, design: str, exc: Exception) -> DiagnosticEvent:
    return build_diagnostic_event(
        code=_CLI_CHECK_INTERNAL,
        message=f"check internal failure: {_exception_message(exc)}",
        element_id="cli.check",
        witness={
            "design": design,
            "error_type": type(exc).__name__,
            "source": "check_command",
        },
    )


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return type(exc).__name__


def _print_preflight_diagnostics(diagnostics: Sequence[DiagnosticEvent]) -> None:
    for event in diagnostics:
        typer.echo(
            "DIAG"
            f" severity={event.severity}"
            f" stage={event.solver_stage}"
            f" code={event.code}"
            f" message={event.message}"
        )


def _print_run_summary(result: SweepResult, frequencies_hz: NDArray[np.float64]) -> None:
    if result.n_points != int(frequencies_hz.shape[0]):
        raise ValueError("sweep result point count does not match frequency vector length")
    for point_index in range(result.n_points):
        typer.echo(
            "POINT"
            f" index={point_index}"
            f" freq_hz={float(frequencies_hz[point_index]):.12g}"
            f" status={result.status[point_index]}"
        )


def _print_sweep_diagnostics(result: SweepResult) -> None:
    for point_index, point_diags in enumerate(result.diagnostics_by_point):
        ordered = tuple(sorted(point_diags, key=sweep_diagnostic_sort_key))
        for diag in ordered:
            typer.echo(
                "DIAG"
                f" point_index={point_index}"
                f" frequency_hz={diag.frequency_hz:.12g}"
                f" frequency_index={diag.frequency_index}"
                f" element_id={diag.element_id}"
                f" severity={diag.severity}"
                f" stage={diag.solver_stage}"
                f" code={diag.code}"
                f" message={diag.message}"
            )


def _print_rf_lines(result: SweepResult) -> None:
    payloads = result.rf_payloads
    if payloads is None:
        return

    for metric_name in payloads.metric_names:
        payload = payloads.get(metric_name)
        if payload is None:
            continue
        if isinstance(payload, SweepRFScalarResult):
            _print_rf_scalar_lines(payload=payload)
            continue
        if isinstance(payload, YParameterResult):
            _print_rf_matrix_lines(
                metric_name=metric_name,
                frequencies_hz=np.asarray(payload.frequencies_hz, dtype=np.float64),
                port_ids=payload.port_ids,
                status=np.asarray(payload.status, dtype=np.str_),
                values=np.asarray(payload.y, dtype=np.complex128),
            )
            continue
        if isinstance(payload, ZParameterResult):
            _print_rf_matrix_lines(
                metric_name=metric_name,
                frequencies_hz=np.asarray(payload.frequencies_hz, dtype=np.float64),
                port_ids=payload.port_ids,
                status=np.asarray(payload.status, dtype=np.str_),
                values=np.asarray(payload.z, dtype=np.complex128),
            )
            continue
        if isinstance(payload, SParameterResult):
            _print_rf_matrix_lines(
                metric_name=metric_name,
                frequencies_hz=np.asarray(payload.frequencies_hz, dtype=np.float64),
                port_ids=payload.port_ids,
                status=np.asarray(payload.status, dtype=np.str_),
                values=np.asarray(payload.s, dtype=np.complex128),
            )
            continue
        raise TypeError(f"unsupported RF payload type for metric '{metric_name}'")


def _print_rf_scalar_lines(payload: SweepRFScalarResult) -> None:
    frequencies_hz = np.asarray(payload.frequencies_hz, dtype=np.float64)
    statuses = np.asarray(payload.status, dtype=np.str_)
    values = np.asarray(payload.values, dtype=np.complex128)
    for point_index in range(int(frequencies_hz.shape[0])):
        value = complex(values[point_index])
        typer.echo(
            "RF"
            f" metric={payload.metric_name}"
            f" point_index={point_index}"
            f" frequency_hz={_format_float(frequencies_hz[point_index])}"
            f" status={statuses[point_index]}"
            f" port_id={payload.port_id}"
            f" value_re={_format_float(value.real)}"
            f" value_im={_format_float(value.imag)}"
        )


def _print_rf_matrix_lines(
    *,
    metric_name: RFMetricName,
    frequencies_hz: NDArray[np.float64],
    port_ids: tuple[str, ...],
    status: NDArray[np.str_],
    values: NDArray[np.complex128],
) -> None:
    n_points = int(frequencies_hz.shape[0])
    for point_index in range(n_points):
        for row_index, row_port_id in enumerate(port_ids):
            for col_index, col_port_id in enumerate(port_ids):
                value = complex(values[point_index, row_index, col_index])
                typer.echo(
                    "RF"
                    f" metric={metric_name}"
                    f" point_index={point_index}"
                    f" frequency_hz={_format_float(frequencies_hz[point_index])}"
                    f" status={status[point_index]}"
                    f" row_index={row_index}"
                    f" row_port={row_port_id}"
                    f" col_index={col_index}"
                    f" col_port={col_port_id}"
                    f" value_re={_format_float(value.real)}"
                    f" value_im={_format_float(value.imag)}"
                )


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def main() -> None:
    app()

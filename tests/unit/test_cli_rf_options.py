from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.parser import PreflightInput
from rfmna.rf_metrics import PortBoundary, SParameterResult, YParameterResult, ZParameterResult
from rfmna.sweep_engine import (
    RFMetricName,
    SweepDiagnostic,
    SweepLayout,
    SweepResult,
    SweepRFPayloads,
    SweepRFRequest,
    SweepRFScalarResult,
)

pytestmark = pytest.mark.unit

runner = CliRunner()
_POINT_COUNT = 2
_EXIT_FAIL = 2


def _build_bundle(
    *,
    frequencies: Sequence[float] = (1.0, 2.0),
    with_rf_ports: bool = True,
) -> cli_main.CliDesignBundle:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, np.ndarray]:
        del index, frequency_hz
        raise AssertionError("assemble_point should not be called in CLI rf option tests")

    rf_ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    )

    return cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=tuple(frequencies),
        sweep_layout=SweepLayout(n_nodes=2, n_aux=0),
        assemble_point=assemble_point,
        rf_ports=rf_ports if with_rf_ports else (),
    )


def _sweep_result(
    *,
    frequencies: Sequence[float],
    statuses: Sequence[str],
    diagnostics_by_point: tuple[tuple[SweepDiagnostic, ...], ...] | None = None,
    rf_payloads: SweepRFPayloads | None = None,
) -> SweepResult:
    n_points = len(frequencies)
    if diagnostics_by_point is None:
        diagnostics_by_point = tuple(() for _ in range(n_points))
    return SweepResult(
        n_points=n_points,
        n_nodes=2,
        n_aux=0,
        V_nodes=np.zeros((n_points, 2), dtype=np.complex128),
        I_aux=np.zeros((n_points, 0), dtype=np.complex128),
        res_l2=np.zeros(n_points, dtype=np.float64),
        res_linf=np.zeros(n_points, dtype=np.float64),
        res_rel=np.zeros(n_points, dtype=np.float64),
        cond_ind=np.ones(n_points, dtype=np.float64),
        status=np.asarray(tuple(statuses), dtype=np.dtype("<U8")),
        diagnostics_by_point=diagnostics_by_point,
        rf_payloads=rf_payloads,
    )


def _rf_payloads(
    metrics: Sequence[RFMetricName],
    *,
    frequencies: Sequence[float] = (1.0, 2.0),
) -> SweepRFPayloads:
    freq = np.asarray(tuple(frequencies), dtype=np.float64)
    n_points = int(freq.shape[0])
    port_ids = ("P1", "P2")
    status = np.asarray(("pass",) * n_points, dtype=np.dtype("<U8"))
    diagnostics = tuple(() for _ in range(n_points))

    y_values = np.asarray(
        [
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            [[5.0 + 0.0j, 6.0 + 0.0j], [7.0 + 0.0j, 8.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    z_values = np.asarray(
        [
            [[11.0 + 0.0j, 12.0 + 0.0j], [13.0 + 0.0j, 14.0 + 0.0j]],
            [[15.0 + 0.0j, 16.0 + 0.0j], [17.0 + 0.0j, 18.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    s_values = np.asarray(
        [
            [[0.1 + 0.01j, 0.2 + 0.02j], [0.3 + 0.03j, 0.4 + 0.04j]],
            [[0.5 + 0.05j, 0.6 + 0.06j], [0.7 + 0.07j, 0.8 + 0.08j]],
        ],
        dtype=np.complex128,
    )

    entries: list[
        tuple[
            RFMetricName,
            YParameterResult | ZParameterResult | SParameterResult | SweepRFScalarResult,
        ]
    ] = []
    for metric in metrics:
        if metric == "y":
            entries.append(
                (
                    "y",
                    YParameterResult(
                        frequencies_hz=freq,
                        port_ids=port_ids,
                        y=y_values,
                        status=status,
                        diagnostics_by_point=diagnostics,
                    ),
                )
            )
            continue
        if metric == "z":
            entries.append(
                (
                    "z",
                    ZParameterResult(
                        frequencies_hz=freq,
                        port_ids=port_ids,
                        z=z_values,
                        status=status,
                        diagnostics_by_point=diagnostics,
                        extraction_mode="direct",
                    ),
                )
            )
            continue
        if metric == "s":
            entries.append(
                (
                    "s",
                    SParameterResult(
                        frequencies_hz=freq,
                        port_ids=port_ids,
                        s=s_values,
                        status=status,
                        diagnostics_by_point=diagnostics,
                        conversion_source="from_z",
                    ),
                )
            )
            continue
        if metric == "zin":
            entries.append(
                (
                    "zin",
                    SweepRFScalarResult(
                        frequencies_hz=freq,
                        port_ids=port_ids,
                        port_id="P1",
                        metric_name="zin",
                        values=np.asarray((51.0 + 1.0j, 52.0 + 2.0j), dtype=np.complex128),
                        status=status,
                        diagnostics_by_point=diagnostics,
                    ),
                )
            )
            continue
        if metric == "zout":
            entries.append(
                (
                    "zout",
                    SweepRFScalarResult(
                        frequencies_hz=freq,
                        port_ids=port_ids,
                        port_id="P2",
                        metric_name="zout",
                        values=np.asarray((61.0 + 1.0j, 62.0 + 2.0j), dtype=np.complex128),
                        status=status,
                        diagnostics_by_point=diagnostics,
                    ),
                )
            )
            continue
        raise AssertionError(f"unsupported metric in test payload setup: {metric}")

    return SweepRFPayloads(by_metric=tuple(entries))


def _rf_metric_block_order(stdout: str) -> tuple[str, ...]:
    lines = [line for line in stdout.splitlines() if line.startswith("RF metric=")]
    block_order: list[str] = []
    for line in lines:
        metric_token = next(token for token in line.split() if token.startswith("metric="))
        metric_name = metric_token.split("=", 1)[1]
        if not block_order or block_order[-1] != metric_name:
            block_order.append(metric_name)
    return tuple(block_order)


@pytest.mark.parametrize("metric", ["y", "z", "s", "zin", "zout"])
def test_run_accepts_each_rf_metric_and_emits_additive_rf_lines(
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    bundle = _build_bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    def _sweep(
        frequencies_hz: Sequence[float],
        sweep_layout: SweepLayout,
        assemble_point: object,
        *,
        rf_request: object,
    ) -> SweepResult:
        del sweep_layout, assemble_point
        assert rf_request is not None
        assert tuple(frequencies_hz) == (1.0, 2.0)
        return _sweep_result(
            frequencies=frequencies_hz,
            statuses=("pass", "pass"),
            rf_payloads=_rf_payloads((metric,), frequencies=frequencies_hz),
        )

    monkeypatch.setattr(cli_main, "_execute_sweep_with_rf", _sweep)

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac", "--rf", metric])
    assert result.exit_code == 0

    point_lines = [line for line in result.stdout.splitlines() if line.startswith("POINT index=")]
    assert len(point_lines) == _POINT_COUNT

    rf_lines = [
        line for line in result.stdout.splitlines() if line.startswith(f"RF metric={metric}")
    ]
    expected = 2 if metric in {"zin", "zout"} else 8
    assert len(rf_lines) == expected


def test_rf_repeat_and_composition_are_canonical_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    expected_metrics = ("y", "s", "zin", "zout")
    calls: list[tuple[str, ...]] = []

    def _sweep(
        frequencies_hz: Sequence[float],
        sweep_layout: SweepLayout,
        assemble_point: object,
        *,
        rf_request: SweepRFRequest,
    ) -> SweepResult:
        del sweep_layout, assemble_point
        calls.append(tuple(rf_request.metrics))
        assert rf_request.s_conversion_source == "from_z"
        return _sweep_result(
            frequencies=frequencies_hz,
            statuses=("pass", "pass"),
            rf_payloads=_rf_payloads(rf_request.metrics, frequencies=frequencies_hz),
        )

    monkeypatch.setattr(cli_main, "_execute_sweep_with_rf", _sweep)

    first = runner.invoke(
        cli_main.app,
        [
            "run",
            "design.net",
            "--analysis",
            "ac",
            "--rf",
            "zout",
            "--rf",
            "s",
            "--rf",
            "y",
            "--rf",
            "zin",
            "--rf",
            "y",
        ],
    )
    second = runner.invoke(
        cli_main.app,
        [
            "run",
            "design.net",
            "--analysis",
            "ac",
            "--rf",
            "y",
            "--rf",
            "zin",
            "--rf",
            "zout",
            "--rf",
            "s",
            "--rf",
            "zin",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert calls == [expected_metrics, expected_metrics]
    assert _rf_metric_block_order(first.stdout) == expected_metrics
    assert _rf_metric_block_order(second.stdout) == expected_metrics
    assert first.stdout == second.stdout


def test_invalid_rf_metric_fails_with_structured_deterministic_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    called = {"sweep": 0}

    def _sweep(*args: object, **kwargs: object) -> SweepResult:
        called["sweep"] += 1
        raise AssertionError("sweep must not run for invalid --rf input")

    monkeypatch.setattr(cli_main, "_execute_sweep_with_rf", _sweep)

    result = runner.invoke(
        cli_main.app,
        ["run", "design.net", "--analysis", "ac", "--rf", "bad2", "--rf", "bad1"],
    )

    assert result.exit_code == _EXIT_FAIL
    assert called["sweep"] == 0
    diag_lines = [line for line in result.stdout.splitlines() if line.startswith("DIAG severity=")]
    assert any(
        "stage=parse" in line and "code=E_CLI_RF_METRIC_INVALID" in line for line in diag_lines
    )
    assert "message=unsupported --rf value 'bad1'" in diag_lines[0]
    assert "message=unsupported --rf value 'bad2'" in diag_lines[1]


def test_missing_rf_ports_is_invalid_combination_with_machine_mappable_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle(with_rf_ports=False)
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    called = {"sweep": 0}

    def _sweep(*args: object, **kwargs: object) -> SweepResult:
        called["sweep"] += 1
        raise AssertionError("sweep must not run when rf ports are missing")

    monkeypatch.setattr(cli_main, "_execute_sweep_with_rf", _sweep)

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac", "--rf", "y"])

    assert result.exit_code == _EXIT_FAIL
    assert called["sweep"] == 0
    assert "code=E_CLI_RF_OPTIONS_INVALID" in result.stdout
    assert "POINT index=" not in result.stdout


@pytest.mark.parametrize(
    ("statuses", "expected_exit"),
    [
        (("pass", "pass"), 0),
        (("pass", "degraded"), 1),
        (("pass", "fail"), 2),
    ],
)
def test_exit_code_contract_unchanged_with_rf(
    monkeypatch: pytest.MonkeyPatch,
    statuses: Sequence[str],
    expected_exit: int,
) -> None:
    bundle = _build_bundle(frequencies=(1.0, 2.0))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep_with_rf",
        lambda freq, layout, assemble, rf_request: _sweep_result(
            frequencies=freq,
            statuses=statuses,
            rf_payloads=_rf_payloads((cast(RFMetricName, "y"),), frequencies=freq),
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac", "--rf", "y"])
    assert result.exit_code == expected_exit


def test_point_and_diag_lines_remain_backward_compatible_with_additive_rf_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    diagnostics_by_point = (
        (
            SweepDiagnostic(
                code="E_NUM_SOLVE_FAILED",
                severity="error",
                message="fail",
                suggested_action="act",
                solver_stage="solve",
                point_index=0,
                frequency_hz=1.0,
            ),
        ),
        (),
    )

    monkeypatch.setattr(
        cli_main,
        "_execute_sweep_with_rf",
        lambda freq, layout, assemble, rf_request: _sweep_result(
            frequencies=freq,
            statuses=("fail", "pass"),
            diagnostics_by_point=diagnostics_by_point,
            rf_payloads=_rf_payloads((cast(RFMetricName, "y"),), frequencies=freq),
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac", "--rf", "y"])

    assert result.exit_code == _EXIT_FAIL
    assert "POINT index=0 freq_hz=1 status=fail" in result.stdout
    assert "POINT index=1 freq_hz=2 status=pass" in result.stdout
    assert (
        "DIAG point_index=0 frequency_hz=1 frequency_index=0 element_id=sweep_engine"
        in result.stdout
    )
    assert "RF metric=y point_index=0 frequency_hz=1 status=pass" in result.stdout

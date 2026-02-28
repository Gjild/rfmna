from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import typer
from numpy.typing import NDArray
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage
from rfmna.parser import PreflightInput
from rfmna.sweep_engine import SweepDiagnostic, SweepLayout, SweepResult

pytestmark = pytest.mark.unit

runner = CliRunner()
EXIT_WARN_OR_FAIL = 2
MIN_EXPECTED_DIAGS = 2


def _diag_event(
    *,
    code: str,
    severity: Severity,
    message: str,
) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=code,
        severity=severity,
        message=message,
        suggested_action="act",
        solver_stage=SolverStage.PREFLIGHT,
        element_id="E1",
    )


def _build_bundle(frequencies: Sequence[float]) -> cli_main.CliDesignBundle:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, np.ndarray]:
        del index, frequency_hz
        raise AssertionError("assemble_point should not be called in CLI tests")

    return cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=tuple(frequencies),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )


def _sweep_result(
    *,
    frequencies: Sequence[float],
    statuses: Sequence[str],
    diagnostics_by_point: tuple[tuple[SweepDiagnostic, ...], ...] | None = None,
) -> SweepResult:
    n_points = len(frequencies)
    status_array = np.asarray(statuses, dtype=np.dtype("<U8"))
    if diagnostics_by_point is None:
        diagnostics_by_point = tuple(() for _ in range(n_points))
    return SweepResult(
        n_points=n_points,
        n_nodes=1,
        n_aux=0,
        V_nodes=np.zeros((n_points, 1), dtype=np.complex128),
        I_aux=np.zeros((n_points, 0), dtype=np.complex128),
        res_l2=np.zeros(n_points, dtype=np.float64),
        res_linf=np.zeros(n_points, dtype=np.float64),
        res_rel=np.zeros(n_points, dtype=np.float64),
        cond_ind=np.ones(n_points, dtype=np.float64),
        status=status_array,
        diagnostics_by_point=diagnostics_by_point,
    )


@dataclass(frozen=True, slots=True)
class _SweepCallLog:
    called: bool = False


def test_run_exit_0_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0, 2.0))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(frequencies=freq, statuses=("pass", "pass")),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == 0


def test_run_exit_1_degraded_without_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0, 2.0, 3.0))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(
            frequencies=freq, statuses=("pass", "degraded", "pass")
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == 1


def test_run_exit_2_on_any_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0, 2.0, 3.0))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(
            frequencies=freq, statuses=("pass", "fail", "degraded")
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == EXIT_WARN_OR_FAIL


def test_run_preflight_error_forces_exit_2_and_skips_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0, 2.0))
    sweep_log = {"called": False}
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(
        cli_main,
        "_execute_preflight",
        lambda _: (
            _diag_event(
                code="E_TOPO_REFERENCE_INVALID", severity=Severity.ERROR, message="bad ref"
            ),
        ),
    )

    def _sweep(*args: object, **kwargs: object) -> SweepResult:
        sweep_log["called"] = True
        raise AssertionError("sweep must not be called when preflight has errors")

    monkeypatch.setattr(cli_main, "_execute_sweep", _sweep)
    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])

    assert result.exit_code == EXIT_WARN_OR_FAIL
    assert sweep_log["called"] is False
    assert "run blocked: preflight contains error diagnostics" in result.stdout


def test_run_analysis_guard_non_ac_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0,))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "tran"])
    assert result.exit_code != 0


def test_check_exit_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((1.0,))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)

    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    ok = runner.invoke(cli_main.app, ["check", "design.net"])
    assert ok.exit_code == 0

    monkeypatch.setattr(
        cli_main,
        "_execute_preflight",
        lambda _: (
            _diag_event(code="W_NUM_ILL_CONDITIONED", severity=Severity.WARNING, message="warn"),
        ),
    )
    warn = runner.invoke(cli_main.app, ["check", "design.net"])
    assert warn.exit_code == 0

    monkeypatch.setattr(
        cli_main,
        "_execute_preflight",
        lambda _: (
            _diag_event(code="E_TOPO_FLOATING_NODE", severity=Severity.ERROR, message="err"),
        ),
    )
    err = runner.invoke(cli_main.app, ["check", "design.net"])
    assert err.exit_code == EXIT_WARN_OR_FAIL


def test_check_unexpected_exception_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_main, "_load_design_bundle", lambda design: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = runner.invoke(cli_main.app, ["check", "design.net"])
    assert result.exit_code == EXIT_WARN_OR_FAIL
    assert "code=E_CLI_CHECK_INTERNAL" in result.stdout
    assert "message=check internal failure: boom" in result.stdout


def test_check_loader_boundary_failure_emits_typed_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_loader_error(_: str) -> cli_main.CliDesignBundle:
        raise typer.BadParameter("loader missing")

    monkeypatch.setattr(cli_main, "_load_design_bundle", _raise_loader_error)

    result = runner.invoke(cli_main.app, ["check", "design.net"])

    assert result.exit_code == EXIT_WARN_OR_FAIL
    assert "code=E_CLI_CHECK_LOADER_FAILED" in result.stdout
    assert "message=check loader boundary failed: loader missing" in result.stdout


def test_check_json_output_is_machine_parseable_and_permutation_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle((1.0,))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)

    warning = DiagnosticEvent(
        code="W_NUM_ILL_CONDITIONED",
        severity=Severity.WARNING,
        message="warn",
        suggested_action="act",
        solver_stage=SolverStage.SOLVE,
        element_id="E2",
        witness={"b": 2, "a": 1},
    )
    error = DiagnosticEvent(
        code="E_TOPO_FLOATING_NODE",
        severity=Severity.ERROR,
        message="err",
        suggested_action="act",
        solver_stage=SolverStage.PREFLIGHT,
        element_id="E1",
        witness={"z": {"k2": 2, "k1": 1}},
    )
    permutations = ((warning, error), (error, warning))
    call_index = {"value": 0}

    def _preflight(_: PreflightInput) -> tuple[DiagnosticEvent, ...]:
        index = call_index["value"]
        call_index["value"] += 1
        return permutations[index]

    monkeypatch.setattr(cli_main, "_execute_preflight", _preflight)

    first = runner.invoke(cli_main.app, ["check", "design.net", "--format", "json"])
    second = runner.invoke(cli_main.app, ["check", "design.net", "--format", "json"])

    assert first.exit_code == EXIT_WARN_OR_FAIL
    assert second.exit_code == EXIT_WARN_OR_FAIL
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    assert payload["schema"] == "docs/spec/schemas/check_output_v1.json"
    assert payload["schema_version"] == 1
    assert payload["design"] == "design.net"
    assert payload["status"] == "fail"
    assert payload["exit_code"] == EXIT_WARN_OR_FAIL
    assert [diag["code"] for diag in payload["diagnostics"]] == [
        "E_TOPO_FLOATING_NODE",
        "W_NUM_ILL_CONDITIONED",
    ]
    assert payload["diagnostics"][0]["witness"] == {"z": {"k1": 1, "k2": 2}}
    assert payload["diagnostics"][1]["witness"] == {"a": 1, "b": 2}


def test_run_unexpected_exception_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_main, "_load_design_bundle", lambda design: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == EXIT_WARN_OR_FAIL
    assert "internal error: boom" in result.stdout


def test_deterministic_output_ordering_and_fail_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle((10.0, 5.0, 1.0))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(
        cli_main,
        "_execute_preflight",
        lambda _: (
            _diag_event(code="W_X", severity=Severity.WARNING, message="later"),
            _diag_event(code="W_A", severity=Severity.WARNING, message="first"),
        ),
    )
    diagnostics = (
        (
            SweepDiagnostic(
                code="W_NUM_ILL_CONDITIONED",
                severity="warning",
                message="warn",
                suggested_action="act",
                solver_stage="solve",
                point_index=0,
                frequency_hz=10.0,
            ),
        ),
        (
            SweepDiagnostic(
                code="E_NUM_SOLVE_FAILED",
                severity="error",
                message="failed",
                suggested_action="act",
                solver_stage="solve",
                point_index=1,
                frequency_hz=5.0,
            ),
        ),
        (),
    )
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(
            frequencies=freq,
            statuses=("pass", "fail", "degraded"),
            diagnostics_by_point=diagnostics,
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == EXIT_WARN_OR_FAIL
    lines = [line for line in result.stdout.splitlines() if line.strip()]

    diag_lines = [line for line in lines if line.startswith("DIAG severity=")]
    assert len(diag_lines) >= MIN_EXPECTED_DIAGS
    assert "code=W_A" in diag_lines[0]
    assert "code=W_X" in diag_lines[1]

    point_lines = [line for line in lines if line.startswith("POINT index=")]
    assert [line.split()[1] for line in point_lines] == ["index=0", "index=1", "index=2"]
    assert "status=fail" in point_lines[1]


def test_run_prints_sweep_diagnostics_with_severity_first_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle((10.0,))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    diagnostics = (
        (
            SweepDiagnostic(
                code="W_NUM_ILL_CONDITIONED",
                severity="warning",
                message="warn",
                suggested_action="act",
                solver_stage="solve",
                point_index=0,
                frequency_hz=10.0,
                witness={"k": 2},
            ),
            SweepDiagnostic(
                code="E_NUM_SOLVE_FAILED",
                severity="error",
                message="bad shape",
                suggested_action="act",
                solver_stage="postprocess",
                point_index=0,
                frequency_hz=10.0,
                witness={"k": 1},
            ),
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(
            frequencies=freq,
            statuses=("fail",),
            diagnostics_by_point=diagnostics,
        ),
    )

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == EXIT_WARN_OR_FAIL
    diag_lines = [
        line for line in result.stdout.splitlines() if line.startswith("DIAG point_index=0")
    ]
    assert len(diag_lines) == MIN_EXPECTED_DIAGS
    assert "severity=error" in diag_lines[0]
    assert "code=E_NUM_SOLVE_FAILED" in diag_lines[0]
    assert "severity=warning" in diag_lines[1]
    assert "code=W_NUM_ILL_CONDITIONED" in diag_lines[1]


def test_run_prints_sweep_diagnostics_with_permutation_stable_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle((10.0,))
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    base_diagnostics = (
        SweepDiagnostic(
            code="W_BETA",
            severity="warning",
            message="warn solve",
            suggested_action="act",
            solver_stage="solve",
            point_index=0,
            frequency_hz=10.0,
            witness={"rank": 2},
        ),
        SweepDiagnostic(
            code="W_ALPHA",
            severity="warning",
            message="warn parse",
            suggested_action="act",
            solver_stage="parse",
            point_index=0,
            frequency_hz=10.0,
            witness={"rank": 1},
        ),
        SweepDiagnostic(
            code="E_NUM_SOLVE_FAILED",
            severity="error",
            message="fail",
            suggested_action="act",
            solver_stage="postprocess",
            point_index=0,
            frequency_hz=10.0,
            witness={"rank": 0},
        ),
    )
    permutations = (
        (base_diagnostics[0], base_diagnostics[1], base_diagnostics[2]),
        (base_diagnostics[2], base_diagnostics[0], base_diagnostics[1]),
    )

    call_index = {"value": 0}

    def _sweep(
        frequencies_hz: Sequence[float] | NDArray[np.float64],
        sweep_layout: SweepLayout,
        assemble_point: object,
    ) -> SweepResult:
        del sweep_layout, assemble_point
        diagnostics = (permutations[call_index["value"]],)
        call_index["value"] += 1
        return _sweep_result(
            frequencies=frequencies_hz,
            statuses=("fail",),
            diagnostics_by_point=diagnostics,
        )

    monkeypatch.setattr(cli_main, "_execute_sweep", _sweep)

    first = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    second = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])

    assert first.exit_code == EXIT_WARN_OR_FAIL
    assert second.exit_code == EXIT_WARN_OR_FAIL
    first_diags = [
        line for line in first.stdout.splitlines() if line.startswith("DIAG point_index=0")
    ]
    second_diags = [
        line for line in second.stdout.splitlines() if line.startswith("DIAG point_index=0")
    ]
    assert first_diags == second_diags
    assert "code=E_NUM_SOLVE_FAILED" in first_diags[0]
    assert "code=W_ALPHA" in first_diags[1]
    assert "code=W_BETA" in first_diags[2]


def test_thin_wrapper_calls_expected_apis_and_preflight_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_bundle((1.0, 2.0))
    calls = {"loader": 0, "preflight": 0, "sweep": 0}

    def _loader(design: str) -> cli_main.CliDesignBundle:
        calls["loader"] += 1
        assert design == "design.net"
        return bundle

    def _preflight(_: PreflightInput) -> tuple[DiagnosticEvent, ...]:
        calls["preflight"] += 1
        return ()

    def _sweep(
        frequencies_hz: Sequence[float] | NDArray[np.float64],
        sweep_layout: SweepLayout,
        assemble_point: object,
    ) -> SweepResult:
        calls["sweep"] += 1
        assert tuple(frequencies_hz) == (1.0, 2.0)
        assert sweep_layout == bundle.sweep_layout
        assert assemble_point == bundle.assemble_point
        return _sweep_result(frequencies=frequencies_hz, statuses=("pass", "pass"))

    monkeypatch.setattr(cli_main, "_load_design_bundle", _loader)
    monkeypatch.setattr(cli_main, "_execute_preflight", _preflight)
    monkeypatch.setattr(cli_main, "_execute_sweep", _sweep)

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == 0
    assert calls == {"loader": 1, "preflight": 1, "sweep": 1}

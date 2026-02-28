from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage
from rfmna.parser import PreflightInput
from rfmna.sweep_engine import SweepLayout, SweepResult

pytestmark = pytest.mark.conformance

runner = CliRunner()
EXIT_FAIL = 2


def _bundle() -> cli_main.CliDesignBundle:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, np.ndarray]:
        del index, frequency_hz
        raise AssertionError("assemble_point not expected in CLI conformance tests")

    return cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=(1.0, 2.0, 3.0),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )


def _diag_error() -> DiagnosticEvent:
    return DiagnosticEvent(
        code="E_TOPO_REFERENCE_INVALID",
        severity=Severity.ERROR,
        message="bad",
        suggested_action="fix",
        solver_stage=SolverStage.PREFLIGHT,
        element_id="REF",
    )


def _sweep_result(statuses: Sequence[str]) -> SweepResult:
    n = len(statuses)
    return SweepResult(
        n_points=n,
        n_nodes=1,
        n_aux=0,
        V_nodes=np.zeros((n, 1), dtype=np.complex128),
        I_aux=np.zeros((n, 0), dtype=np.complex128),
        res_l2=np.zeros(n, dtype=np.float64),
        res_linf=np.zeros(n, dtype=np.float64),
        res_rel=np.zeros(n, dtype=np.float64),
        cond_ind=np.ones(n, dtype=np.float64),
        status=np.asarray(tuple(statuses), dtype=np.dtype("<U8")),
        diagnostics_by_point=tuple(() for _ in range(n)),
    )


def test_run_exit_mapping_and_preflight_gate_and_fail_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())

    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(("pass", "pass", "pass")),
    )
    all_pass = runner.invoke(cli_main.app, ["run", "d.net", "--analysis", "ac"])
    assert all_pass.exit_code == 0

    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(("pass", "degraded", "pass")),
    )
    degraded = runner.invoke(cli_main.app, ["run", "d.net", "--analysis", "ac"])
    assert degraded.exit_code == 1

    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _sweep_result(("pass", "fail", "pass")),
    )
    fail = runner.invoke(cli_main.app, ["run", "d.net", "--analysis", "ac"])
    assert fail.exit_code == EXIT_FAIL
    assert "POINT index=1" in fail.stdout
    assert "status=fail" in fail.stdout

    calls = {"sweep": 0}
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: (_diag_error(),))

    def _sweep(*args: object, **kwargs: object) -> SweepResult:
        calls["sweep"] += 1
        return _sweep_result(("pass", "pass", "pass"))

    monkeypatch.setattr(cli_main, "_execute_sweep", _sweep)
    gated = runner.invoke(cli_main.app, ["run", "d.net", "--analysis", "ac"])
    assert gated.exit_code == EXIT_FAIL
    assert calls["sweep"] == 0


def test_check_exit_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle()
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)

    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    ok = runner.invoke(cli_main.app, ["check", "d.net"])
    assert ok.exit_code == 0

    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: (_diag_error(),))
    err = runner.invoke(cli_main.app, ["check", "d.net"])
    assert err.exit_code == EXIT_FAIL


def test_unexpected_exceptions_map_to_contract_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(design: str) -> cli_main.CliDesignBundle:
        del design
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_main, "_load_design_bundle", _boom)

    check_result = runner.invoke(cli_main.app, ["check", "d.net"])
    run_result = runner.invoke(cli_main.app, ["run", "d.net", "--analysis", "ac"])

    assert check_result.exit_code == EXIT_FAIL
    assert run_result.exit_code == EXIT_FAIL

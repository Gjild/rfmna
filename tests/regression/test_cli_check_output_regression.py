from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage
from rfmna.parser import PreflightInput
from rfmna.sweep_engine import SweepLayout

pytestmark = pytest.mark.regression

runner = CliRunner()
_EXIT_FAIL = 2


def _bundle() -> cli_main.CliDesignBundle:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, np.ndarray]:
        del index, frequency_hz
        raise AssertionError("assemble_point not expected in check regression tests")

    return cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=(1.0,),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )


def test_check_text_mode_grammar_and_ordering_are_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: _bundle())
    diagnostics = (
        DiagnosticEvent(
            code="W_NUM_ILL_CONDITIONED",
            severity=Severity.WARNING,
            message="warn-second",
            suggested_action="act",
            solver_stage=SolverStage.PREFLIGHT,
            element_id="E2",
        ),
        DiagnosticEvent(
            code="E_TOPO_FLOATING_NODE",
            severity=Severity.ERROR,
            message="err-first",
            suggested_action="act",
            solver_stage=SolverStage.PREFLIGHT,
            element_id="E1",
        ),
    )
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: diagnostics)

    result = runner.invoke(cli_main.app, ["check", "design.net"])

    assert result.exit_code == _EXIT_FAIL
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        "DIAG severity=error stage=preflight code=E_TOPO_FLOATING_NODE message=err-first",
        "DIAG severity=warning stage=preflight code=W_NUM_ILL_CONDITIONED message=warn-second",
    ]

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.rf_metrics import PortBoundary
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.conformance

_DOC_PATHS = (
    "README.md",
    "docs/dev/phase1_usage.md",
    "docs/dev/cli_rf_composition_matrix.md",
)
_NODEID_RE = re.compile(r"tests/[A-Za-z0-9_./-]+::[A-Za-z0-9_]+")
_ASPIRATIONAL_PHRASES = (
    "coming soon",
    "future work",
    "to be implemented",
    "planned but not implemented",
    "not yet implemented",
)
_SENTINEL_EVIDENCE_NODEIDS = {
    "tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence",
    "tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_s_from_z_dependency_failure_propagates_to_sentinel_and_diagnostics",
}
_COMPOSITION_EVIDENCE_NODEIDS = {
    "tests/unit/test_cli_rf_options.py::test_rf_repeat_and_composition_are_canonical_and_deterministic",
    "tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_rf_composition_matrix_rows_use_explicit_dependency_paths",
    "tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_rf_composition_canonical_execution_order_is_request_order_invariant",
}
_CLI_HELP_COMMANDS = (
    "uv run rfmna --help",
    "uv run rfmna check --help",
    "uv run rfmna run --help",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_text(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _extract_nodeids(text: str) -> set[str]:
    return set(_NODEID_RE.findall(text))


def _cli_args_from_uv_command(command: str) -> list[str]:
    prefix = "uv run rfmna "
    assert command.startswith(prefix)
    suffix = command[len(prefix) :]
    return [token for token in suffix.split(" ") if token]


def _discover_repo_test_nodeids() -> set[str]:
    root = _repo_root()
    nodeids: set[str] = set()
    for test_file in sorted((root / "tests").rglob("test_*.py")):
        relative = test_file.relative_to(root).as_posix()
        content = test_file.read_text(encoding="utf-8")
        for match in re.finditer(r"^def (test_[A-Za-z0-9_]+)\(", content, re.MULTILINE):
            nodeids.add(f"{relative}::{match.group(1)}")
    return nodeids


def test_phase1_docs_reference_resolvable_test_nodeids() -> None:
    collected_nodeids = _discover_repo_test_nodeids()
    missing: list[str] = []

    for path in _DOC_PATHS:
        references = _extract_nodeids(_read_repo_text(path))
        assert references, f"{path} must include explicit test nodeid evidence"
        for nodeid in sorted(references):
            if nodeid not in collected_nodeids:
                missing.append(f"{path}:{nodeid}")

    assert missing == []


def test_phase1_docs_are_non_aspirational() -> None:
    for path in _DOC_PATHS:
        lowered = _read_repo_text(path).lower()
        for phrase in _ASPIRATIONAL_PHRASES:
            assert phrase not in lowered, f"{path} contains aspirational phrase: {phrase}"


def test_sentinel_and_cli_composition_claims_have_test_evidence() -> None:
    for path in _DOC_PATHS:
        text = _read_repo_text(path)
        references = _extract_nodeids(text)
        lowered = text.lower()
        if "sentinel" in lowered:
            assert references & _SENTINEL_EVIDENCE_NODEIDS, (
                f"{path} mentions sentinel behavior without sentinel evidence nodeids"
            )
        if "--rf" in text or "composition" in lowered:
            assert references & _COMPOSITION_EVIDENCE_NODEIDS, (
                f"{path} mentions CLI composition without composition evidence nodeids"
            )


def test_documented_cli_help_commands_remain_executable() -> None:
    readme = _read_repo_text("README.md")
    usage = _read_repo_text("docs/dev/phase1_usage.md")
    for command in _CLI_HELP_COMMANDS:
        assert command in readme or command in usage

    runner = CliRunner()
    for command in _CLI_HELP_COMMANDS:
        result = runner.invoke(cli_main.app, _cli_args_from_uv_command(command))
        assert result.exit_code == 0, f"documented command failed: {command}"


def test_phase1_usage_marks_design_loader_dependent_commands_as_bounded() -> None:
    usage = _read_repo_text("docs/dev/phase1_usage.md")
    assert "require a project-specific design-loader integration" in usage
    assert "Without design-loader wiring, CLI returns a typed parameter error" in usage


def test_phase1_usage_api_example_is_executable_in_repo_context() -> None:
    frequencies = np.asarray((1.0, 2.0), dtype=np.float64)
    layout = SweepLayout(n_nodes=2, n_aux=0)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("y", "z", "s", "zin", "zout"),
    )

    def assemble_point(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
        y = np.asarray(
            [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]],
            dtype=np.complex128,
        )
        return csc_matrix(y), np.zeros(2, dtype=np.complex128)

    result = run_sweep(frequencies, layout, assemble_point, rf_request=request)

    assert result.status.tolist() == ["pass", "pass"]
    assert result.rf_payloads is not None
    assert result.rf_payloads.metric_names == ("y", "z", "s", "zin", "zout")

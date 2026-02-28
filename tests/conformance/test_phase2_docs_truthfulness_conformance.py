from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.conformance

_DOC_PATHS = ("README.md", "docs/dev/phase2_usage.md")
_NODEID_RE = re.compile(r"tests/[A-Za-z0-9_./-]+::[A-Za-z0-9_]+")
_ASPIRATIONAL_PHRASES = (
    "coming soon",
    "future work",
    "to be implemented",
    "planned but not implemented",
    "not yet implemented",
)
_REQUIRED_VALIDATION_COMMANDS = (
    "uv run ruff check .",
    "uv run mypy src",
    "uv run pytest -m unit",
    "uv run pytest -m conformance",
    "uv run pytest -m property",
    "uv run pytest -m regression",
    "uv run pytest -m cross_check",
)
_REQUIRED_PHASE2_EVIDENCE_NODEIDS = {
    "tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_a_runtime_inventory_guard_passes_baseline",
    "tests/conformance/test_check_command_contract_conformance.py::test_check_json_output_validates_against_canonical_schema",
    "tests/property/test_preflight_properties.py::test_preflight_input_order_permutation_invariance",
    "tests/regression/test_regression_golden_tolerance_suite.py::test_regression_goldens_are_stable_and_tolerance_enforced",
    "tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_text(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _extract_nodeids(text: str) -> set[str]:
    return set(_NODEID_RE.findall(text))


def _discover_repo_test_nodeids() -> set[str]:
    root = _repo_root()
    nodeids: set[str] = set()
    for test_file in sorted((root / "tests").rglob("test_*.py")):
        relative = test_file.relative_to(root).as_posix()
        content = test_file.read_text(encoding="utf-8")
        for match in re.finditer(r"^def (test_[A-Za-z0-9_]+)\(", content, re.MULTILINE):
            nodeids.add(f"{relative}::{match.group(1)}")
    return nodeids


def test_phase2_docs_reference_resolvable_test_nodeids() -> None:
    collected_nodeids = _discover_repo_test_nodeids()
    missing: list[str] = []

    for path in _DOC_PATHS:
        references = _extract_nodeids(_read_repo_text(path))
        assert references, f"{path} must include explicit test nodeid evidence"
        for nodeid in sorted(references):
            if nodeid not in collected_nodeids:
                missing.append(f"{path}:{nodeid}")

    assert missing == []


def test_phase2_docs_are_non_aspirational() -> None:
    for path in _DOC_PATHS:
        lowered = _read_repo_text(path).lower()
        for phrase in _ASPIRATIONAL_PHRASES:
            assert phrase not in lowered, f"{path} contains aspirational phrase: {phrase}"


def test_readme_lists_post_p2_09_validation_commands() -> None:
    readme = _read_repo_text("README.md")
    for command in _REQUIRED_VALIDATION_COMMANDS:
        assert command in readme
    assert "docs/dev/phase2_usage.md" in readme


def test_phase2_usage_covers_required_topics_with_evidence() -> None:
    usage = _read_repo_text("docs/dev/phase2_usage.md")
    for heading in (
        "Hardened `check` Contract",
        "Diagnostics Catalog Closure (Phase 2)",
        "Property/Regression/Cross-Check Workflow",
        "CI Lanes and Governance Traceability",
    ):
        assert heading in usage

    references = _extract_nodeids(usage)
    assert references >= _REQUIRED_PHASE2_EVIDENCE_NODEIDS

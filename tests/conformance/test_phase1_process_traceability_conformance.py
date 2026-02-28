from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.conformance

_TRACEABILITY_DOC = "docs/dev/phase1_process_traceability.md"
_PHASE1_GATE_DOC = "docs/dev/phase1_gate.md"

_REQUIRED_HEADINGS = (
    "## Assumptions",
    "## Scope Boundaries",
    "## Governance Links",
)

_REQUIRED_KEYS = (
    "`phase_contract`",
    "`design_loader_boundary`",
    "`test_evidence_scope`",
    "`in_scope`",
    "`out_of_scope`",
    "`authority_backlog`",
    "`authority_agents`",
    "`baseline_dr`",
    "`baseline_migration_note`",
    "`phase_gate`",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_text(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def test_phase1_traceability_artifact_exists_and_declares_required_sections() -> None:
    content = _read_repo_text(_TRACEABILITY_DOC)
    for heading in _REQUIRED_HEADINGS:
        assert heading in content
    for key in _REQUIRED_KEYS:
        assert key in content


def test_phase1_gate_links_to_traceability_artifact() -> None:
    gate = _read_repo_text(_PHASE1_GATE_DOC)
    assert _TRACEABILITY_DOC in gate

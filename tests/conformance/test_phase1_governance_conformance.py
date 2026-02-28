from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.conformance

_POLICY_FRAGMENT = "No unapproved normative/spec edits"
_EVIDENCE_LINE = "DR + migration note + conformance evidence"
_BASELINE_DR = "docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md"
_BASELINE_MIGRATION = "docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md"
_FORBIDDEN_ABSOLUTE_LINE = "No normative/spec edits."
_V4_CONTRACT = "docs/spec/v4_contract.md"
_PHASE1_GATE = "docs/dev/phase1_gate.md"

_DR_FROZEN_SOURCES = {
    "docs/spec/v4_contract.md",
    "docs/spec/frozen_artifacts_v4_0_0.md",
    "docs/spec/stamp_appendix_v4_0_0.md",
    "docs/spec/port_wave_conventions_v4_0_0.md",
    "docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md",
    "docs/spec/thresholds_v4_0_0.yaml",
    "docs/spec/diagnostics_taxonomy_v4_0_0.md",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_text(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _extract_inline_paths(text: str, *, prefix: str) -> set[str]:
    paths = {
        match.group(1)
        for match in re.finditer(r"`([^`]+)`", text)
        if match.group(1).startswith(prefix)
    }
    return paths


def test_p1_00_backlog_governance_policy_is_explicit_and_auditable() -> None:
    backlog = _read_repo_text("docs/dev/codex_backlog.md")

    assert _POLICY_FRAGMENT in backlog
    assert _EVIDENCE_LINE in backlog
    assert _BASELINE_DR in backlog
    assert _BASELINE_MIGRATION in backlog
    assert _FORBIDDEN_ABSOLUTE_LINE not in backlog


def test_phase1_gate_governance_policy_requires_change_control_evidence() -> None:
    gate = _read_repo_text(_PHASE1_GATE)

    assert _POLICY_FRAGMENT in gate
    assert _EVIDENCE_LINE in gate
    assert _BASELINE_DR in gate
    assert _BASELINE_MIGRATION in gate
    assert "Confirm no normative/spec edits are included in task scope." not in gate

    for required_line in (
        "- Semantic version bump.",
        "- Decision record in `docs/spec/decision_records/`.",
        "- Conformance updates.",
        "- Migration note.",
        "- Reproducibility-impact statement.",
    ):
        assert required_line in gate


def test_baseline_governance_dr_and_migration_note_are_cross_linked_and_accepted() -> None:
    dr_path = _repo_root() / _BASELINE_DR
    migration_path = _repo_root() / _BASELINE_MIGRATION
    assert dr_path.exists()
    assert migration_path.exists()

    dr = _read_repo_text(_BASELINE_DR)
    migration = _read_repo_text(_BASELINE_MIGRATION)

    assert "Status: `accepted`" in dr
    assert _BASELINE_MIGRATION in dr
    assert f"Related DR: `{_BASELINE_DR}`" in migration


def test_baseline_dr_declares_auditable_frozen_sources_that_exist() -> None:
    dr = _read_repo_text(_BASELINE_DR)
    declared_paths = _extract_inline_paths(dr, prefix="docs/spec/")

    assert _DR_FROZEN_SOURCES.issubset(declared_paths)
    for path in sorted(_DR_FROZEN_SOURCES):
        full_path = _repo_root() / path
        assert full_path.exists()
        assert full_path.read_text(encoding="utf-8").strip()


def test_phase_gate_frozen_artifact_paths_are_present_and_auditable() -> None:
    gate = _read_repo_text(_PHASE1_GATE)
    path_references = _extract_inline_paths(gate, prefix="docs/spec/") | _extract_inline_paths(
        gate,
        prefix=".github/",
    )

    expected = {
        "docs/spec/v4_contract.md",
        "docs/spec/stamp_appendix_v4_0_0.md",
        "docs/spec/port_wave_conventions_v4_0_0.md",
        "docs/spec/thresholds_v4_0_0.yaml",
        "docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md",
        ".github/workflows/ci.yml",
    }
    assert expected.issubset(path_references)
    for path in sorted(expected):
        assert (_repo_root() / path).exists()


def test_backlog_and_phase_gate_governance_language_remains_aligned() -> None:
    backlog = _read_repo_text("docs/dev/codex_backlog.md")
    gate = _read_repo_text(_PHASE1_GATE)
    contract = _read_repo_text(_V4_CONTRACT)

    for required_fragment in (
        "No unapproved normative/spec edits",
        "DR + migration note + conformance evidence",
    ):
        assert required_fragment in backlog
        assert required_fragment in gate

    assert _BASELINE_DR in backlog
    assert _BASELINE_MIGRATION in backlog
    assert _BASELINE_DR in gate
    assert _BASELINE_MIGRATION in gate
    assert _BASELINE_DR in contract
    assert _BASELINE_MIGRATION in contract

    for forbidden in (
        _FORBIDDEN_ABSOLUTE_LINE,
        "No normative/spec changes are allowed under any circumstances.",
    ):
        assert forbidden not in backlog
        assert forbidden not in gate

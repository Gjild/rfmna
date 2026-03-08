from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from rfmna.governance import phase3_gate as phase3_gate_module
from rfmna.governance.phase2_gate import (
    DetectionPattern,
    FrozenRule,
    GovernanceArtifactPaths,
    _load_rule_table,
    evaluate_category_bootstrap_gate,
    evaluate_governance_gate,
)
from rfmna.governance.phase3_gate import (
    Phase3ArtifactPaths,
    _base_ref_commit_utc_date,
    _load_phase3_rule_table,
    derive_touched_phase3_surface_ids,
    evaluate_anti_tamper_gate,
    evaluate_contract_surface_governance_gate,
    evaluate_optional_track_gate,
    main,
)

pytestmark = pytest.mark.conformance

_FROZEN_ID_THREAD_DEFAULTS = 12

_PHASE3_SURFACE_EVIDENCE = {
    "policy_docs": [
        "docs/dev/optional_track_activation_policy.md",
        "docs/dev/phase3_change_surface_policy.md",
        "docs/dev/phase3_gate.md",
    ],
    "schema_artifacts": [
        "docs/dev/optional_track_activation_schema_v1.json",
        "docs/dev/phase3_change_surface_schema_v1.json",
    ],
    "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
    "ci_enforcement": [".github/workflows/ci.yml"],
    "process_traceability": ["docs/dev/phase3_process_traceability.md"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_text(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _ci_test_job() -> dict[str, object]:
    payload = yaml.safe_load((_repo_root() / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    job = jobs.get("test")
    assert isinstance(job, dict)
    return job


def _ci_named_steps() -> dict[str, tuple[int, dict[str, object]]]:
    steps = _ci_test_job().get("steps")
    assert isinstance(steps, list)
    named: dict[str, tuple[int, dict[str, object]]] = {}
    for index, step in enumerate(steps):
        if isinstance(step, dict) and isinstance(step.get("name"), str):
            named[step["name"]] = (index, step)
    return named


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _seed_minimal_category_bootstrap_repo(
    repo_root: Path,
    *,
    markers: tuple[str, ...],
    workflow_fragments: tuple[str, ...] | None = None,
) -> None:
    marker_lines = "\n".join(f"    {marker}" for marker in markers)
    _write_text(
        repo_root / "pytest.ini",
        "[pytest]\naddopts = -ra --strict-markers\nmarkers =\n" + marker_lines + "\n",
    )
    _write_text(
        repo_root / "tests/cross_check/test_smoke.py",
        "def test_cross_check_placeholder() -> None:\n    assert True\n",
    )
    _write_text(
        repo_root / "tests/regression/test_smoke.py",
        "def test_regression_placeholder() -> None:\n    assert True\n",
    )
    _write_text(
        repo_root / "docs/dev/regression_fixture_schema_convention.md",
        "regression fixture convention\n",
    )
    _write_text(
        repo_root / "docs/dev/phase2_ci_category_enforcement.md",
        "\n".join(
            (
                "`unit`",
                "`conformance`",
                "`property`",
                "`regression`",
                "`cross_check`",
                "non-empty",
                "--strict-markers",
            )
        )
        + "\n",
    )
    _write_text(
        repo_root / "AGENTS.md",
        "`cross_check` (mandatory for Phase 2 robustness work; reference-comparison tests with documented tolerances)\n",
    )
    workflow = workflow_fragments or (
        "Phase 2 governance sub-gate (blocking)",
        "Phase 2 category bootstrap sub-gate (blocking)",
        "Audit AGENTS Phase 2 cross_check policy alignment",
        "uv run pytest -m unit --collect-only -q",
        "uv run pytest -m unit --junitxml=test-reports/unit.xml",
        "uv run pytest -m conformance --collect-only -q",
        "Tests (thread-controls conformance guard)",
        "test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults",
        "test_thread_controls_conformance.py::test_envrc_declares_deterministic_thread_defaults",
        "uv run pytest -m conformance --junitxml=test-reports/conformance.xml",
        "uv run pytest -m property --collect-only -q",
        "uv run pytest -m property --junitxml=test-reports/property.xml",
        "uv run pytest -m regression --collect-only -q",
        "uv run pytest -m regression --junitxml=test-reports/regression.xml",
        "uv run pytest tests/regression -m regression -q",
        "uv run pytest -m cross_check --collect-only -q",
        "uv run pytest -m cross_check --junitxml=test-reports/cross_check.xml",
        "Upload calibration/regression/cross_check diagnostics (failure)",
    )
    _write_text(repo_root / ".github/workflows/ci.yml", "\n".join(workflow) + "\n")


def _write_phase3_change_surface(
    tmp_path: Path,
    *,
    declared_surface_ids: str | list[str],
    evidence: dict[str, object],
) -> Path:
    return _write_json(
        tmp_path / "phase3_change_surface.yaml",
        {
            "schema_version": 1,
            "declared_surface_ids": declared_surface_ids,
            "evidence": evidence,
        },
    )


def _write_optional_track_activation(  # noqa: PLR0913
    tmp_path: Path,
    *,
    p3_10_state: str,
    p3_10_evidence: dict[str, object] | None,
    p3_10_usage_date: str | None,
    p3_10_impacted_frozen_ids: list[int],
    p3_10_approval: dict[str, object] | None,
) -> Path:
    payload = {
        "schema_version": 1,
        "tracks": [
            {
                "track_id": "p3_10_cccs_ccvs",
                "state": p3_10_state,
                "usage_evidence_source": p3_10_evidence,
                "usage_evidence_date": p3_10_usage_date,
                "activation_rationale": "test rationale",
                "impacted_frozen_ids": p3_10_impacted_frozen_ids,
                "approval_record": p3_10_approval,
            },
            {
                "track_id": "p3_11_mutual_inductance",
                "state": "deferred",
                "usage_evidence_source": None,
                "usage_evidence_date": None,
                "activation_rationale": "test rationale",
                "impacted_frozen_ids": [],
                "approval_record": None,
            },
        ],
    }
    return _write_json(tmp_path / "optional_track_activation.yaml", payload)


def _write_change_scope(tmp_path: Path, *, declared_frozen_ids: str | list[int]) -> Path:
    return _write_json(
        tmp_path / "change_scope.yaml",
        {
            "schema_version": 1,
            "declared_frozen_ids": declared_frozen_ids,
            "evidence": {
                "semver_bump": None,
                "decision_records": [],
                "conformance_updates": [],
                "migration_notes": [],
                "reproducibility_impact_statement_path": None,
            },
        },
    )


def _phase3_artifact_paths(  # noqa: PLR0913
    *,
    phase3_change_surface_path: str | None = None,
    phase3_change_surface_schema_path: str | None = None,
    phase3_change_surface_policy_path: str | None = None,
    optional_track_activation_path: str | None = None,
    optional_track_activation_schema_path: str | None = None,
    optional_track_policy_path: str | None = None,
    change_scope_path: str | None = None,
    phase3_rule_table_path: str | None = None,
    frozen_rule_table_path: str | None = None,
) -> Phase3ArtifactPaths:
    return Phase3ArtifactPaths(
        phase3_change_surface_path=phase3_change_surface_path or "docs/dev/phase3_change_surface.yaml",
        phase3_change_surface_schema_path=phase3_change_surface_schema_path
        or "docs/dev/phase3_change_surface_schema_v1.json",
        phase3_change_surface_policy_path=phase3_change_surface_policy_path
        or "docs/dev/phase3_change_surface_policy.md",
        optional_track_activation_path=optional_track_activation_path
        or "docs/dev/optional_track_activation.yaml",
        optional_track_activation_schema_path=optional_track_activation_schema_path
        or "docs/dev/optional_track_activation_schema_v1.json",
        optional_track_policy_path=optional_track_policy_path
        or "docs/dev/optional_track_activation_policy.md",
        change_scope_path=change_scope_path or "docs/dev/change_scope.yaml",
        phase3_rule_table_path=phase3_rule_table_path
        or "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rule_table_path=frozen_rule_table_path or "docs/dev/frozen_change_governance_rules.yaml",
    )


def test_phase3_contract_surface_rule_table_detects_bootstrap_and_optional_contract_paths() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    touched = derive_touched_phase3_surface_ids(
        changed_paths=(
            "docs/dev/phase3_gate.md",
            "docs/dev/optional_track_activation.yaml",
            "src/rfmna/governance/phase3_gate.py",
        ),
        changed_lines_by_path={},
        rules=rule_table.surface_rules,
    )
    assert touched == ("optional_track_activation_contract", "phase3_governance_bootstrap")


def test_phase3_contract_surface_rule_table_detects_design_bundle_contract_paths() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    touched = derive_touched_phase3_surface_ids(
        changed_paths=(
            "docs/spec/schemas/design_bundle_v1.json",
            "tests/conformance/test_design_bundle_loader_conformance.py",
        ),
        changed_lines_by_path={},
        rules=rule_table.surface_rules,
    )
    assert touched == ("design_bundle_contract",)


def test_phase3_contract_surface_rule_table_detects_design_bundle_schema_only_change() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    touched = derive_touched_phase3_surface_ids(
        changed_paths=("docs/spec/schemas/design_bundle_v1.json",),
        changed_lines_by_path={},
        rules=rule_table.surface_rules,
    )
    assert touched == ("design_bundle_contract",)


def test_phase3_contract_surface_rule_table_detects_packaged_exclusion_mirror_change() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    touched = derive_touched_phase3_surface_ids(
        changed_paths=("src/rfmna/parser/resources/p3_loader_temporary_exclusions.yaml",),
        changed_lines_by_path={},
        rules=rule_table.surface_rules,
    )
    assert touched == ("design_bundle_contract",)


def test_phase3_contract_surface_rule_table_ignores_frozen_only_loader_bridge_path() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    touched = derive_touched_phase3_surface_ids(
        changed_paths=("src/rfmna/cli/main.py",),
        changed_lines_by_path={},
        rules=rule_table.surface_rules,
    )
    assert touched == ()


def test_phase3_contract_surface_gate_passes_for_bootstrap_scope(tmp_path: Path) -> None:
    repo_root = _repo_root()
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=[
            "optional_track_activation_contract",
            "phase3_governance_bootstrap",
        ],
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "docs/dev/phase3_gate.md",
            "docs/dev/optional_track_activation.yaml",
            "src/rfmna/governance/phase3_gate.py",
            ".github/workflows/ci.yml",
        ),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "      - name: Phase 3 contract-surface governance sub-gate (blocking)",
                "      - name: Phase 3 gate status (informational)",
            )
        },
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    try:
        assert result.passed, result.errors
        assert result.touched_surface_ids == (
            "optional_track_activation_contract",
            "phase3_governance_bootstrap",
        )
    finally:
        change_surface_path.unlink(missing_ok=True)


def test_phase3_contract_surface_gate_accepts_docs_spec_schema_artifacts_for_design_bundle_surface(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["design_bundle_contract"],
        evidence={
            "policy_docs": ["docs/dev/design_bundle_contract.md"],
            "schema_artifacts": [
                "docs/dev/p3_loader_temporary_exclusions_schema_v1.json",
                "docs/spec/schemas/design_bundle_v1.json",
            ],
            "conformance_updates": ["tests/conformance/test_design_bundle_loader_conformance.py"],
            "ci_enforcement": [],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "src/rfmna/cli/design_loader.py",
            "tests/conformance/test_design_bundle_loader_conformance.py",
        ),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    try:
        assert result.passed, result.errors
        assert result.touched_surface_ids == ("design_bundle_contract",)
    finally:
        change_surface_path.unlink(missing_ok=True)


def test_phase3_contract_surface_gate_accepts_schema_only_design_bundle_surface_evidence_trigger(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["design_bundle_contract"],
        evidence={
            "policy_docs": ["docs/dev/design_bundle_contract.md"],
            "schema_artifacts": [
                "docs/dev/p3_loader_temporary_exclusions_schema_v1.json",
                "docs/spec/schemas/design_bundle_v1.json",
            ],
            "conformance_updates": ["tests/conformance/test_design_bundle_loader_conformance.py"],
            "ci_enforcement": [],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=("docs/spec/schemas/design_bundle_v1.json",),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    try:
        assert result.passed, result.errors
        assert result.touched_surface_ids == ("design_bundle_contract",)
    finally:
        change_surface_path.unlink(missing_ok=True)


def test_phase3_contract_surface_gate_accepts_actual_p3_01_path_mix(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["design_bundle_contract"],
        evidence={
            "policy_docs": ["docs/dev/design_bundle_contract.md"],
            "schema_artifacts": [
                "docs/dev/p3_loader_temporary_exclusions_schema_v1.json",
                "docs/spec/schemas/design_bundle_v1.json",
            ],
            "conformance_updates": ["tests/conformance/test_design_bundle_loader_conformance.py"],
            "ci_enforcement": [],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "docs/spec/schemas/design_bundle_v1.json",
            "src/rfmna/cli/main.py",
            "src/rfmna/parser/design_bundle.py",
            "tests/conformance/test_design_bundle_loader_conformance.py",
        ),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    try:
        assert result.passed, result.errors
        assert result.touched_surface_ids == ("design_bundle_contract",)
    finally:
        change_surface_path.unlink(missing_ok=True)


def test_phase3_rule_table_marks_design_bundle_schemas_as_canonical_requirements() -> None:
    repo_root = _repo_root()
    rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )

    design_bundle_requirement = rule_table.schema_requirements["docs/spec/schemas/design_bundle_v1.json"]
    exclusions_requirement = rule_table.schema_requirements[
        "docs/dev/p3_loader_temporary_exclusions_schema_v1.json"
    ]

    assert design_bundle_requirement.title == "rfmna design bundle v1"
    assert design_bundle_requirement.required_fields == (
        "schema",
        "schema_version",
        "design",
        "analysis",
    )
    assert design_bundle_requirement.required_properties == (
        "schema",
        "schema_version",
        "design",
        "analysis",
    )
    assert exclusions_requirement.title == "P3 loader temporary exclusions v1"
    assert exclusions_requirement.required_fields == ("schema_version", "exclusions")
    assert exclusions_requirement.required_properties == (
        "schema_version",
        "exclusions",
        "notes",
    )


def test_phase3_contract_surface_gate_fails_on_declared_vs_detected_mismatch(tmp_path: Path) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids="none",
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("declared_surface_ids mismatch" in error for error in result.errors)


def test_phase3_contract_surface_gate_ignores_stale_checked_in_declaration_on_unrelated_diff() -> None:
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("README.md",),
        changed_lines_by_path={},
    )
    assert result.passed, result.errors
    assert result.touched_surface_ids == ()


def test_phase3_contract_surface_gate_blocks_missing_required_evidence(tmp_path: Path) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["phase3_governance_bootstrap"],
        evidence={
            "policy_docs": [],
            "schema_artifacts": [],
            "conformance_updates": [],
            "ci_enforcement": [],
            "process_traceability": [],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("phase3 contract surface change requires evidence" in error for error in result.errors)


def test_phase3_contract_surface_gate_blocks_missing_required_exact_artifact_path(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["optional_track_activation_contract"],
        evidence={
            "policy_docs": ["docs/dev/phase3_gate.md"],
            "schema_artifacts": ["docs/dev/phase3_change_surface_schema_v1.json"],
            "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
            "ci_enforcement": [".github/workflows/ci.yml"],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/optional_track_activation_policy.md",),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("missing required artifact paths" in error for error in result.errors)


def test_phase3_contract_surface_gate_machine_validates_change_surface_even_when_declared_none(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids="none",
        evidence={
            "policy_docs": ["../../etc/passwd"],
            "schema_artifacts": [],
            "conformance_updates": [],
            "ci_enforcement": [],
            "process_traceability": [],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/README-placeholder.md",),
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("phase3_change_surface.evidence.policy_docs path must be under [docs/dev/]" in error for error in result.errors)


def test_phase3_contract_surface_gate_requires_valid_change_scope_authority(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["phase3_governance_bootstrap"],
        evidence={
            "policy_docs": ["docs/dev/phase3_change_surface_policy.md", "docs/dev/phase3_gate.md"],
            "schema_artifacts": ["docs/dev/phase3_change_surface_schema_v1.json"],
            "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
            "ci_enforcement": [".github/workflows/ci.yml"],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            change_scope_path="docs/dev/__missing_change_scope.yaml",
        ),
    )
    assert not result.passed
    assert any("phase3 cross-consistency requires valid change_scope authority" in error for error in result.errors)


def test_phase3_contract_surface_gate_blocks_frozen_overlap_on_same_changed_path(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["phase3_governance_bootstrap"],
        evidence={
            "policy_docs": ["docs/dev/phase3_change_surface_policy.md", "docs/dev/phase3_gate.md"],
            "schema_artifacts": ["docs/dev/phase3_change_surface_schema_v1.json"],
            "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
            "ci_enforcement": [".github/workflows/ci.yml"],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[12])
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "          echo \"RFMNA_PHASE3_GOV_BASE=$gov_base\" >> \"$GITHUB_ENV\"",
                "      OPENBLAS_NUM_THREADS: \"2\"",
            )
        },
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert any("overlaps frozen declaration authority" in error for error in result.errors)


def test_phase3_contract_surface_gate_uses_baseline_frozen_rule_table_for_overlap_detection(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_frozen_rules = _load_rule_table(
        repo_root / "docs/dev/frozen_change_governance_rules.yaml"
    ).rules
    payload = json.loads(
        (repo_root / "docs/dev/frozen_change_governance_rules.yaml").read_text(encoding="utf-8")
    )
    for rule in payload["frozen_artifact_rules"]:
        if rule["id"] == _FROZEN_ID_THREAD_DEFAULTS:
            rule["detection"] = [{"path_glob": ".envrc"}]
            break
    tampered_frozen_rule_table_path = _write_json(
        tmp_path / "frozen_change_governance_rules.yaml", payload
    )
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["phase3_governance_bootstrap"],
        evidence={
            "policy_docs": ["docs/dev/phase3_change_surface_policy.md", "docs/dev/phase3_gate.md"],
            "schema_artifacts": ["docs/dev/phase3_change_surface_schema_v1.json"],
            "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
            "ci_enforcement": [".github/workflows/ci.yml"],
            "process_traceability": ["docs/dev/phase3_process_traceability.md"],
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[12])
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "      OPENBLAS_NUM_THREADS: \"2\"",
                "      - name: Phase 3 contract-surface governance sub-gate (blocking)",
            )
        },
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            change_scope_path=str(change_scope_path),
            frozen_rule_table_path=str(tampered_frozen_rule_table_path),
        ),
        baseline_frozen_rules=baseline_frozen_rules,
    )
    assert not result.passed
    assert any("overlaps frozen declaration authority" in error for error in result.errors)


def test_phase3_contract_surface_rule_table_rejects_frozen_overlap(tmp_path: Path) -> None:
    payload = json.loads(
        (_repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"][0]["detection"] = [{"path_glob": "docs/spec/v4_contract.md"}]
    target = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    with pytest.raises(ValueError, match="overlaps frozen detection surface"):
        _load_phase3_rule_table(
            target,
            frozen_rules=_load_rule_table(
                _repo_root() / "docs/dev/frozen_change_governance_rules.yaml"
            ).rules,
        )


def test_phase3_contract_surface_rule_table_rejects_frozen_overlap_with_different_tokens(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"][0]["detection"] = [
        {
            "path_glob": "docs/spec/v4_contract.md",
            "line_tokens_any": ["Residual formula"],
        }
    ]
    target = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    with pytest.raises(ValueError, match="overlaps frozen detection surface"):
        _load_phase3_rule_table(
            target,
            frozen_rules=_load_rule_table(
                _repo_root() / "docs/dev/frozen_change_governance_rules.yaml"
            ).rules,
        )


def test_phase3_contract_surface_rule_table_rejects_frozen_overlap_via_touch_paths(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"][0]["touch_paths"] = ["src/rfmna/cli/main.py"]
    target = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    with pytest.raises(ValueError, match="touch_path overlaps frozen detection surface"):
        _load_phase3_rule_table(
            target,
            frozen_rules=_load_rule_table(
                _repo_root() / "docs/dev/frozen_change_governance_rules.yaml"
            ).rules,
        )


def test_phase3_rule_table_payload_allows_historical_baseline_schema_subset() -> None:
    payload = json.loads(
        (_repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["canonical_schema_requirements"] = payload["canonical_schema_requirements"][-2:]

    rule_table = phase3_gate_module._load_phase3_rule_table_payload(
        payload,
        parse_context=phase3_gate_module._RuleTableParseContext(
            source="HEAD:docs/dev/phase3_contract_surface_governance_rules.yaml",
            frozen_rules=None,
            required_canonical_schema_paths=None,
        ),
    )

    assert set(rule_table.schema_requirements) == {
        "docs/dev/phase3_change_surface_schema_v1.json",
        "docs/dev/optional_track_activation_schema_v1.json",
    }


def test_phase3_contract_surface_rule_table_rejects_glob_vs_glob_frozen_overlap(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"][0]["detection"] = [
        {"path_glob": "docs/spec/v4*"}
    ]
    target = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    frozen_rules = {
        1: FrozenRule(
            frozen_id=1,
            label="glob overlap regression",
            detection=(DetectionPattern(path_glob="docs/spec/*", line_tokens_any=()),),
        )
    }
    with pytest.raises(ValueError, match="overlaps frozen detection surface"):
        _load_phase3_rule_table(
            target,
            frozen_rules=frozen_rules,
        )


def test_phase3_anti_tamper_gate_blocks_missing_base_ref() -> None:
    result = evaluate_anti_tamper_gate(
        repo_root=_repo_root(),
        base_ref=None,
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_gate.md",),
    )
    assert not result.passed
    assert any("non-empty, non-zero base-ref" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_unresolvable_base_ref() -> None:
    result = evaluate_anti_tamper_gate(
        repo_root=_repo_root(),
        base_ref="refs/heads/does-not-exist",
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_gate.md",),
    )
    assert not result.passed
    assert any("unresolvable" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_ambiguous_base_ref_token() -> None:
    result = evaluate_anti_tamper_gate(
        repo_root=_repo_root(),
        base_ref="main",
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_gate.md",),
    )
    assert not result.passed
    assert any("unambiguous commit-ish" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_rule_table_policy_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    change_surface_policy_requirement = next(
        requirement
        for requirement in payload["canonical_policy_requirements"]
        if requirement["path"] == "docs/dev/phase3_change_surface_policy.md"
    )
    change_surface_policy_requirement["required_fragments"] = ["declared_surface_ids"]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref="HEAD",
        ),
    )
    result = evaluate_anti_tamper_gate(
        repo_root=repo_root,
        base_ref="HEAD",
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_contract_surface_governance_rules.yaml",),
        artifact_paths=_phase3_artifact_paths(phase3_rule_table_path=str(rule_table_path)),
    )
    assert not result.passed
    assert any("weakens canonical policy requirement" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_rule_table_required_artifact_path_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"][0]["required_artifact_paths"]["policy_docs"] = [
        "docs/dev/phase3_gate.md"
    ]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref="HEAD",
        ),
    )
    result = evaluate_anti_tamper_gate(
        repo_root=repo_root,
        base_ref="HEAD",
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_contract_surface_governance_rules.yaml",),
        artifact_paths=_phase3_artifact_paths(phase3_rule_table_path=str(rule_table_path)),
    )
    assert not result.passed
    assert any("weakens required artifact paths" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_rule_table_schema_required_field_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["canonical_schema_requirements"][0]["required_fields"] = ["schema_version"]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref="HEAD",
        ),
    )
    result = evaluate_anti_tamper_gate(
        repo_root=repo_root,
        base_ref="HEAD",
        head_ref="HEAD",
        changed_paths=("docs/dev/phase3_contract_surface_governance_rules.yaml",),
        artifact_paths=_phase3_artifact_paths(phase3_rule_table_path=str(rule_table_path)),
    )
    assert not result.passed
    assert any("weakens canonical schema required fields" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_activation_state_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    baseline_activation = phase3_gate_module.OptionalTrackActivationData(
        tracks={
            "p3_10_cccs_ccvs": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_10_cccs_ccvs",
                state="activated",
                usage_evidence_source=phase3_gate_module.OptionalTrackEvidenceSource(
                    evidence_type="design_bundle_fixture",
                    reference="tests/fixtures/cross_check/approved_hashes_v1.json",
                ),
                usage_evidence_date="2026-03-01",
                activation_rationale="baseline activated",
                impacted_frozen_ids=(1,),
                approval_record=phase3_gate_module.OptionalTrackApprovalRecord(
                    status="approved",
                    approved_by="phase3-governance",
                    decision_date="2026-03-01",
                    decision_ref="docs/dev/phase3_gate.md",
                ),
            ),
            "p3_11_mutual_inductance": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_11_mutual_inductance",
                state="deferred",
                usage_evidence_source=None,
                usage_evidence_date=None,
                activation_rationale="baseline deferred",
                impacted_frozen_ids=(),
                approval_record=None,
            ),
        }
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref="HEAD",
        ),
    )
    monkeypatch.setattr(phase3_gate_module, "_git_ref_contains_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation_from_git_ref",
        lambda *args, **kwargs: baseline_activation,
    )
    result = evaluate_anti_tamper_gate(
        repo_root=repo_root,
        base_ref="HEAD",
        head_ref="HEAD",
        changed_paths=("docs/dev/optional_track_activation.yaml",),
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
        ),
    )
    assert not result.passed
    assert any("relaxes active track state" in error for error in result.errors)


def test_phase3_anti_tamper_gate_blocks_active_track_record_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date="2026-03-02",
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": "2026-03-02",
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    baseline_activation = phase3_gate_module.OptionalTrackActivationData(
        tracks={
            "p3_10_cccs_ccvs": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_10_cccs_ccvs",
                state="activated",
                usage_evidence_source=phase3_gate_module.OptionalTrackEvidenceSource(
                    evidence_type="design_bundle_fixture",
                    reference="tests/fixtures/cross_check/approved_hashes_v1.json",
                ),
                usage_evidence_date="2026-03-01",
                activation_rationale="baseline activated",
                impacted_frozen_ids=(1,),
                approval_record=phase3_gate_module.OptionalTrackApprovalRecord(
                    status="approved",
                    approved_by="phase3-governance",
                    decision_date="2026-03-01",
                    decision_ref="docs/dev/phase3_gate.md",
                ),
            ),
            "p3_11_mutual_inductance": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_11_mutual_inductance",
                state="deferred",
                usage_evidence_source=None,
                usage_evidence_date=None,
                activation_rationale="baseline deferred",
                impacted_frozen_ids=(),
                approval_record=None,
            ),
        }
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref="HEAD",
        ),
    )
    monkeypatch.setattr(phase3_gate_module, "_git_ref_contains_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation_from_git_ref",
        lambda *args, **kwargs: baseline_activation,
    )
    result = evaluate_anti_tamper_gate(
        repo_root=repo_root,
        base_ref="HEAD",
        head_ref="HEAD",
        changed_paths=("docs/dev/optional_track_activation.yaml",),
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
        ),
    )
    assert not result.passed
    assert any("mutates active track usage_evidence_date" in error for error in result.errors)


def test_phase3_contract_surface_gate_uses_baseline_rules_to_block_tampering_attempt(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    bootstrap_rule = next(
        rule
        for rule in payload["phase3_contract_surface_rules"]
        if rule["surface_id"] == "phase3_governance_bootstrap"
    )
    bootstrap_rule["detection"] = [{"path_glob": "docs/dev/__tampered_bootstrap_rule.md"}]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    tampered_rule_table = _load_phase3_rule_table(rule_table_path, frozen_rules=None)
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids="none",
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    guarded = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "docs/dev/phase3_contract_surface_governance_rules.yaml",
            "src/rfmna/governance/phase3_gate.py",
        ),
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            phase3_rule_table_path=str(rule_table_path),
        ),
        baseline_rule_table=baseline_rule_table,
    )
    assert not guarded.passed
    assert any("declared_surface_ids mismatch" in error for error in guarded.errors)

    bypass = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "docs/dev/phase3_contract_surface_governance_rules.yaml",
            "src/rfmna/governance/phase3_gate.py",
        ),
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            phase3_rule_table_path=str(rule_table_path),
        ),
        baseline_rule_table=tampered_rule_table,
    )
    assert bypass.passed, bypass.errors


def test_phase3_contract_surface_gate_uses_baseline_policy_requirements(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    change_surface_policy_requirement = next(
        requirement
        for requirement in payload["canonical_policy_requirements"]
        if requirement["path"] == "docs/dev/phase3_change_surface_policy.md"
    )
    change_surface_policy_requirement["required_fragments"] = ["declared_surface_ids"]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    tampered_rule_table = _load_phase3_rule_table(rule_table_path, frozen_rules=None)
    tampered_policy_path = repo_root / "docs/dev/__tmp_tampered_phase3_change_surface_policy.md"
    tampered_policy_path.write_text("# Tampered\n\ndeclared_surface_ids\n", encoding="utf-8")
    try:
        change_surface_path = _write_phase3_change_surface(
            tmp_path,
            declared_surface_ids=["phase3_governance_bootstrap"],
            evidence=_PHASE3_SURFACE_EVIDENCE,
        )

        guarded = evaluate_contract_surface_governance_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            artifact_paths=_phase3_artifact_paths(
                phase3_change_surface_path=str(change_surface_path),
                phase3_change_surface_policy_path="docs/dev/__tmp_tampered_phase3_change_surface_policy.md",
                phase3_rule_table_path=str(rule_table_path),
            ),
            baseline_rule_table=baseline_rule_table,
        )
        assert not guarded.passed
        assert any(
            "policy document is missing required content fragment" in error
            for error in guarded.errors
        )

        bypass = evaluate_contract_surface_governance_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            artifact_paths=_phase3_artifact_paths(
                phase3_change_surface_path=str(change_surface_path),
                phase3_change_surface_policy_path="docs/dev/__tmp_tampered_phase3_change_surface_policy.md",
                phase3_rule_table_path=str(rule_table_path),
            ),
            baseline_rule_table=tampered_rule_table,
        )
        assert bypass.passed, bypass.errors
    finally:
        tampered_policy_path.unlink(missing_ok=True)


def test_phase3_contract_surface_gate_blocks_new_surface_added_only_in_current_rule_table(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["phase3_contract_surface_rules"].append(
        {
            "surface_id": "new_contract_surface",
            "label": "New contract surface for regression coverage",
            "detection": [{"path_glob": "docs/dev/new_contract.md"}],
            "required_evidence": [
                "policy_docs",
                "schema_artifacts",
                "conformance_updates",
                "ci_enforcement",
                "process_traceability",
            ],
            "required_artifact_paths": {
                "policy_docs": ["docs/dev/phase3_change_surface_policy.md"],
                "schema_artifacts": ["docs/dev/phase3_change_surface_schema_v1.json"],
                "conformance_updates": ["tests/conformance/test_phase3_gate_conformance.py"],
                "ci_enforcement": [".github/workflows/ci.yml"],
                "process_traceability": ["docs/dev/phase3_process_traceability.md"],
            },
        }
    )
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids=["phase3_governance_bootstrap"],
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "docs/dev/new_contract.md",
            "docs/dev/phase3_contract_surface_governance_rules.yaml",
        ),
        artifact_paths=_phase3_artifact_paths(
            phase3_change_surface_path=str(change_surface_path),
            phase3_rule_table_path=str(rule_table_path),
        ),
        baseline_rule_table=baseline_rule_table,
    )
    assert not result.passed
    assert any("declared_surface_ids mismatch" in error for error in result.errors)


def test_optional_track_gate_passes_when_inactive() -> None:
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_ignores_missing_activation_artifact_when_inactive(
    tmp_path: Path,
) -> None:
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(
                tmp_path / "missing_optional_track_activation.yaml"
            ),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_ignores_malformed_activation_artifact_when_inactive(
    tmp_path: Path,
) -> None:
    invalid_activation_path = _write_json(
        tmp_path / "optional_track_activation.yaml",
        {
            "schema_version": 1,
            "notes": "missing tracks",
        },
    )
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(invalid_activation_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_blocks_malformed_activation_artifact_when_artifact_is_touched(
    tmp_path: Path,
) -> None:
    invalid_activation_path = _write_json(
        tmp_path / "optional_track_activation.yaml",
        {
            "schema_version": 1,
            "notes": "missing tracks",
        },
    )
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/optional_track_activation.yaml",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(invalid_activation_path),
        ),
    )
    assert not result.passed
    assert any("optional_track_activation missing required keys: tracks" in error for error in result.errors)


def test_optional_track_gate_blocks_malformed_activation_artifact_when_track_is_active_at_base_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_activation_path = _write_json(
        tmp_path / "optional_track_activation.yaml",
        {
            "schema_version": 1,
            "notes": "missing tracks",
        },
    )
    baseline_activation = phase3_gate_module.OptionalTrackActivationData(
        tracks={
            "p3_10_cccs_ccvs": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_10_cccs_ccvs",
                state="activated",
                usage_evidence_source=phase3_gate_module.OptionalTrackEvidenceSource(
                    evidence_type="design_bundle_fixture",
                    reference="tests/fixtures/cross_check/approved_hashes_v1.json",
                ),
                usage_evidence_date="2026-03-01",
                activation_rationale="baseline activated",
                impacted_frozen_ids=(1,),
                approval_record=phase3_gate_module.OptionalTrackApprovalRecord(
                    status="approved",
                    approved_by="phase3-governance",
                    decision_date="2026-03-01",
                    decision_ref="docs/dev/phase3_gate.md",
                ),
            ),
            "p3_11_mutual_inductance": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_11_mutual_inductance",
                state="deferred",
                usage_evidence_source=None,
                usage_evidence_date=None,
                activation_rationale="baseline deferred",
                impacted_frozen_ids=(),
                approval_record=None,
            ),
        }
    )
    monkeypatch.setattr(phase3_gate_module, "_git_ref_contains_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation_from_git_ref",
        lambda *args, **kwargs: baseline_activation,
    )
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(invalid_activation_path),
        ),
    )
    assert not result.passed
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)
    assert any("optional_track_activation missing required keys: tracks" in error for error in result.errors)


def test_optional_track_gate_accepts_boundary_freshness_when_activated(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=(base_date - timedelta(days=90)).isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)


def test_optional_track_gate_requires_change_scope_match_for_activation_decision(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=base_date.isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert any("impacted_frozen_ids mismatch" in error for error in result.errors)


def test_optional_track_gate_rejects_stale_usage_evidence(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=(base_date - timedelta(days=91)).isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert any("usage evidence is stale" in error for error in result.errors)


def test_optional_track_gate_blocks_scope_touch_without_activation(tmp_path: Path) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/elements/cccs.py",),
        changed_lines_by_path={"src/rfmna/elements/cccs.py": ("helper refactor",)},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert any("active by scope detection" in error for error in result.errors)


def test_optional_track_gate_ignores_deactivation_attempt_without_head_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    baseline_activation = phase3_gate_module.OptionalTrackActivationData(
        tracks={
            "p3_10_cccs_ccvs": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_10_cccs_ccvs",
                state="activated",
                usage_evidence_source=phase3_gate_module.OptionalTrackEvidenceSource(
                    evidence_type="design_bundle_fixture",
                    reference="tests/fixtures/cross_check/approved_hashes_v1.json",
                ),
                usage_evidence_date="2026-03-01",
                activation_rationale="baseline activated",
                impacted_frozen_ids=(1,),
                approval_record=phase3_gate_module.OptionalTrackApprovalRecord(
                    status="approved",
                    approved_by="phase3-governance",
                    decision_date="2026-03-01",
                    decision_ref="docs/dev/phase3_gate.md",
                ),
            ),
            "p3_11_mutual_inductance": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_11_mutual_inductance",
                state="deferred",
                usage_evidence_source=None,
                usage_evidence_date=None,
                activation_rationale="baseline deferred",
                impacted_frozen_ids=(),
                approval_record=None,
            ),
        }
    )
    monkeypatch.setattr(phase3_gate_module, "_git_ref_contains_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation_from_git_ref",
        lambda *args, **kwargs: baseline_activation,
    )
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/optional_track_activation.yaml",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_blocks_shared_factory_change_with_reserved_token(
    tmp_path: Path,
) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/elements/factory.py",),
        changed_lines_by_path={"src/rfmna/elements/factory.py": ("register CCCS element",)},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)
    assert any("active by scope detection" in error for error in result.errors)


def test_optional_track_gate_blocks_uppercase_shared_factory_registry_token(
    tmp_path: Path,
) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/elements/factory.py",),
        changed_lines_by_path={
            "src/rfmna/elements/factory.py": (
                '            "CCCS": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_cccs),',
            )
        },
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)
    assert any("active by scope detection" in error for error in result.errors)


def test_optional_track_gate_ignores_shared_factory_change_without_reserved_token(
    tmp_path: Path,
) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/elements/factory.py",),
        changed_lines_by_path={"src/rfmna/elements/factory.py": ("refactor registration helpers",)},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_inherited_phase2_governance_negative_scope_mismatch_still_fails(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )
    assert not result.passed
    assert any("declared_frozen_ids mismatch" in error for error in result.errors)


def test_inherited_phase2_governance_negative_missing_evidence_still_fails(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[9])
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )
    assert not result.passed
    assert any("full governance evidence" in error for error in result.errors)


def test_inherited_phase2_category_bootstrap_negative_missing_collection_guard_still_fails(
    tmp_path: Path,
) -> None:
    _seed_minimal_category_bootstrap_repo(
        tmp_path,
        markers=("unit", "conformance", "property", "regression", "cross_check"),
        workflow_fragments=(
            "Phase 2 governance sub-gate (blocking)",
            "Phase 2 category bootstrap sub-gate (blocking)",
            "Audit AGENTS Phase 2 cross_check policy alignment",
            "uv run pytest -m unit --collect-only -q",
            "uv run pytest -m unit --junitxml=test-reports/unit.xml",
            "uv run pytest -m conformance --collect-only -q",
            "Tests (thread-controls conformance guard)",
            "test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults",
            "test_thread_controls_conformance.py::test_envrc_declares_deterministic_thread_defaults",
            "uv run pytest -m conformance --junitxml=test-reports/conformance.xml",
            "uv run pytest -m property --collect-only -q",
            "uv run pytest -m property --junitxml=test-reports/property.xml",
            "uv run pytest -m regression --junitxml=test-reports/regression.xml",
            "uv run pytest tests/regression -m regression -q",
            "uv run pytest -m cross_check --collect-only -q",
            "uv run pytest -m cross_check --junitxml=test-reports/cross_check.xml",
            "Upload calibration/regression/cross_check diagnostics (failure)",
        ),
    )
    result = evaluate_category_bootstrap_gate(repo_root=tmp_path)
    assert not result.passed
    assert any(
        "CI workflow missing required Phase 2 fragment: uv run pytest -m regression --collect-only -q"
        in error
        for error in result.errors
    )


def test_optional_track_gate_blocks_cross_check_scope_touch_without_activation(
    tmp_path: Path,
) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("tests/cross_check/test_mutual_inductance_cross_check.py",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert result.active_optional_track_ids == ("p3_11_mutual_inductance",)
    assert any("active by scope detection" in error for error in result.errors)


def test_optional_track_gate_ignores_unreserved_parser_changes(tmp_path: Path) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/parser/expressions.py",),
        changed_lines_by_path={"src/rfmna/parser/expressions.py": ("helper refactor",)},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_skips_change_scope_validation_when_inactive(tmp_path: Path) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    invalid_change_scope_path = _write_json(
        tmp_path / "change_scope.yaml",
        {
            "schema_version": 1,
            "declared_frozen_ids": "none",
        },
    )
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(invalid_change_scope_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ()


def test_optional_track_gate_does_not_require_change_scope_match_on_unrelated_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=base_date.isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    baseline_activation = phase3_gate_module.OptionalTrackActivationData(
        tracks={
            "p3_10_cccs_ccvs": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_10_cccs_ccvs",
                state="activated",
                usage_evidence_source=phase3_gate_module.OptionalTrackEvidenceSource(
                    evidence_type="design_bundle_fixture",
                    reference="tests/fixtures/cross_check/approved_hashes_v1.json",
                ),
                usage_evidence_date=base_date.isoformat(),
                activation_rationale="baseline activated",
                impacted_frozen_ids=(1,),
                approval_record=phase3_gate_module.OptionalTrackApprovalRecord(
                    status="approved",
                    approved_by="phase3-governance",
                    decision_date=base_date.isoformat(),
                    decision_ref="docs/dev/phase3_gate.md",
                ),
            ),
            "p3_11_mutual_inductance": phase3_gate_module.OptionalTrackActivationRecord(
                track_id="p3_11_mutual_inductance",
                state="deferred",
                usage_evidence_source=None,
                usage_evidence_date=None,
                activation_rationale="baseline deferred",
                impacted_frozen_ids=(),
                approval_record=None,
            ),
        }
    )
    monkeypatch.setattr(phase3_gate_module, "_git_ref_contains_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation_from_git_ref",
        lambda *args, **kwargs: baseline_activation,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert result.passed, result.errors
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)
    assert not any("impacted_frozen_ids mismatch" in error for error in result.errors)


def test_optional_track_gate_rechecks_historical_freshness_on_scope_touch(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=(base_date - timedelta(days=91)).isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("src/rfmna/elements/cccs.py",),
        changed_lines_by_path={"src/rfmna/elements/cccs.py": ("maintenance change",)},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert result.active_optional_track_ids == ("p3_10_cccs_ccvs",)
    assert any("usage evidence is stale" in error for error in result.errors)


def test_optional_track_gate_blocks_new_touch_path_added_only_in_current_rule_table(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["optional_track_rules"][0]["touch_detection"].append(
        {"path_glob": "docs/dev/new_optional_track_scope.md"}
    )
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="deferred",
        p3_10_evidence=None,
        p3_10_usage_date=None,
        p3_10_impacted_frozen_ids=[],
        p3_10_approval=None,
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids="none")
    result = evaluate_optional_track_gate(
        repo_root=repo_root,
        changed_paths=("docs/dev/new_optional_track_scope.md",),
        changed_lines_by_path={},
        base_ref="HEAD",
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
            phase3_rule_table_path=str(rule_table_path),
        ),
        baseline_rule_table=baseline_rule_table,
    )
    assert not result.passed
    assert any("active by scope detection" in error for error in result.errors)


def test_phase3_contract_surface_gate_blocks_invalid_schema_artifact_json(tmp_path: Path) -> None:
    repo_root = _repo_root()
    invalid_schema_path = repo_root / "docs/dev/__tmp_invalid_phase3_schema.json"
    invalid_schema_path.write_text("{\n", encoding="utf-8")
    try:
        change_surface_path = _write_phase3_change_surface(
            tmp_path,
            declared_surface_ids=["phase3_governance_bootstrap"],
            evidence={
                **_PHASE3_SURFACE_EVIDENCE,
                "schema_artifacts": ["docs/dev/__tmp_invalid_phase3_schema.json"],
            },
        )
        result = evaluate_contract_surface_governance_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
        )
        assert not result.passed
        assert any("must contain valid JSON" in error for error in result.errors)
    finally:
        invalid_schema_path.unlink(missing_ok=True)


def test_phase3_contract_surface_gate_blocks_relaxed_canonical_required_fields(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    relaxed_schema_path = repo_root / "docs/dev/__tmp_relaxed_phase3_schema.json"
    relaxed_schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "docs/dev/__tmp_relaxed_phase3_schema.json",
                "title": "Phase 3 Change Surface Declaration",
                "type": "object",
                "required": ["schema_version"],
                "additionalProperties": False,
                "properties": {
                    "schema_version": {"const": 1},
                    "declared_surface_ids": {"type": "array"},
                    "evidence": {"type": "object"},
                    "notes": {"type": "string"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        change_surface_path = _write_phase3_change_surface(
            tmp_path,
            declared_surface_ids=["phase3_governance_bootstrap"],
            evidence=_PHASE3_SURFACE_EVIDENCE,
        )
        result = evaluate_contract_surface_governance_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            artifact_paths=_phase3_artifact_paths(
                phase3_change_surface_path=str(change_surface_path),
                phase3_change_surface_schema_path="docs/dev/__tmp_relaxed_phase3_schema.json",
            ),
        )
        assert not result.passed
        assert any("missing canonical required fields" in error for error in result.errors)
    finally:
        relaxed_schema_path.unlink(missing_ok=True)


def test_optional_track_gate_blocks_missing_base_ref_when_active(tmp_path: Path) -> None:
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date="2026-03-01",
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": "2026-03-01",
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    result = evaluate_optional_track_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase3_gate.md",),
        changed_lines_by_path={},
        base_ref=None,
        head_ref="HEAD",
        artifact_paths=_phase3_artifact_paths(
            optional_track_activation_path=str(activation_path),
            change_scope_path=str(change_scope_path),
        ),
    )
    assert not result.passed
    assert any("non-empty, non-zero base-ref" in error for error in result.errors)


def test_optional_track_gate_blocks_invalid_canonical_schema_json(tmp_path: Path) -> None:
    repo_root = _repo_root()
    invalid_schema_path = repo_root / "docs/dev/__tmp_invalid_optional_track_schema.json"
    invalid_schema_path.write_text("{\n", encoding="utf-8")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date="2026-03-01",
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": "2026-03-01",
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    try:
        result = evaluate_optional_track_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            changed_lines_by_path={},
            base_ref="HEAD",
            head_ref="HEAD",
            artifact_paths=_phase3_artifact_paths(
                optional_track_activation_path=str(activation_path),
                optional_track_activation_schema_path="docs/dev/__tmp_invalid_optional_track_schema.json",
                change_scope_path=str(change_scope_path),
            ),
        )
        assert not result.passed
        assert any("must contain valid JSON" in error for error in result.errors)
    finally:
        invalid_schema_path.unlink(missing_ok=True)


def test_optional_track_gate_blocks_relaxed_canonical_required_fields(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    relaxed_schema_path = repo_root / "docs/dev/__tmp_relaxed_optional_track_schema.json"
    relaxed_schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "docs/dev/__tmp_relaxed_optional_track_schema.json",
                "title": "Optional Track Activation Evidence",
                "type": "object",
                "required": ["schema_version"],
                "additionalProperties": False,
                "properties": {
                    "schema_version": {"const": 1},
                    "tracks": {"type": "array"},
                    "notes": {"type": "string"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date="2026-03-01",
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": "2026-03-01",
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    try:
        result = evaluate_optional_track_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            changed_lines_by_path={},
            base_ref="HEAD",
            head_ref="HEAD",
            artifact_paths=_phase3_artifact_paths(
                optional_track_activation_path=str(activation_path),
                optional_track_activation_schema_path="docs/dev/__tmp_relaxed_optional_track_schema.json",
                change_scope_path=str(change_scope_path),
            ),
        )
        assert not result.passed
        assert any("missing canonical required fields" in error for error in result.errors)
    finally:
        relaxed_schema_path.unlink(missing_ok=True)


def test_optional_track_gate_blocks_invalid_canonical_policy_document(tmp_path: Path) -> None:
    repo_root = _repo_root()
    invalid_policy_path = repo_root / "docs/dev/__tmp_invalid_optional_track_policy.md"
    invalid_policy_path.write_text("# Invalid\n", encoding="utf-8")
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date="2026-03-01",
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": "2026-03-01",
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    try:
        result = evaluate_optional_track_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            changed_lines_by_path={},
            base_ref="HEAD",
            head_ref="HEAD",
            artifact_paths=_phase3_artifact_paths(
                optional_track_activation_path=str(activation_path),
                optional_track_policy_path="docs/dev/__tmp_invalid_optional_track_policy.md",
                change_scope_path=str(change_scope_path),
            ),
        )
        assert not result.passed
        assert any("policy document is missing required content fragment" in error for error in result.errors)
    finally:
        invalid_policy_path.unlink(missing_ok=True)


def test_optional_track_gate_uses_baseline_policy_requirements(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref="HEAD")
    baseline_rule_table = _load_phase3_rule_table(
        repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    payload = json.loads(
        (repo_root / "docs/dev/phase3_contract_surface_governance_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    optional_track_policy_requirement = next(
        requirement
        for requirement in payload["canonical_policy_requirements"]
        if requirement["path"] == "docs/dev/optional_track_activation_policy.md"
    )
    optional_track_policy_requirement["required_fragments"] = [
        "Freshness window: `90` days inclusive"
    ]
    rule_table_path = _write_json(tmp_path / "phase3_contract_surface_governance_rules.yaml", payload)
    tampered_rule_table = _load_phase3_rule_table(rule_table_path, frozen_rules=None)
    activation_path = _write_optional_track_activation(
        tmp_path,
        p3_10_state="activated",
        p3_10_evidence={
            "evidence_type": "design_bundle_fixture",
            "reference": "tests/fixtures/cross_check/approved_hashes_v1.json",
        },
        p3_10_usage_date=base_date.isoformat(),
        p3_10_impacted_frozen_ids=[1],
        p3_10_approval={
            "status": "approved",
            "approved_by": "phase3-governance",
            "decision_date": base_date.isoformat(),
            "decision_ref": "docs/dev/phase3_gate.md",
        },
    )
    change_scope_path = _write_change_scope(tmp_path, declared_frozen_ids=[1])
    tampered_policy_path = repo_root / "docs/dev/__tmp_tampered_optional_track_activation_policy.md"
    tampered_policy_path.write_text(
        "# Tampered\n\nFreshness window: `90` days inclusive\n",
        encoding="utf-8",
    )
    try:
        guarded = evaluate_optional_track_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            changed_lines_by_path={},
            base_ref="HEAD",
            head_ref="HEAD",
            artifact_paths=_phase3_artifact_paths(
                optional_track_activation_path=str(activation_path),
                optional_track_policy_path="docs/dev/__tmp_tampered_optional_track_activation_policy.md",
                change_scope_path=str(change_scope_path),
                phase3_rule_table_path=str(rule_table_path),
            ),
            baseline_rule_table=baseline_rule_table,
        )
        assert not guarded.passed
        assert any(
            "policy document is missing required content fragment" in error
            for error in guarded.errors
        )

        bypass = evaluate_optional_track_gate(
            repo_root=repo_root,
            changed_paths=("docs/dev/phase3_gate.md",),
            changed_lines_by_path={},
            base_ref="HEAD",
            head_ref="HEAD",
            artifact_paths=_phase3_artifact_paths(
                optional_track_activation_path=str(activation_path),
                optional_track_policy_path="docs/dev/__tmp_tampered_optional_track_activation_policy.md",
                change_scope_path=str(change_scope_path),
                phase3_rule_table_path=str(rule_table_path),
            ),
            baseline_rule_table=tampered_rule_table,
        )
        assert bypass.passed, bypass.errors
    finally:
        tampered_policy_path.unlink(missing_ok=True)


def test_phase3_gate_main_requires_base_ref_for_contract_surface_even_with_changed_paths_file(
    tmp_path: Path,
) -> None:
    changed_paths_file = tmp_path / "changed_paths.txt"
    changed_paths_file.write_text(
        "docs/dev/phase3_gate.md\ndocs/dev/optional_track_activation.yaml\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--repo-root",
            str(_repo_root()),
            "--sub-gate",
            "contract-surface",
            "--changed-paths-file",
            str(changed_paths_file),
        ]
    )
    assert exit_code == 1


def test_phase3_gate_main_contract_surface_ignores_optional_track_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_rule_table = _load_phase3_rule_table(
        _repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    changed_paths_file = tmp_path / "changed_paths.txt"
    changed_paths_file.write_text(
        "docs/dev/phase3_gate.md\ndocs/dev/optional_track_activation.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_optional_track_activation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected optional-track load")),
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=True,
            baseline_ref="HEAD",
        ),
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_change_surface",
        lambda *args, **kwargs: (
            ("optional_track_activation_contract", "phase3_governance_bootstrap"),
            phase3_gate_module.Phase3ChangeSurfaceEvidence(
                policy_docs=tuple(_PHASE3_SURFACE_EVIDENCE["policy_docs"]),
                schema_artifacts=tuple(_PHASE3_SURFACE_EVIDENCE["schema_artifacts"]),
                conformance_updates=tuple(_PHASE3_SURFACE_EVIDENCE["conformance_updates"]),
                ci_enforcement=tuple(_PHASE3_SURFACE_EVIDENCE["ci_enforcement"]),
                process_traceability=tuple(_PHASE3_SURFACE_EVIDENCE["process_traceability"]),
            ),
        ),
    )
    exit_code = main(
        [
            "--repo-root",
            str(_repo_root()),
            "--sub-gate",
            "contract-surface",
            "--base-ref",
            "HEAD",
            "--head-ref",
            "HEAD",
            "--changed-paths-file",
            str(changed_paths_file),
        ]
    )
    assert exit_code == 0


def test_phase3_gate_main_anti_tamper_ignores_unrelated_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_rule_table = _load_phase3_rule_table(
        _repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    changed_paths_file = tmp_path / "changed_paths.txt"
    changed_paths_file.write_text("docs/dev/phase3_gate.md\n", encoding="utf-8")
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_change_surface",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected change-surface load")),
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_change_scope",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected change-scope load")),
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=True,
            baseline_ref="HEAD",
        ),
    )
    exit_code = main(
        [
            "--repo-root",
            str(_repo_root()),
            "--sub-gate",
            "anti-tamper",
            "--base-ref",
            "HEAD",
            "--head-ref",
            "HEAD",
            "--changed-paths-file",
            str(changed_paths_file),
        ]
    )
    assert exit_code == 0


def test_phase3_gate_main_optional_track_ignores_phase3_change_surface_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_rule_table = _load_phase3_rule_table(
        _repo_root() / "docs/dev/phase3_contract_surface_governance_rules.yaml",
        frozen_rules=None,
    )
    changed_paths_file = tmp_path / "changed_paths.txt"
    changed_paths_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_change_surface",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected change-surface load")),
    )
    monkeypatch.setattr(
        phase3_gate_module,
        "_load_phase3_baseline_artifacts",
        lambda *args, **kwargs: phase3_gate_module.Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=True,
            baseline_ref="HEAD",
        ),
    )
    exit_code = main(
        [
            "--repo-root",
            str(_repo_root()),
            "--sub-gate",
            "optional-track",
            "--base-ref",
            "HEAD",
            "--head-ref",
            "HEAD",
            "--changed-paths-file",
            str(changed_paths_file),
        ]
    )
    assert exit_code == 0


def test_ci_workflow_contains_phase3_subgates_and_status_artifact() -> None:
    named_steps = _ci_named_steps()
    required_steps = {
        "Resolve Phase 3 governance diff range",
        "Phase 3 contract-surface governance sub-gate (blocking)",
        "Phase 3 anti-tamper sub-gate (blocking)",
        "Phase 3 optional-track activation sub-gate (blocking)",
        "Phase 3 gate status (informational)",
        "Upload Phase 3 gate status (informational)",
    }
    assert required_steps.issubset(named_steps)

    resolve_run = named_steps["Resolve Phase 3 governance diff range"][1].get("run")
    assert isinstance(resolve_run, str)
    assert "RFMNA_PHASE3_GOV_BASE" in resolve_run
    assert "non-empty base ref" in resolve_run
    assert "git rev-parse --verify --quiet \"${gov_base}^{commit}\"" in resolve_run
    assert "git rev-parse \"${{ github.sha }}^\"" not in resolve_run

    for step_name in (
        "Phase 3 contract-surface governance sub-gate (blocking)",
        "Phase 3 anti-tamper sub-gate (blocking)",
        "Phase 3 optional-track activation sub-gate (blocking)",
    ):
        run = named_steps[step_name][1].get("run")
        assert isinstance(run, str)
        assert "python -m rfmna.governance.phase3_gate" in run
        assert "--base-ref \"$RFMNA_PHASE3_GOV_BASE\"" in run
        assert "--head-ref \"$RFMNA_PHASE3_GOV_HEAD\"" in run

    status_run = named_steps["Phase 3 gate status (informational)"][1].get("run")
    assert isinstance(status_run, str)
    assert "phase3_gate_status.txt" in status_run
    assert "docs/dev/phase3_gate.md" in status_run
    assert "docs/dev/phase3_process_traceability.md" in status_run

    upload_step = named_steps["Upload Phase 3 gate status (informational)"][1]
    assert upload_step.get("if") == "${{ always() }}"
    with_block = upload_step.get("with")
    assert isinstance(with_block, dict)
    upload_path = with_block.get("path")
    assert isinstance(upload_path, str)
    assert "phase3_gate_status.txt" in upload_path
    assert "phase3_contract_surface_subgate_status.txt" in upload_path
    assert "phase3_anti_tamper_subgate_status.txt" in upload_path
    assert "phase3_optional_track_subgate_status.txt" in upload_path


def test_phase3_contract_surface_gate_detects_ci_command_body_change_without_step_name_change(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids="none",
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "            --base-ref \"$RFMNA_PHASE3_GOV_BASE\" \\",
                "            --head-ref \"$RFMNA_PHASE3_GOV_HEAD\" \\",
            )
        },
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("declared_surface_ids mismatch" in error for error in result.errors)


def test_phase3_contract_surface_gate_detects_resolve_step_body_change(
    tmp_path: Path,
) -> None:
    change_surface_path = _write_phase3_change_surface(
        tmp_path,
        declared_surface_ids="none",
        evidence=_PHASE3_SURFACE_EVIDENCE,
    )
    result = evaluate_contract_surface_governance_gate(
        repo_root=_repo_root(),
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "          if [ -z \"$gov_base\" ] || [ \"$gov_base\" = \"$ZERO_SHA\" ]; then",
                "            echo \"Phase 3 governance requires a non-empty base ref.\"",
                "          if ! git rev-parse --verify --quiet \"${gov_base}^{commit}\" >/dev/null; then",
            )
        },
        artifact_paths=_phase3_artifact_paths(phase3_change_surface_path=str(change_surface_path)),
    )
    assert not result.passed
    assert any("declared_surface_ids mismatch" in error for error in result.errors)


def test_phase3_gate_docs_and_traceability_are_explicit() -> None:
    gate = _read_repo_text("docs/dev/phase3_gate.md")
    traceability = _read_repo_text("docs/dev/phase3_process_traceability.md")
    for fragment in (
        "docs/dev/phase3_change_surface.yaml",
        "docs/dev/phase3_contract_surface_governance_rules.yaml",
        "docs/dev/optional_track_activation.yaml",
        "python -m rfmna.governance.phase3_gate --sub-gate contract-surface",
        "python -m rfmna.governance.phase3_gate --sub-gate anti-tamper",
        "python -m rfmna.governance.phase3_gate --sub-gate optional-track",
    ):
        assert fragment in gate
    for fragment in (
        "`base_ref_contract`",
        "`phase3_change_surface_declaration`",
        "`optional_track_default_state`",
        "src/rfmna/governance/phase3_gate.py",
        ".github/workflows/ci.yml",
    ):
        assert fragment in traceability

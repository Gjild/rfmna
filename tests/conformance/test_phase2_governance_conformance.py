from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path

import pytest

from rfmna.governance.phase2_gate import (
    GovernanceArtifactPaths,
    GovernanceBaselineArtifacts,
    GovernanceRuleTableData,
    ToleranceClassificationData,
    _changed_lines_from_git,
    _changed_paths_from_git,
    _load_rule_table,
    _load_tolerance_classification,
    _parse_changed_lines_file,
    derive_touched_frozen_ids,
    evaluate_governance_gate,
)

pytestmark = pytest.mark.conformance

_FULL_EVIDENCE = {
    "semver_bump": {"from_version": "0.1.0", "to_version": "0.2.0"},
    "decision_records": ["docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md"],
    "conformance_updates": ["tests/conformance/test_phase2_governance_conformance.py"],
    "migration_notes": ["docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md"],
    "reproducibility_impact_statement_path": "docs/dev/phase2_process_traceability.md",
}

_FROZEN_RULE_SAMPLE_PATHS = {
    1: "src/rfmna/elements/passive.py",
    2: "docs/spec/stamp_appendix_v4_0_0.md",
    3: "docs/spec/port_wave_conventions_v4_0_0.md",
    4: "src/rfmna/solver/solve.py",
    5: "docs/spec/thresholds_v4_0_0.yaml",
    6: "src/rfmna/solver/fallback.py",
    7: "src/rfmna/ir/serialize.py",
    8: "src/rfmna/sweep_engine/frequency_grid.py",
    9: "src/rfmna/cli/main.py",
    10: "src/rfmna/sweep_engine/types.py",
    11: "src/rfmna/sweep_engine/run.py",
    12: ".envrc",
}
_FROZEN_ID_THREAD_DEFAULTS = 12
_FROZEN_ID_CLI_EXIT = 9
_FROZEN_ID_THRESHOLD_STATUS_BANDS = 5
_REGRESSION_TOLERANCE_SOURCE = "docs/dev/tolerances/regression_baseline_v1.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_file_paths() -> tuple[str, ...]:
    root = _repo_root()
    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git/" not in path.as_posix()
    ]
    return tuple(sorted(files))


def _write_change_scope(
    tmp_path: Path,
    *,
    declared_frozen_ids: str | list[int],
    evidence: dict[str, object],
) -> Path:
    payload = {
        "schema_version": 1,
        "declared_frozen_ids": declared_frozen_ids,
        "evidence": evidence,
    }
    target = tmp_path / "change_scope.yaml"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _write_tolerance_classification(
    tmp_path: Path,
    *,
    thresholds_classification: str,
    calibration_seed_classification: str = "calibration_only",
    merge_gating_source: str = "docs/spec/thresholds_v4_0_0.yaml",
) -> Path:
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "path": "docs/spec/thresholds_v4_0_0.yaml",
                "classification": thresholds_classification,
                "notes": "test override",
            },
            {
                "path": "docs/dev/tolerances/calibration_seed_v1.yaml",
                "classification": calibration_seed_classification,
                "notes": "test override",
            },
        ],
        "merge_gating_tolerance_sources": [merge_gating_source],
        "promotion_policy_note": "test override",
    }
    target = tmp_path / "threshold_tolerance_classification.yaml"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


@pytest.mark.parametrize("frozen_id,sample_path", sorted(_FROZEN_RULE_SAMPLE_PATHS.items()))
def test_frozen_rule_table_detects_each_id_mapping_path(frozen_id: int, sample_path: str) -> None:
    repo_root = _repo_root()
    rules = _load_rule_table(repo_root / "docs/dev/frozen_change_governance_rules.yaml").rules

    touched = derive_touched_frozen_ids(
        changed_paths=(sample_path,),
        changed_lines_by_path={},
        rules=rules,
    )

    assert frozen_id in touched


@pytest.mark.parametrize("frozen_id,sample_path", sorted(_FROZEN_RULE_SAMPLE_PATHS.items()))
def test_frozen_rule_table_negative_case_per_mapping_path(frozen_id: int, sample_path: str) -> None:
    repo_root = _repo_root()
    rules = _load_rule_table(repo_root / "docs/dev/frozen_change_governance_rules.yaml").rules

    touched = derive_touched_frozen_ids(
        changed_paths=(f"{sample_path}.nonmatching",),
        changed_lines_by_path={},
        rules=rules,
    )

    assert frozen_id not in touched


def test_frozen_rule_table_each_detection_entry_path_is_exercised() -> None:
    root = _repo_root()
    repo_files = _repo_file_paths()
    rules = _load_rule_table(root / "docs/dev/frozen_change_governance_rules.yaml").rules

    for frozen_id, rule in rules.items():
        for detection in rule.detection:
            matches = [path for path in repo_files if fnmatch(path, detection.path_glob)]
            assert matches, f"no repo files match detection path_glob: {detection.path_glob}"
            path = matches[0]
            changed_lines = (
                {path: (f"  {detection.line_tokens_any[0]}  ",)}
                if detection.line_tokens_any
                else {}
            )
            touched = derive_touched_frozen_ids(
                changed_paths=(path,),
                changed_lines_by_path=changed_lines,
                rules=rules,
            )
            assert frozen_id in touched, (
                "detection path did not trigger mapped frozen id: "
                f"id={frozen_id} path_glob={detection.path_glob} chosen={path}"
            )


def test_id12_ci_workflow_detection_requires_thread_env_token_match() -> None:
    repo_root = _repo_root()
    rules = _load_rule_table(repo_root / "docs/dev/frozen_change_governance_rules.yaml").rules

    touched = derive_touched_frozen_ids(
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                "      - name: Tests (cross_check)",
                "        run: uv run pytest -m cross_check",
            )
        },
        rules=rules,
    )

    assert _FROZEN_ID_THREAD_DEFAULTS not in touched


def test_id12_detection_accepts_yaml_spacing_variants_for_thread_defaults() -> None:
    repo_root = _repo_root()
    rules = _load_rule_table(repo_root / "docs/dev/frozen_change_governance_rules.yaml").rules
    touched = derive_touched_frozen_ids(
        changed_paths=(".github/workflows/ci.yml",),
        changed_lines_by_path={
            ".github/workflows/ci.yml": (
                '      OPENBLAS_NUM_THREADS : "2"',
                '      MKL_NUM_THREADS : "2"',
            )
        },
        rules=rules,
    )
    assert _FROZEN_ID_THREAD_DEFAULTS in touched


def test_governance_gate_passes_for_non_frozen_change_scope_none(tmp_path: Path) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids="none",
        evidence={
            "semver_bump": None,
            "decision_records": [],
            "conformance_updates": [],
            "migration_notes": [],
            "reproducibility_impact_statement_path": None,
        },
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )
    assert result.passed, result.errors


def test_governance_gate_fails_on_declared_vs_detected_scope_mismatch() -> None:
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
    )
    assert not result.passed
    assert any("declared_frozen_ids mismatch" in error for error in result.errors)


def test_governance_gate_blocks_missing_evidence_for_frozen_scope(tmp_path: Path) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids=[9],
        evidence={
            "semver_bump": None,
            "decision_records": [],
            "conformance_updates": [],
            "migration_notes": [],
            "reproducibility_impact_statement_path": None,
        },
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any("full governance evidence" in error for error in result.errors)


def test_governance_gate_accepts_full_evidence_for_frozen_scope(tmp_path: Path) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids=[9],
        evidence=_FULL_EVIDENCE,
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert result.passed, result.errors


def test_governance_gate_rejects_absolute_evidence_paths_for_frozen_scope(tmp_path: Path) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids=[9],
        evidence={
            "semver_bump": {"from_version": "0.1.0", "to_version": "0.2.0"},
            "decision_records": ["/etc/hosts"],
            "conformance_updates": ["/etc/hosts"],
            "migration_notes": ["/etc/hosts"],
            "reproducibility_impact_statement_path": "/etc/hosts",
        },
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any("path must be repo-relative" in error for error in result.errors)


def test_governance_gate_rejects_wrong_prefix_evidence_paths_for_frozen_scope(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids=[9],
        evidence={
            "semver_bump": {"from_version": "0.1.0", "to_version": "0.2.0"},
            "decision_records": ["README.md"],
            "conformance_updates": ["src/rfmna/__init__.py"],
            "migration_notes": ["pyproject.toml"],
            "reproducibility_impact_statement_path": "src/rfmna/__init__.py",
        },
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("src/rfmna/cli/main.py",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any("decision_records path must be under" in error for error in result.errors)
    assert any("conformance_updates path must be under" in error for error in result.errors)
    assert any("migration_notes path must be under" in error for error in result.errors)
    assert any(
        "reproducibility_impact_statement_path path must be under" in error
        for error in result.errors
    )


def test_governance_gate_blocks_calibration_only_merge_gating_source(tmp_path: Path) -> None:
    classification_path = _write_tolerance_classification(
        tmp_path,
        thresholds_classification="calibration_only",
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=(),
        artifact_paths=GovernanceArtifactPaths(
            tolerance_classification_path=str(classification_path)
        ),
    )

    assert not result.passed
    assert any("cannot be calibration_only" in error for error in result.errors)
    assert any("must be normative_gating" in error for error in result.errors)


def test_baseline_classification_marks_regression_tolerance_as_normative_merge_gating() -> None:
    classification = _load_tolerance_classification(
        _repo_root() / "docs/dev/threshold_tolerance_classification.yaml"
    )
    assert classification.entries[_REGRESSION_TOLERANCE_SOURCE] == "normative_gating"
    assert _REGRESSION_TOLERANCE_SOURCE in classification.merge_gating_sources
    assert (
        classification.entries["docs/dev/tolerances/calibration_seed_v1.yaml"] == "calibration_only"
    )


def test_governance_gate_requires_change_scope_artifact() -> None:
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=(),
        artifact_paths=GovernanceArtifactPaths(
            change_scope_path="docs/dev/missing_change_scope.yaml"
        ),
    )
    assert not result.passed
    assert any("required artifact is missing" in error for error in result.errors)


def test_governance_gate_rejects_schema_invalid_change_scope_even_without_frozen_touches(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids="none",
        evidence={},
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any("change_scope.evidence missing required keys" in error for error in result.errors)


def test_governance_gate_rejects_schema_invalid_semver_empty_strings_even_without_frozen_touches(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids="none",
        evidence={
            "semver_bump": {"from_version": "", "to_version": ""},
            "decision_records": [],
            "conformance_updates": [],
            "migration_notes": [],
            "reproducibility_impact_statement_path": None,
        },
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any(
        "semver_bump.from_version/to_version must be non-empty strings" in error
        for error in result.errors
    )


def test_governance_gate_rejects_schema_invalid_notes_type(tmp_path: Path) -> None:
    change_scope_path = tmp_path / "change_scope.yaml"
    change_scope_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "declared_frozen_ids": "none",
                "evidence": {
                    "semver_bump": None,
                    "decision_records": [],
                    "conformance_updates": [],
                    "migration_notes": [],
                    "reproducibility_impact_statement_path": None,
                },
                "notes": 123,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert not result.passed
    assert any(
        "change_scope.notes must be a string when present" in error for error in result.errors
    )


def test_governance_gate_blocks_any_touched_normative_gating_tolerance_source_without_evidence(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids="none",
        evidence={
            "semver_bump": None,
            "decision_records": [],
            "conformance_updates": [],
            "migration_notes": [],
            "reproducibility_impact_statement_path": None,
        },
    )
    classification_path = _write_tolerance_classification(
        tmp_path,
        thresholds_classification="normative_gating",
        calibration_seed_classification="normative_gating",
        merge_gating_source="docs/dev/tolerances/calibration_seed_v1.yaml",
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/tolerances/calibration_seed_v1.yaml",),
        artifact_paths=GovernanceArtifactPaths(
            change_scope_path=str(change_scope_path),
            tolerance_classification_path=str(classification_path),
        ),
    )

    assert not result.passed
    assert any(
        "touching normative_gating tolerance sources is blocked without full evidence" in error
        for error in result.errors
    )


def test_rule_table_declares_required_evidence_mapping_for_each_frozen_id() -> None:
    required_evidence_by_id = _load_rule_table(
        _repo_root() / "docs/dev/frozen_change_governance_rules.yaml"
    ).required_evidence_by_frozen_id
    assert set(required_evidence_by_id) == set(range(1, 13))
    for frozen_id, required in required_evidence_by_id.items():
        assert "semver_bump" in required
        assert "decision_records" in required
        assert "conformance_updates" in required
        assert "migration_notes" in required
        assert "reproducibility_impact_statement_path" in required
        assert required == tuple(sorted(set(required))), frozen_id


def test_rule_table_requires_frozen_id_5_sources_in_threshold_source_list(tmp_path: Path) -> None:
    source_path = _repo_root() / "docs/dev/frozen_change_governance_rules.yaml"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["frozen_threshold_status_band_sources"] = []

    target = tmp_path / "frozen_change_governance_rules.yaml"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="must include all frozen rule 5 sources"):
        _load_rule_table(target)


def test_governance_gate_accepts_true_yaml_syntax_for_change_scope_artifact(tmp_path: Path) -> None:
    change_scope_path = tmp_path / "change_scope.yaml"
    change_scope_path.write_text(
        "\n".join(
            (
                "# true YAML (not strict JSON)",
                "schema_version: 1",
                "declared_frozen_ids: none",
                "evidence:",
                "  semver_bump: null",
                "  decision_records: []",
                "  conformance_updates: []",
                "  migration_notes: []",
                "  reproducibility_impact_statement_path: null",
                "notes: YAML parsing acceptance test",
            )
        ),
        encoding="utf-8",
    )

    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md",),
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )

    assert result.passed, result.errors


def test_changed_path_detection_handles_zero_sha_base_ref_without_runtime_error() -> None:
    repo_root = _repo_root()
    paths = _changed_paths_from_git(
        repo_root=repo_root,
        base_ref="0000000000000000000000000000000000000000",
        head_ref="HEAD",
    )
    lines = _changed_lines_from_git(
        repo_root=repo_root,
        base_ref="0000000000000000000000000000000000000000",
        head_ref="HEAD",
    )
    assert isinstance(paths, tuple)
    assert isinstance(lines, dict)


def test_governance_gate_uses_baseline_rule_table_to_block_tampering_attempt(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    baseline_rule_table = _load_rule_table(
        repo_root / "docs/dev/frozen_change_governance_rules.yaml"
    )
    baseline_classification = _load_tolerance_classification(
        repo_root / "docs/dev/threshold_tolerance_classification.yaml"
    )
    tampered_rules = {
        frozen_id: rule
        for frozen_id, rule in baseline_rule_table.rules.items()
        if frozen_id != _FROZEN_ID_CLI_EXIT
    }
    tampered_rules[_FROZEN_ID_CLI_EXIT] = type(next(iter(baseline_rule_table.rules.values())))(
        frozen_id=_FROZEN_ID_CLI_EXIT,
        label="CLI exit semantics and partial-sweep behavior",
        detection=tuple(),
    )
    tampered_rule_table = GovernanceRuleTableData(
        rules=tampered_rules,
        threshold_sources=baseline_rule_table.threshold_sources,
        required_evidence_by_frozen_id=baseline_rule_table.required_evidence_by_frozen_id,
    )
    classification_data = ToleranceClassificationData(
        entries={
            "docs/spec/thresholds_v4_0_0.yaml": "normative_gating",
            "docs/dev/tolerances/calibration_seed_v1.yaml": "calibration_only",
        },
        merge_gating_sources=("docs/spec/thresholds_v4_0_0.yaml",),
    )

    result = evaluate_governance_gate(
        repo_root=repo_root,
        changed_paths=(
            "src/rfmna/cli/main.py",
            "docs/dev/frozen_change_governance_rules.yaml",
        ),
        baseline_artifacts=GovernanceBaselineArtifacts(
            rule_table_data=baseline_rule_table,
            tolerance_classification_data=baseline_classification,
        ),
    )
    assert not result.passed
    assert any("declared_frozen_ids mismatch" in error for error in result.errors)

    # Demonstrate the bypass if baseline artifacts were not used.
    bypass_result = evaluate_governance_gate(
        repo_root=repo_root,
        changed_paths=("src/rfmna/cli/main.py",),
        baseline_artifacts=GovernanceBaselineArtifacts(
            rule_table_data=tampered_rule_table,
            tolerance_classification_data=classification_data,
        ),
        artifact_paths=GovernanceArtifactPaths(
            change_scope_path=str(
                _write_change_scope(
                    tmp_path,
                    declared_frozen_ids="none",
                    evidence={
                        "semver_bump": None,
                        "decision_records": [],
                        "conformance_updates": [],
                        "migration_notes": [],
                        "reproducibility_impact_statement_path": None,
                    },
                )
            )
        ),
    )
    assert bypass_result.passed


def test_governance_gate_allows_governance_control_file_touch_without_frozen_detection(
    tmp_path: Path,
) -> None:
    change_scope_path = _write_change_scope(
        tmp_path,
        declared_frozen_ids="none",
        evidence={
            "semver_bump": None,
            "decision_records": [],
            "conformance_updates": [],
            "migration_notes": [],
            "reproducibility_impact_statement_path": None,
        },
    )
    result = evaluate_governance_gate(
        repo_root=_repo_root(),
        changed_paths=("docs/dev/phase2_gate.md", ".github/workflows/ci.yml"),
        changed_lines_by_path={
            ".github/workflows/ci.yml": ("      - name: Some non-thread-control line",),
        },
        artifact_paths=GovernanceArtifactPaths(change_scope_path=str(change_scope_path)),
    )
    assert result.passed, result.errors


def test_parse_changed_lines_file_allows_blank_lines(tmp_path: Path) -> None:
    payload_path = tmp_path / "changed_lines.yaml"
    payload_path.write_text(
        json.dumps({"foo.txt": ["", "non-empty line"]}, indent=2),
        encoding="utf-8",
    )
    parsed = _parse_changed_lines_file(payload_path)
    assert parsed["foo.txt"] == ("", "non-empty line")

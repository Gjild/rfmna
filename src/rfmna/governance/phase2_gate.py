from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Final

import yaml  # type: ignore[import-untyped]

_REQUIRED_FROZEN_IDS: Final[tuple[int, ...]] = tuple(range(1, 13))
_CHANGE_SCOPE_PATH: Final[str] = "docs/dev/change_scope.yaml"
_RULE_TABLE_PATH: Final[str] = "docs/dev/frozen_change_governance_rules.yaml"
_TOLERANCE_CLASSIFICATION_PATH: Final[str] = "docs/dev/threshold_tolerance_classification.yaml"
_GIT_DIFF_HEADER_PART_COUNT_MIN: Final[int] = 4
_ZERO_GIT_SHA: Final[str] = "0000000000000000000000000000000000000000"
_CHANGE_SCOPE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "declared_frozen_ids",
    "evidence",
)
_CHANGE_SCOPE_ALLOWED_KEYS: Final[tuple[str, ...]] = _CHANGE_SCOPE_REQUIRED_KEYS + ("notes",)
_CHANGE_SCOPE_EVIDENCE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "semver_bump",
    "decision_records",
    "conformance_updates",
    "migration_notes",
    "reproducibility_impact_statement_path",
)
_ALLOWED_EVIDENCE_REQUIREMENT_ITEMS: Final[set[str]] = set(_CHANGE_SCOPE_EVIDENCE_REQUIRED_KEYS)
_SEMVER_BUMP_REQUIRED_KEYS: Final[tuple[str, ...]] = ("from_version", "to_version")
_FROZEN_ID_THRESHOLD_STATUS_BANDS: Final[int] = 5
_PATH_GLOB_META_CHARS: Final[set[str]] = {"*", "?", "[", "]"}
_EVIDENCE_ALLOWED_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "decision_records": ("docs/spec/decision_records/",),
    "conformance_updates": ("tests/conformance/",),
    "migration_notes": ("docs/spec/migration_notes/",),
    "reproducibility_impact_statement_path": ("docs/",),
}


@dataclass(frozen=True, slots=True)
class DetectionPattern:
    path_glob: str
    line_tokens_any: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenRule:
    frozen_id: int
    label: str
    detection: tuple[DetectionPattern, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    sub_gate: str
    passed: bool
    touched_frozen_ids: tuple[int, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChangeScopeEvidence:
    semver_bump: dict[str, str] | None
    decision_records: tuple[str, ...]
    conformance_updates: tuple[str, ...]
    migration_notes: tuple[str, ...]
    reproducibility_impact_statement_path: str | None


@dataclass(frozen=True, slots=True)
class GovernanceArtifactPaths:
    change_scope_path: str = _CHANGE_SCOPE_PATH
    rule_table_path: str = _RULE_TABLE_PATH
    tolerance_classification_path: str = _TOLERANCE_CLASSIFICATION_PATH


@dataclass(frozen=True, slots=True)
class GovernancePolicyInputs:
    repo_root: Path
    changed_paths: tuple[str, ...]
    declared_ids: tuple[int, ...]
    touched_ids: tuple[int, ...]
    evidence: ChangeScopeEvidence
    threshold_sources: tuple[str, ...]
    classification_entries: dict[str, str]
    merge_gating_sources: tuple[str, ...]
    required_evidence_by_frozen_id: dict[int, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class GovernanceRuleTableData:
    rules: dict[int, FrozenRule]
    threshold_sources: tuple[str, ...]
    required_evidence_by_frozen_id: dict[int, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ToleranceClassificationData:
    entries: dict[str, str]
    merge_gating_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GovernanceBaselineArtifacts:
    rule_table_data: GovernanceRuleTableData
    tolerance_classification_data: ToleranceClassificationData


def _load_structured_dict(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact is missing: {path.as_posix()}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid JSON/YAML payload: {path.as_posix()}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a mapping: {path.as_posix()}")
    return payload


def _load_structured_dict_from_text(*, source: str, payload_text: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(payload_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid JSON/YAML payload: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a mapping: {source}")
    return payload


def _read_nonempty_lines(path: Path) -> tuple[str, ...]:
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            entries.append(stripped)
    return tuple(sorted(set(entries)))


def _coerce_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")

    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        entries.append(item)
    return tuple(entries)


def _coerce_line_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")

    entries: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        entries.append(item)
    return tuple(entries)


def _validate_mapping_keys(
    *,
    mapping: dict[str, object],
    required_keys: tuple[str, ...],
    allowed_keys: tuple[str, ...],
    field_name: str,
) -> None:
    key_set = set(mapping)
    required_set = set(required_keys)
    allowed_set = set(allowed_keys)

    missing = sorted(required_set - key_set)
    if missing:
        raise ValueError(f"{field_name} missing required keys: {', '.join(missing)}")

    extras = sorted(key_set - allowed_set)
    if extras:
        raise ValueError(f"{field_name} has unsupported keys: {', '.join(extras)}")


def _parse_detection_patterns(value: object, *, rule_id: int) -> tuple[DetectionPattern, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"frozen rule {rule_id} must define non-empty detection rules")

    patterns: list[DetectionPattern] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"frozen rule {rule_id} detection entries must be mappings")
        path_glob = entry.get("path_glob")
        if not isinstance(path_glob, str) or not path_glob:
            raise ValueError(f"frozen rule {rule_id} detection.path_glob must be non-empty")

        tokens_raw = entry.get("line_tokens_any", [])
        if not isinstance(tokens_raw, list):
            raise ValueError(f"frozen rule {rule_id} line_tokens_any must be a string list")

        tokens: list[str] = []
        for token in tokens_raw:
            if not isinstance(token, str) or not token:
                raise ValueError(
                    f"frozen rule {rule_id} line_tokens_any entries must be non-empty strings"
                )
            tokens.append(token)

        patterns.append(
            DetectionPattern(
                path_glob=path_glob,
                line_tokens_any=tuple(tokens),
            )
        )
    return tuple(patterns)


def _parse_required_evidence_items(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of evidence identifiers")

    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        if entry not in _ALLOWED_EVIDENCE_REQUIREMENT_ITEMS:
            raise ValueError(f"{field_name} contains unsupported evidence identifier: {entry}")
        items.append(entry)

    deduped = tuple(sorted(set(items)))
    return deduped


def _contains_glob_metacharacters(path_glob: str) -> bool:
    return any(char in path_glob for char in _PATH_GLOB_META_CHARS)


def _load_required_evidence_mapping(payload: dict[str, object]) -> dict[int, tuple[str, ...]]:
    raw_mapping = payload.get("required_evidence_by_frozen_id")
    if not isinstance(raw_mapping, dict):
        raise ValueError("required_evidence_by_frozen_id must be a mapping")

    parsed: dict[int, tuple[str, ...]] = {}
    for raw_key, raw_value in raw_mapping.items():
        if isinstance(raw_key, int):
            frozen_id = raw_key
        elif isinstance(raw_key, str) and raw_key.isdigit():
            frozen_id = int(raw_key)
        else:
            raise ValueError("required_evidence_by_frozen_id keys must be frozen IDs 1..12")

        if frozen_id not in _REQUIRED_FROZEN_IDS:
            raise ValueError("required_evidence_by_frozen_id keys must be within 1..12")
        if frozen_id in parsed:
            raise ValueError(f"duplicate required_evidence_by_frozen_id key: {frozen_id}")

        parsed[frozen_id] = _parse_required_evidence_items(
            raw_value,
            field_name=f"required_evidence_by_frozen_id[{frozen_id}]",
        )

    if set(parsed) != set(_REQUIRED_FROZEN_IDS):
        raise ValueError(
            "required_evidence_by_frozen_id must declare each frozen id exactly once (1..12)"
        )
    return parsed


def _validate_frozen_id_5_threshold_source_coverage(
    *,
    parsed_rules: dict[int, FrozenRule],
    threshold_sources: tuple[str, ...],
) -> None:
    rule = parsed_rules[_FROZEN_ID_THRESHOLD_STATUS_BANDS]
    id_5_sources: list[str] = []
    for detection in rule.detection:
        path_glob = detection.path_glob
        if _contains_glob_metacharacters(path_glob):
            raise ValueError(
                "frozen rule 5 detection.path_glob must be explicit path (no glob metacharacters): "
                f"{path_glob}"
            )
        id_5_sources.append(path_glob)

    missing = sorted(set(id_5_sources) - set(threshold_sources))
    if missing:
        raise ValueError(
            "frozen_threshold_status_band_sources must include all frozen rule 5 sources: "
            + ", ".join(missing)
        )


def _load_rule_table_payload(payload: dict[str, object]) -> GovernanceRuleTableData:
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("frozen governance rule table must use schema_version=1")

    raw_rules = payload.get("frozen_artifact_rules")
    if not isinstance(raw_rules, list):
        raise ValueError("frozen_artifact_rules must be a list")

    parsed_rules: dict[int, FrozenRule] = {}
    for entry in raw_rules:
        if not isinstance(entry, dict):
            raise ValueError("frozen_artifact_rules entries must be mappings")

        frozen_id = entry.get("id")
        label = entry.get("label")
        if not isinstance(frozen_id, int):
            raise ValueError("frozen_artifact rule id must be an integer")
        if not isinstance(label, str) or not label:
            raise ValueError(f"frozen rule {frozen_id} label must be non-empty")
        if frozen_id in parsed_rules:
            raise ValueError(f"duplicate frozen rule id: {frozen_id}")

        parsed_rules[frozen_id] = FrozenRule(
            frozen_id=frozen_id,
            label=label,
            detection=_parse_detection_patterns(entry.get("detection"), rule_id=frozen_id),
        )

    if set(parsed_rules) != set(_REQUIRED_FROZEN_IDS):
        raise ValueError("frozen_artifact_rules must declare each frozen id exactly once (1..12)")

    threshold_sources = _coerce_string_sequence(
        payload.get("frozen_threshold_status_band_sources"),
        field_name="frozen_threshold_status_band_sources",
    )
    _validate_frozen_id_5_threshold_source_coverage(
        parsed_rules=parsed_rules,
        threshold_sources=threshold_sources,
    )
    required_evidence_by_frozen_id = _load_required_evidence_mapping(payload)
    return GovernanceRuleTableData(
        rules=parsed_rules,
        threshold_sources=threshold_sources,
        required_evidence_by_frozen_id=required_evidence_by_frozen_id,
    )


def _load_rule_table(path: Path) -> GovernanceRuleTableData:
    return _load_rule_table_payload(_load_structured_dict(path))


def _parse_declared_frozen_ids(value: object) -> tuple[int, ...]:
    if value == "none":
        return ()
    if not isinstance(value, list):
        raise ValueError("declared_frozen_ids must be 'none' or a list of ints")

    ids: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError("declared_frozen_ids list entries must be integers")
        ids.append(item)

    if sorted(ids) != ids or len(set(ids)) != len(ids):
        raise ValueError("declared_frozen_ids must be sorted ascending with unique values")
    if any(item not in _REQUIRED_FROZEN_IDS for item in ids):
        raise ValueError("declared_frozen_ids entries must be within 1..12")
    return tuple(ids)


def _parse_semver_bump(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("change_scope.evidence.semver_bump must be a mapping or null")
    _validate_mapping_keys(
        mapping=value,
        required_keys=_SEMVER_BUMP_REQUIRED_KEYS,
        allowed_keys=_SEMVER_BUMP_REQUIRED_KEYS,
        field_name="change_scope.evidence.semver_bump",
    )

    from_version = value["from_version"]
    to_version = value["to_version"]
    if not isinstance(from_version, str) or not isinstance(to_version, str):
        raise ValueError("semver_bump.from_version/to_version must be strings")
    if not from_version.strip() or not to_version.strip():
        raise ValueError("semver_bump.from_version/to_version must be non-empty strings")
    return {"from_version": from_version, "to_version": to_version}


def _parse_reproducibility_statement_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError("reproducibility_impact_statement_path must be a non-empty string or null")


def _parse_change_scope_evidence(payload: dict[str, object]) -> ChangeScopeEvidence:
    evidence_raw = payload.get("evidence")
    if not isinstance(evidence_raw, dict):
        raise ValueError("change_scope.evidence must be a mapping")
    _validate_mapping_keys(
        mapping=evidence_raw,
        required_keys=_CHANGE_SCOPE_EVIDENCE_REQUIRED_KEYS,
        allowed_keys=_CHANGE_SCOPE_EVIDENCE_REQUIRED_KEYS,
        field_name="change_scope.evidence",
    )

    return ChangeScopeEvidence(
        semver_bump=_parse_semver_bump(evidence_raw["semver_bump"]),
        decision_records=_coerce_string_sequence(
            evidence_raw["decision_records"],
            field_name="change_scope.evidence.decision_records",
        ),
        conformance_updates=_coerce_string_sequence(
            evidence_raw["conformance_updates"],
            field_name="change_scope.evidence.conformance_updates",
        ),
        migration_notes=_coerce_string_sequence(
            evidence_raw["migration_notes"],
            field_name="change_scope.evidence.migration_notes",
        ),
        reproducibility_impact_statement_path=_parse_reproducibility_statement_path(
            evidence_raw["reproducibility_impact_statement_path"]
        ),
    )


def _load_change_scope(path: Path) -> tuple[tuple[int, ...], ChangeScopeEvidence]:
    payload = _load_structured_dict(path)
    _validate_mapping_keys(
        mapping=payload,
        required_keys=_CHANGE_SCOPE_REQUIRED_KEYS,
        allowed_keys=_CHANGE_SCOPE_ALLOWED_KEYS,
        field_name="change_scope",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("change_scope schema_version must be 1")
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("change_scope.notes must be a string when present")

    declared_ids = _parse_declared_frozen_ids(payload.get("declared_frozen_ids"))
    evidence = _parse_change_scope_evidence(payload)
    return declared_ids, evidence


def _load_tolerance_classification_payload(
    payload: dict[str, object],
) -> ToleranceClassificationData:
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("threshold/tolerance classification schema_version must be 1")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("threshold/tolerance classification entries must be a non-empty list")

    entries: dict[str, str] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("threshold/tolerance classification entry must be a mapping")
        path_value = entry.get("path")
        classification = entry.get("classification")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("classification entry path must be a non-empty string")
        if classification not in {"normative_gating", "calibration_only"}:
            raise ValueError(
                "classification entry classification must be normative_gating or calibration_only"
            )
        if path_value in entries:
            raise ValueError(f"duplicate classification entry path: {path_value}")
        entries[path_value] = classification

    merge_sources = _coerce_string_sequence(
        payload.get("merge_gating_tolerance_sources"),
        field_name="merge_gating_tolerance_sources",
    )
    promotion_note = payload.get("promotion_policy_note")
    if not isinstance(promotion_note, str) or not promotion_note.strip():
        raise ValueError("promotion_policy_note must be a non-empty string")

    return ToleranceClassificationData(entries=entries, merge_gating_sources=merge_sources)


def _load_tolerance_classification(path: Path) -> ToleranceClassificationData:
    return _load_tolerance_classification_payload(_load_structured_dict(path))


def derive_touched_frozen_ids(
    *,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    rules: dict[int, FrozenRule],
) -> tuple[int, ...]:
    touched: set[int] = set()
    for frozen_id, rule in rules.items():
        for detection in rule.detection:
            matching_paths = [path for path in changed_paths if fnmatch(path, detection.path_glob)]
            if not matching_paths:
                continue

            if not detection.line_tokens_any:
                touched.add(frozen_id)
                break

            matched_by_token = False
            for path in matching_paths:
                lines = changed_lines_by_path.get(path)
                if lines is None:
                    matched_by_token = True
                    break
                if any(
                    _line_contains_token(line=line, token=token)
                    for token in detection.line_tokens_any
                    for line in lines
                ):
                    matched_by_token = True
                    break
            if matched_by_token:
                touched.add(frozen_id)
                break
    return tuple(sorted(touched))


def _validate_required_path_list(
    *,
    repo_root: Path,
    paths: tuple[str, ...],
    list_name: str,
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    root_resolved = repo_root.resolve()
    for rel_path in paths:
        if Path(rel_path).is_absolute():
            missing.append(f"{list_name} path must be repo-relative: {rel_path}")
            continue
        if not rel_path.startswith(allowed_prefixes):
            prefix_text = ", ".join(allowed_prefixes)
            missing.append(f"{list_name} path must be under [{prefix_text}]: {rel_path}")
            continue

        candidate = (repo_root / rel_path).resolve(strict=False)
        if not candidate.is_relative_to(root_resolved):
            missing.append(f"{list_name} path must remain within repository root: {rel_path}")
            continue
        if not candidate.exists() or not candidate.is_file():
            missing.append(f"{list_name} path does not exist: {rel_path}")
    return tuple(missing)


def _validate_required_frozen_evidence(
    *,
    repo_root: Path,
    evidence: ChangeScopeEvidence,
    required_evidence_items: tuple[str, ...],
) -> tuple[str, ...]:
    errors: list[str] = []

    required_items = set(required_evidence_items)

    if "semver_bump" in required_items and evidence.semver_bump is None:
        errors.append("missing semver_bump evidence")
    elif "semver_bump" in required_items:
        assert evidence.semver_bump is not None
        from_version = evidence.semver_bump["from_version"].strip()
        to_version = evidence.semver_bump["to_version"].strip()
        if not from_version or not to_version:
            errors.append("semver_bump versions must be non-empty")
        elif from_version == to_version:
            errors.append("semver_bump from_version and to_version must differ")

    if "decision_records" in required_items and not evidence.decision_records:
        errors.append("missing decision_records evidence")
    if "conformance_updates" in required_items and not evidence.conformance_updates:
        errors.append("missing conformance_updates evidence")
    if "migration_notes" in required_items and not evidence.migration_notes:
        errors.append("missing migration_notes evidence")
    if (
        "reproducibility_impact_statement_path" in required_items
        and evidence.reproducibility_impact_statement_path is None
    ):
        errors.append("missing reproducibility_impact_statement_path evidence")

    if "decision_records" in required_items:
        errors.extend(
            _validate_required_path_list(
                repo_root=repo_root,
                paths=evidence.decision_records,
                list_name="decision_records",
                allowed_prefixes=_EVIDENCE_ALLOWED_PREFIXES["decision_records"],
            )
        )
    if "conformance_updates" in required_items:
        errors.extend(
            _validate_required_path_list(
                repo_root=repo_root,
                paths=evidence.conformance_updates,
                list_name="conformance_updates",
                allowed_prefixes=_EVIDENCE_ALLOWED_PREFIXES["conformance_updates"],
            )
        )
    if "migration_notes" in required_items:
        errors.extend(
            _validate_required_path_list(
                repo_root=repo_root,
                paths=evidence.migration_notes,
                list_name="migration_notes",
                allowed_prefixes=_EVIDENCE_ALLOWED_PREFIXES["migration_notes"],
            )
        )

    reproducibility_path = evidence.reproducibility_impact_statement_path
    if (
        "reproducibility_impact_statement_path" in required_items
        and reproducibility_path is not None
    ):
        errors.extend(
            _validate_required_path_list(
                repo_root=repo_root,
                paths=(reproducibility_path,),
                list_name="reproducibility_impact_statement_path",
                allowed_prefixes=_EVIDENCE_ALLOWED_PREFIXES[
                    "reproducibility_impact_statement_path"
                ],
            )
        )

    return tuple(errors)


def _line_contains_token(*, line: str, token: str) -> bool:
    if token in line:
        return True
    normalized_line = "".join(line.split())
    normalized_token = "".join(token.split())
    return normalized_token in normalized_line


def _empty_evidence() -> ChangeScopeEvidence:
    return ChangeScopeEvidence(
        semver_bump=None,
        decision_records=(),
        conformance_updates=(),
        migration_notes=(),
        reproducibility_impact_statement_path=None,
    )


def _run_governance_policy_checks(
    inputs: GovernancePolicyInputs,
) -> tuple[str, ...]:
    errors: list[str] = []
    if inputs.declared_ids != inputs.touched_ids:
        errors.append(
            "declared_frozen_ids mismatch: "
            f"declared={list(inputs.declared_ids)} detected={list(inputs.touched_ids)}"
        )

    required_for_touched = tuple(
        sorted(
            {
                item
                for frozen_id in inputs.touched_ids
                for item in inputs.required_evidence_by_frozen_id.get(frozen_id, ())
            }
        )
    )
    touched_scope_evidence_errors = _validate_required_frozen_evidence(
        repo_root=inputs.repo_root,
        evidence=inputs.evidence,
        required_evidence_items=required_for_touched,
    )
    has_touched_scope_evidence = not touched_scope_evidence_errors
    if inputs.touched_ids and not has_touched_scope_evidence:
        errors.append(
            "frozen artifact change requires full governance evidence: "
            + "; ".join(touched_scope_evidence_errors)
        )

    full_required_evidence = tuple(
        sorted({item for items in inputs.required_evidence_by_frozen_id.values() for item in items})
    )
    full_evidence_errors = _validate_required_frozen_evidence(
        repo_root=inputs.repo_root,
        evidence=inputs.evidence,
        required_evidence_items=full_required_evidence,
    )
    has_full_evidence = not full_evidence_errors

    for source in inputs.threshold_sources:
        classification = inputs.classification_entries.get(source)
        if classification is None:
            errors.append(
                f"missing threshold/status-band classification entry for source: {source}"
            )
            continue
        if classification != "normative_gating":
            errors.append(
                "frozen threshold/status-band source must be normative_gating: "
                f"{source} is {classification}"
            )

    for source in inputs.merge_gating_sources:
        classification = inputs.classification_entries.get(source)
        if classification is None:
            errors.append(f"merge-gating tolerance source is unclassified: {source}")
            continue
        if classification != "normative_gating":
            errors.append(
                "merge-gating tolerance source cannot be calibration_only: "
                f"{source} is {classification}"
            )

    normative_sources = {
        path
        for path, classification in inputs.classification_entries.items()
        if classification == "normative_gating"
    }
    touched_normative_sources = sorted(
        path for path in inputs.changed_paths if path in normative_sources
    )
    if touched_normative_sources and not has_full_evidence:
        errors.append(
            "touching normative_gating tolerance sources is blocked without full evidence: "
            + ", ".join(touched_normative_sources)
        )

    touched_threshold_sources = sorted(
        path for path in inputs.changed_paths if path in set(inputs.threshold_sources)
    )
    if touched_threshold_sources and not has_full_evidence:
        errors.append(
            "touching normative threshold/status-band sources is blocked without full evidence: "
            + ", ".join(touched_threshold_sources)
        )
    return tuple(errors)


def evaluate_governance_gate(
    *,
    repo_root: Path,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]] | None = None,
    artifact_paths: GovernanceArtifactPaths | None = None,
    baseline_artifacts: GovernanceBaselineArtifacts | None = None,
) -> GateResult:
    errors: list[str] = []
    changed_lines = changed_lines_by_path or {}
    paths = artifact_paths or GovernanceArtifactPaths()
    rules: dict[int, FrozenRule] = {}
    threshold_sources: tuple[str, ...] = ()
    required_evidence_by_frozen_id: dict[int, tuple[str, ...]] = {}
    declared_ids: tuple[int, ...] = ()
    evidence = _empty_evidence()
    classification_entries: dict[str, str] = {}
    merge_gating_sources: tuple[str, ...] = ()

    if baseline_artifacts is None:
        try:
            loaded_rule_table = _load_rule_table(repo_root / paths.rule_table_path)
            rules = loaded_rule_table.rules
            threshold_sources = loaded_rule_table.threshold_sources
            required_evidence_by_frozen_id = loaded_rule_table.required_evidence_by_frozen_id
        except ValueError as exc:
            errors.append(str(exc))
    else:
        rules = baseline_artifacts.rule_table_data.rules
        threshold_sources = baseline_artifacts.rule_table_data.threshold_sources
        required_evidence_by_frozen_id = (
            baseline_artifacts.rule_table_data.required_evidence_by_frozen_id
        )

    try:
        declared_ids, evidence = _load_change_scope(repo_root / paths.change_scope_path)
    except ValueError as exc:
        errors.append(str(exc))

    if baseline_artifacts is None:
        try:
            loaded_classification = _load_tolerance_classification(
                repo_root / paths.tolerance_classification_path
            )
            classification_entries = loaded_classification.entries
            merge_gating_sources = loaded_classification.merge_gating_sources
        except ValueError as exc:
            errors.append(str(exc))
    else:
        classification_entries = baseline_artifacts.tolerance_classification_data.entries
        merge_gating_sources = baseline_artifacts.tolerance_classification_data.merge_gating_sources

    touched_ids: tuple[int, ...] = ()
    if rules:
        touched_ids = derive_touched_frozen_ids(
            changed_paths=changed_paths,
            changed_lines_by_path=changed_lines,
            rules=rules,
        )

    if not errors:
        errors.extend(
            _run_governance_policy_checks(
                GovernancePolicyInputs(
                    repo_root=repo_root,
                    changed_paths=changed_paths,
                    declared_ids=declared_ids,
                    touched_ids=touched_ids,
                    evidence=evidence,
                    threshold_sources=threshold_sources,
                    classification_entries=classification_entries,
                    merge_gating_sources=merge_gating_sources,
                    required_evidence_by_frozen_id=required_evidence_by_frozen_id,
                )
            )
        )

    return GateResult(
        sub_gate="governance",
        passed=not errors,
        touched_frozen_ids=touched_ids,
        errors=tuple(errors),
    )


def _load_text_if_present(*, path: Path, errors: list[str], label: str) -> str:
    if not path.exists():
        errors.append(f"missing required file: {path.as_posix()} ({label})")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"failed to read {path.as_posix()} ({label}): {exc}")
        return ""


def evaluate_category_bootstrap_gate(*, repo_root: Path) -> GateResult:
    errors: list[str] = []

    pytest_ini_path = repo_root / "pytest.ini"
    pytest_ini_text = _load_text_if_present(
        path=pytest_ini_path, errors=errors, label="pytest config"
    )
    if "--strict-markers" not in pytest_ini_text:
        errors.append("pytest.ini must enable --strict-markers")
    if "cross_check" not in pytest_ini_text:
        errors.append("pytest.ini markers must declare cross_check")

    cross_check_dir = repo_root / "tests/cross_check"
    cross_check_tests = (
        sorted(cross_check_dir.glob("test_*.py")) if cross_check_dir.exists() else []
    )
    if not cross_check_dir.exists():
        errors.append(f"missing required directory: {cross_check_dir.as_posix()}")
    if not cross_check_tests:
        errors.append("tests/cross_check must contain at least one test_*.py file")

    regression_dir = repo_root / "tests/regression"
    regression_tests = sorted(regression_dir.glob("test_*.py")) if regression_dir.exists() else []
    if not regression_dir.exists():
        errors.append(f"missing required directory: {regression_dir.as_posix()}")
    if not regression_tests:
        errors.append("tests/regression must contain at least one test_*.py file")

    regression_note = repo_root / "docs/dev/regression_fixture_schema_convention.md"
    if not regression_note.exists():
        errors.append("regression fixture/schema convention note is missing")

    workflow_text = _load_text_if_present(
        path=repo_root / ".github/workflows/ci.yml",
        errors=errors,
        label="CI workflow",
    )
    required_fragments = (
        "Phase 2 governance sub-gate (blocking)",
        "Phase 2 category bootstrap sub-gate (blocking)",
        "uv run pytest -m cross_check --collect-only -q",
        "uv run pytest -m cross_check",
        "uv run pytest tests/regression -m regression -q",
    )
    for fragment in required_fragments:
        if fragment not in workflow_text:
            errors.append(f"CI workflow missing required Phase 2 fragment: {fragment}")

    return GateResult(
        sub_gate="category-bootstrap",
        passed=not errors,
        touched_frozen_ids=(),
        errors=tuple(errors),
    )


def _serialize_report(result: GateResult) -> str:
    lines = [
        f"sub_gate={result.sub_gate}",
        f"status={'pass' if result.passed else 'fail'}",
        f"touched_frozen_ids={','.join(str(item) for item in result.touched_frozen_ids) or 'none'}",
        f"error_count={len(result.errors)}",
    ]
    for index, error in enumerate(result.errors, start=1):
        lines.append(f"error_{index}={error}")
    return "\n".join(lines) + "\n"


def _git_command(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git command failed ({' '.join(args)}): {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def _load_rule_table_from_git_ref(
    *, repo_root: Path, ref: str, rel_path: str
) -> GovernanceRuleTableData:
    payload_text = _git_command(repo_root, "show", f"{ref}:{rel_path}")
    payload = _load_structured_dict_from_text(
        source=f"{ref}:{rel_path}",
        payload_text=payload_text,
    )
    return _load_rule_table_payload(payload)


def _load_tolerance_classification_from_git_ref(
    *,
    repo_root: Path,
    ref: str,
    rel_path: str,
) -> ToleranceClassificationData:
    payload_text = _git_command(repo_root, "show", f"{ref}:{rel_path}")
    payload = _load_structured_dict_from_text(
        source=f"{ref}:{rel_path}",
        payload_text=payload_text,
    )
    return _load_tolerance_classification_payload(payload)


def _git_ref_has_path(*, repo_root: Path, ref: str, rel_path: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{ref}:{rel_path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _is_zero_git_sha(ref: str) -> bool:
    return ref == _ZERO_GIT_SHA


def _git_ref_exists(*, repo_root: Path, ref: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _resolve_diff_base_ref(*, repo_root: Path, base_ref: str, head_ref: str) -> str:
    if not _git_ref_exists(repo_root=repo_root, ref=head_ref):
        raise RuntimeError(f"head ref does not resolve to a commit: {head_ref}")

    if _is_zero_git_sha(base_ref) or not _git_ref_exists(repo_root=repo_root, ref=base_ref):
        parent_ref = f"{head_ref}^"
        if _git_ref_exists(repo_root=repo_root, ref=parent_ref):
            return parent_ref
        return head_ref
    return base_ref


def _changed_paths_from_git(*, repo_root: Path, base_ref: str, head_ref: str) -> tuple[str, ...]:
    resolved_base = _resolve_diff_base_ref(
        repo_root=repo_root, base_ref=base_ref, head_ref=head_ref
    )
    output = _git_command(repo_root, "diff", "--name-only", resolved_base, head_ref)
    entries = [line.strip() for line in output.splitlines() if line.strip()]
    return tuple(sorted(set(entries)))


def _changed_lines_from_git(
    *,
    repo_root: Path,
    base_ref: str,
    head_ref: str,
) -> dict[str, tuple[str, ...]]:
    resolved_base = _resolve_diff_base_ref(
        repo_root=repo_root, base_ref=base_ref, head_ref=head_ref
    )
    output = _git_command(repo_root, "diff", "--unified=0", "--no-color", resolved_base, head_ref)

    current_path: str | None = None
    changes: dict[str, list[str]] = {}
    for raw_line in output.splitlines():
        if raw_line.startswith("diff --git "):
            parts = raw_line.split(" ")
            current_path = (
                parts[3][2:]
                if len(parts) >= _GIT_DIFF_HEADER_PART_COUNT_MIN and parts[3].startswith("b/")
                else None
            )
            continue

        if current_path is None:
            continue
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue
        if raw_line.startswith("+") or raw_line.startswith("-"):
            changes.setdefault(current_path, []).append(raw_line[1:])

    return {path: tuple(lines) for path, lines in changes.items()}


def _parse_changed_lines_file(path: Path) -> dict[str, tuple[str, ...]]:
    payload = _load_structured_dict(path)
    result: dict[str, tuple[str, ...]] = {}
    for changed_path, value in payload.items():
        if not isinstance(changed_path, str) or not changed_path:
            raise ValueError("changed_lines mapping keys must be non-empty paths")
        result[changed_path] = _coerce_line_sequence(
            value,
            field_name=f"changed_lines[{changed_path}]",
        )
    return result


def _resolve_changed_inputs(
    *,
    repo_root: Path,
    base_ref: str | None,
    head_ref: str | None,
    changed_paths_file: str | None,
    changed_lines_file: str | None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if changed_paths_file is not None:
        paths = _read_nonempty_lines(repo_root / changed_paths_file)
        lines = (
            _parse_changed_lines_file(repo_root / changed_lines_file)
            if changed_lines_file is not None
            else {}
        )
        return paths, lines

    if base_ref is None or head_ref is None:
        return (), {}

    paths = _changed_paths_from_git(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
    lines = _changed_lines_from_git(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
    return paths, lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 governance gate checks")
    parser.add_argument(
        "--sub-gate",
        choices=("governance", "category-bootstrap"),
        required=True,
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref")
    parser.add_argument("--changed-paths-file")
    parser.add_argument("--changed-lines-file")
    parser.add_argument("--report-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    try:
        if args.sub_gate == "category-bootstrap":
            result = evaluate_category_bootstrap_gate(repo_root=repo_root)
        else:
            baseline_artifacts: GovernanceBaselineArtifacts | None = None
            changed_paths, changed_lines = _resolve_changed_inputs(
                repo_root=repo_root,
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                changed_paths_file=args.changed_paths_file,
                changed_lines_file=args.changed_lines_file,
            )
            if (
                args.base_ref is not None
                and args.head_ref is not None
                and args.changed_paths_file is None
            ):
                baseline_ref = _resolve_diff_base_ref(
                    repo_root=repo_root,
                    base_ref=args.base_ref,
                    head_ref=args.head_ref,
                )
                if _git_ref_has_path(
                    repo_root=repo_root, ref=baseline_ref, rel_path=_RULE_TABLE_PATH
                ) and _git_ref_has_path(
                    repo_root=repo_root,
                    ref=baseline_ref,
                    rel_path=_TOLERANCE_CLASSIFICATION_PATH,
                ):
                    baseline_rule_table = _load_rule_table_from_git_ref(
                        repo_root=repo_root,
                        ref=baseline_ref,
                        rel_path=_RULE_TABLE_PATH,
                    )
                    baseline_tolerance_classification = _load_tolerance_classification_from_git_ref(
                        repo_root=repo_root,
                        ref=baseline_ref,
                        rel_path=_TOLERANCE_CLASSIFICATION_PATH,
                    )
                    baseline_artifacts = GovernanceBaselineArtifacts(
                        rule_table_data=baseline_rule_table,
                        tolerance_classification_data=baseline_tolerance_classification,
                    )
            result = evaluate_governance_gate(
                repo_root=repo_root,
                changed_paths=changed_paths,
                changed_lines_by_path=changed_lines,
                baseline_artifacts=baseline_artifacts,
            )
    except (RuntimeError, ValueError, OSError) as exc:
        result = GateResult(
            sub_gate=args.sub_gate,
            passed=False,
            touched_frozen_ids=(),
            errors=(str(exc),),
        )

    report = _serialize_report(result)
    print(report, end="")
    if args.report_file is not None:
        (repo_root / args.report_file).write_text(report, encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

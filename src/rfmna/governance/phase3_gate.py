from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import Final

from rfmna.governance.phase2_gate import (
    DetectionPattern,
    FrozenRule,
    _coerce_string_sequence,
    _git_command,
    _load_change_scope,
    _load_rule_table,
    _load_rule_table_from_git_ref,
    _load_structured_dict,
    _load_structured_dict_from_text,
    _parse_changed_lines_file,
    _read_nonempty_lines,
    _validate_mapping_keys,
    derive_touched_frozen_ids,
)

_PHASE3_CHANGE_SURFACE_PATH: Final[str] = "docs/dev/phase3_change_surface.yaml"
_PHASE3_RULE_TABLE_PATH: Final[str] = "docs/dev/phase3_contract_surface_governance_rules.yaml"
_OPTIONAL_TRACK_ACTIVATION_PATH: Final[str] = "docs/dev/optional_track_activation.yaml"
_GIT_SHA_LENGTH: Final[int] = 40
_ZERO_GIT_SHA: Final[str] = "0000000000000000000000000000000000000000"
_MAX_FROZEN_ID: Final[int] = 12
_GIT_DIFF_HEADER_PART_COUNT_MIN: Final[int] = 4
_PATH_GLOB_META_CHARS: Final[set[str]] = {"*", "?", "[", "]"}
_PRINTABLE_ASCII: Final[tuple[str, ...]] = tuple(chr(code) for code in range(32, 127))
_PHASE3_CHANGE_SURFACE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "declared_surface_ids",
    "evidence",
)
_PHASE3_CHANGE_SURFACE_ALLOWED_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "declared_surface_ids",
    "evidence",
    "notes",
)
_PHASE3_CHANGE_SURFACE_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "policy_docs",
    "schema_artifacts",
    "conformance_updates",
    "ci_enforcement",
    "process_traceability",
)
_PHASE3_ALLOWED_EVIDENCE_ITEMS: Final[set[str]] = set(_PHASE3_CHANGE_SURFACE_EVIDENCE_KEYS)
_PHASE3_EVIDENCE_ALLOWED_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "policy_docs": ("docs/dev/",),
    "schema_artifacts": ("docs/dev/", "docs/spec/schemas/", "src/rfmna/parser/resources/"),
    "conformance_updates": ("tests/conformance/",),
    "ci_enforcement": (".github/workflows/",),
    "process_traceability": ("docs/dev/",),
}
_OPTIONAL_TRACK_ALLOWED_STATES: Final[set[str]] = {"deferred", "activated"}
_OPTIONAL_TRACK_APPROVAL_STATUSES: Final[set[str]] = {"approved"}
_BOOTSTRAP_REQUIRED_CHANGED_PATHS: Final[tuple[str, ...]] = (
    _PHASE3_RULE_TABLE_PATH,
    "src/rfmna/governance/phase3_gate.py",
)
_OPTIONAL_TRACK_POLICY_DOC: Final[str] = "docs/dev/optional_track_activation_policy.md"
_PHASE3_POLICY_DOC: Final[str] = "docs/dev/phase3_change_surface_policy.md"
_PHASE3_GATE_DOC: Final[str] = "docs/dev/phase3_gate.md"
_PHASE3_PROCESS_TRACEABILITY_DOC: Final[str] = "docs/dev/phase3_process_traceability.md"
_PHASE3_CHANGE_SURFACE_SCHEMA_DOC: Final[str] = "docs/dev/phase3_change_surface_schema_v1.json"
_OPTIONAL_TRACK_SCHEMA_DOC: Final[str] = "docs/dev/optional_track_activation_schema_v1.json"
_DESIGN_BUNDLE_SCHEMA_DOC: Final[str] = "docs/spec/schemas/design_bundle_v1.json"
_PACKAGED_DESIGN_BUNDLE_SCHEMA_DOC: Final[str] = "src/rfmna/parser/resources/design_bundle_v1.json"
_LOADER_TEMP_EXCLUSIONS_SCHEMA_DOC: Final[str] = "docs/dev/p3_loader_temporary_exclusions_schema_v1.json"
_REQUIRED_CANONICAL_POLICY_PATHS: Final[tuple[str, ...]] = (
    _PHASE3_POLICY_DOC,
    _OPTIONAL_TRACK_POLICY_DOC,
)
_REQUIRED_CANONICAL_SCHEMA_PATHS: Final[tuple[str, ...]] = (
    _DESIGN_BUNDLE_SCHEMA_DOC,
    _LOADER_TEMP_EXCLUSIONS_SCHEMA_DOC,
    _PHASE3_CHANGE_SURFACE_SCHEMA_DOC,
    _OPTIONAL_TRACK_SCHEMA_DOC,
)


@dataclass(frozen=True, slots=True)
class Phase3GateResult:
    sub_gate: str
    passed: bool
    touched_surface_ids: tuple[str, ...]
    active_optional_track_ids: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase3ArtifactPaths:
    phase3_change_surface_path: str = _PHASE3_CHANGE_SURFACE_PATH
    phase3_rule_table_path: str = _PHASE3_RULE_TABLE_PATH
    optional_track_activation_path: str = _OPTIONAL_TRACK_ACTIVATION_PATH
    phase3_change_surface_schema_path: str = _PHASE3_CHANGE_SURFACE_SCHEMA_DOC
    optional_track_activation_schema_path: str = _OPTIONAL_TRACK_SCHEMA_DOC
    phase3_change_surface_policy_path: str = _PHASE3_POLICY_DOC
    optional_track_policy_path: str = _OPTIONAL_TRACK_POLICY_DOC
    change_scope_path: str = "docs/dev/change_scope.yaml"
    frozen_rule_table_path: str = "docs/dev/frozen_change_governance_rules.yaml"


@dataclass(frozen=True, slots=True)
class Phase3ChangeSurfaceEvidence:
    policy_docs: tuple[str, ...]
    schema_artifacts: tuple[str, ...]
    conformance_updates: tuple[str, ...]
    ci_enforcement: tuple[str, ...]
    process_traceability: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase3ContractSurfaceRule:
    surface_id: str
    label: str
    detection: tuple[DetectionPattern, ...]
    touch_paths: tuple[str, ...]
    required_evidence: tuple[str, ...]
    required_artifact_paths: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class OptionalTrackRule:
    track_id: str
    label: str
    activation_key: str
    touch_detection: tuple[DetectionPattern, ...]
    allowed_usage_evidence_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptionalTrackPolicy:
    freshness_window_days: int
    usage_evidence_date_format: str
    required_approval_status: str


@dataclass(frozen=True, slots=True)
class CanonicalPolicyRequirement:
    path: str
    required_fragments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalSchemaRequirement:
    path: str
    title: str
    required_properties: tuple[str, ...]
    required_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase3RuleTableData:
    surface_rules: dict[str, Phase3ContractSurfaceRule]
    policy_requirements: dict[str, CanonicalPolicyRequirement]
    schema_requirements: dict[str, CanonicalSchemaRequirement]
    optional_track_rules: dict[str, OptionalTrackRule]
    optional_track_policy: OptionalTrackPolicy


@dataclass(frozen=True, slots=True)
class OptionalTrackEvidenceSource:
    evidence_type: str
    reference: str


@dataclass(frozen=True, slots=True)
class OptionalTrackApprovalRecord:
    status: str
    approved_by: str
    decision_date: str
    decision_ref: str


@dataclass(frozen=True, slots=True)
class OptionalTrackActivationRecord:
    track_id: str
    state: str
    usage_evidence_source: OptionalTrackEvidenceSource | None
    usage_evidence_date: str | None
    activation_rationale: str
    impacted_frozen_ids: tuple[int, ...]
    approval_record: OptionalTrackApprovalRecord | None


@dataclass(frozen=True, slots=True)
class OptionalTrackActivationData:
    tracks: dict[str, OptionalTrackActivationRecord]


@dataclass(frozen=True, slots=True)
class Phase3BaselineArtifacts:
    rule_table_data: Phase3RuleTableData
    bootstrap_mode: bool
    baseline_ref: str | None


@dataclass(frozen=True, slots=True)
class ChangedInputs:
    changed_paths: tuple[str, ...]
    changed_lines_by_path: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class StrictGitRefs:
    base_ref: str
    head_ref: str


@dataclass(frozen=True, slots=True)
class _RuleTableParseContext:
    source: str
    frozen_rules: Mapping[int, FrozenRule] | None
    required_canonical_schema_paths: tuple[str, ...] | None


def _line_matches_detection(*, line: str, token: str) -> bool:
    normalized_line = _normalize_token(line)
    normalized_token = _normalize_token(token)
    return normalized_token in normalized_line


def _normalize_token(token: str) -> str:
    return "".join(token.split()).casefold()


def _contains_glob_metacharacters(path_glob: str) -> bool:
    return any(char in path_glob for char in _PATH_GLOB_META_CHARS)


@dataclass(frozen=True, slots=True)
class _GlobToken:
    kind: str
    literal: str | None = None
    chars: frozenset[str] | None = None


def _parse_glob_char_class(pattern: str, start: int) -> tuple[_GlobToken | None, int]:
    end = pattern.find("]", start + 1)
    if end == -1:
        return None, start + 1
    body = pattern[start + 1 : end]
    if not body:
        return None, start + 1
    negated = body[0] in {"!", "^"}
    if negated:
        body = body[1:]
    if not body:
        return None, start + 1

    matched_chars: set[str] = set()
    index = 0
    while index < len(body):
        char = body[index]
        if index + 2 < len(body) and body[index + 1] == "-":
            range_end = body[index + 2]
            start_ord = ord(char)
            end_ord = ord(range_end)
            if start_ord <= end_ord:
                matched_chars.update(chr(code) for code in range(start_ord, end_ord + 1))
            else:
                matched_chars.update(chr(code) for code in range(end_ord, start_ord + 1))
            index += 3
            continue
        matched_chars.add(char)
        index += 1
    if negated:
        matched_chars = set(_PRINTABLE_ASCII) - matched_chars
    return _GlobToken(kind="char_set", chars=frozenset(matched_chars)), end + 1


def _tokenize_glob_pattern(pattern: str) -> tuple[_GlobToken, ...]:
    tokens: list[_GlobToken] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            tokens.append(_GlobToken(kind="star"))
            index += 1
            continue
        if char == "?":
            tokens.append(_GlobToken(kind="char_set", chars=frozenset(_PRINTABLE_ASCII)))
            index += 1
            continue
        if char == "[":
            token, next_index = _parse_glob_char_class(pattern, index)
            if token is not None:
                tokens.append(token)
                index = next_index
                continue
        tokens.append(_GlobToken(kind="literal", literal=char))
        index += 1
    return tuple(tokens)


def _token_matches_char(token: _GlobToken, char: str) -> bool:
    if token.kind == "literal":
        assert token.literal is not None
        return token.literal == char
    if token.kind == "char_set":
        assert token.chars is not None
        return char in token.chars
    return False


def _single_char_tokens_overlap(token_a: _GlobToken, token_b: _GlobToken) -> bool:
    for char in _PRINTABLE_ASCII:
        if _token_matches_char(token_a, char) and _token_matches_char(token_b, char):
            return True
    return False


def _glob_patterns_overlap(pattern_a: str, pattern_b: str) -> bool:
    tokens_a = _tokenize_glob_pattern(pattern_a)
    tokens_b = _tokenize_glob_pattern(pattern_b)

    @cache
    def _overlap(index_a: int, index_b: int) -> bool:
        if index_a == len(tokens_a) and index_b == len(tokens_b):
            return True

        result = False
        if index_a < len(tokens_a) and tokens_a[index_a].kind == "star":
            result = _overlap(index_a + 1, index_b)
            if not result and index_b < len(tokens_b):
                token_b = tokens_b[index_b]
                if token_b.kind == "star" or any(
                    _token_matches_char(token_b, char) for char in _PRINTABLE_ASCII
                ):
                    result = _overlap(index_a, index_b + 1)
        elif index_b < len(tokens_b) and tokens_b[index_b].kind == "star":
            result = _overlap(index_a, index_b + 1)
            if not result and index_a < len(tokens_a):
                token_a = tokens_a[index_a]
                if token_a.kind == "star" or any(
                    _token_matches_char(token_a, char) for char in _PRINTABLE_ASCII
                ):
                    result = _overlap(index_a + 1, index_b)
        elif index_a < len(tokens_a) and index_b < len(tokens_b):
            result = _single_char_tokens_overlap(
                tokens_a[index_a], tokens_b[index_b]
            ) and _overlap(index_a + 1, index_b + 1)
        return result

    return _overlap(0, 0)


def _path_globs_overlap(path_glob_a: str, path_glob_b: str) -> bool:
    if path_glob_a == path_glob_b:
        return True
    if not _contains_glob_metacharacters(path_glob_a) and fnmatch(path_glob_a, path_glob_b):
        return True
    if not _contains_glob_metacharacters(path_glob_b) and fnmatch(path_glob_b, path_glob_a):
        return True
    return _glob_patterns_overlap(path_glob_a, path_glob_b)


def _detections_overlap(detection_a: DetectionPattern, detection_b: DetectionPattern) -> bool:
    if not _path_globs_overlap(detection_a.path_glob, detection_b.path_glob):
        return False
    if not detection_a.line_tokens_any or not detection_b.line_tokens_any:
        return True
    normalized_a = {_normalize_token(token) for token in detection_a.line_tokens_any}
    normalized_b = {_normalize_token(token) for token in detection_b.line_tokens_any}
    return not normalized_a.isdisjoint(normalized_b)


def _dedupe_detection_patterns(
    detections: tuple[DetectionPattern, ...],
) -> tuple[DetectionPattern, ...]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[DetectionPattern] = []
    for detection in detections:
        key = (detection.path_glob, detection.line_tokens_any)
        if key in seen:
            continue
        seen.add(key)
        result.append(detection)
    return tuple(result)


def _merge_surface_rules(
    *,
    current_rules: dict[str, Phase3ContractSurfaceRule],
    baseline_rules: dict[str, Phase3ContractSurfaceRule] | None,
) -> dict[str, Phase3ContractSurfaceRule]:
    if baseline_rules is None:
        return current_rules

    merged: dict[str, Phase3ContractSurfaceRule] = {}
    for surface_id in sorted(set(current_rules) | set(baseline_rules)):
        current_rule = current_rules.get(surface_id)
        baseline_rule = baseline_rules.get(surface_id)
        if current_rule is None:
            assert baseline_rule is not None
            merged[surface_id] = baseline_rule
            continue
        if baseline_rule is None:
            merged[surface_id] = current_rule
            continue
        merged_required_artifact_paths: dict[str, tuple[str, ...]] = {}
        for evidence_key in sorted(
            set(current_rule.required_artifact_paths) | set(baseline_rule.required_artifact_paths)
        ):
            merged_required_artifact_paths[evidence_key] = tuple(
                sorted(
                    set(baseline_rule.required_artifact_paths.get(evidence_key, ()))
                    | set(current_rule.required_artifact_paths.get(evidence_key, ()))
                )
            )
        merged[surface_id] = Phase3ContractSurfaceRule(
            surface_id=surface_id,
            label=baseline_rule.label,
            detection=_dedupe_detection_patterns(baseline_rule.detection + current_rule.detection),
            touch_paths=tuple(sorted(set(baseline_rule.touch_paths) | set(current_rule.touch_paths))),
            required_evidence=tuple(
                sorted(set(baseline_rule.required_evidence) | set(current_rule.required_evidence))
            ),
            required_artifact_paths=merged_required_artifact_paths,
        )
    return merged


def _merge_optional_track_rules(
    *,
    current_rules: dict[str, OptionalTrackRule],
    baseline_rules: dict[str, OptionalTrackRule] | None,
) -> dict[str, OptionalTrackRule]:
    if baseline_rules is None:
        return current_rules

    merged: dict[str, OptionalTrackRule] = {}
    for track_id in sorted(set(current_rules) | set(baseline_rules)):
        current_rule = current_rules.get(track_id)
        baseline_rule = baseline_rules.get(track_id)
        if current_rule is None:
            assert baseline_rule is not None
            merged[track_id] = baseline_rule
            continue
        if baseline_rule is None:
            merged[track_id] = current_rule
            continue
        merged[track_id] = OptionalTrackRule(
            track_id=track_id,
            label=baseline_rule.label,
            activation_key=baseline_rule.activation_key,
            touch_detection=_dedupe_detection_patterns(
                baseline_rule.touch_detection + current_rule.touch_detection
            ),
            allowed_usage_evidence_types=baseline_rule.allowed_usage_evidence_types,
        )
    return merged


def _merge_policy_requirements(
    *,
    current_requirements: dict[str, CanonicalPolicyRequirement],
    baseline_requirements: dict[str, CanonicalPolicyRequirement] | None,
) -> dict[str, CanonicalPolicyRequirement]:
    if baseline_requirements is None:
        return current_requirements
    merged: dict[str, CanonicalPolicyRequirement] = {}
    for path in sorted(set(current_requirements) | set(baseline_requirements)):
        current_requirement = current_requirements.get(path)
        baseline_requirement = baseline_requirements.get(path)
        if current_requirement is None:
            assert baseline_requirement is not None
            merged[path] = baseline_requirement
            continue
        if baseline_requirement is None:
            merged[path] = current_requirement
            continue
        merged[path] = CanonicalPolicyRequirement(
            path=path,
            required_fragments=tuple(
                sorted(
                    set(baseline_requirement.required_fragments)
                    | set(current_requirement.required_fragments)
                )
            ),
        )
    return merged


def _merge_schema_requirements(
    *,
    current_requirements: dict[str, CanonicalSchemaRequirement],
    baseline_requirements: dict[str, CanonicalSchemaRequirement] | None,
) -> dict[str, CanonicalSchemaRequirement]:
    if baseline_requirements is None:
        return current_requirements
    merged: dict[str, CanonicalSchemaRequirement] = {}
    for path in sorted(set(current_requirements) | set(baseline_requirements)):
        current_requirement = current_requirements.get(path)
        baseline_requirement = baseline_requirements.get(path)
        if current_requirement is None:
            assert baseline_requirement is not None
            merged[path] = baseline_requirement
            continue
        if baseline_requirement is None:
            merged[path] = current_requirement
            continue
        merged[path] = CanonicalSchemaRequirement(
            path=path,
            title=baseline_requirement.title,
            required_properties=tuple(
                sorted(
                    set(baseline_requirement.required_properties)
                    | set(current_requirement.required_properties)
                )
            ),
            required_fields=tuple(
                sorted(
                    set(baseline_requirement.required_fields)
                    | set(current_requirement.required_fields)
                )
            ),
        )
    return merged


def _detections_match(
    *,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    detections: tuple[DetectionPattern, ...],
    honor_line_tokens: bool = True,
) -> bool:
    for detection in detections:
        matching_paths = [path for path in changed_paths if fnmatch(path, detection.path_glob)]
        if not matching_paths:
            continue
        if not honor_line_tokens or not detection.line_tokens_any:
            return True
        for path in matching_paths:
            lines = changed_lines_by_path.get(path)
            if lines is None:
                return True
            if any(
                _line_matches_detection(line=line, token=token)
                for token in detection.line_tokens_any
                for line in lines
            ):
                return True
    return False


def derive_touched_phase3_surface_ids(
    *,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    rules: dict[str, Phase3ContractSurfaceRule],
) -> tuple[str, ...]:
    return _derive_touched_phase3_surface_ids(
        changed_paths=changed_paths,
        changed_lines_by_path=changed_lines_by_path,
        rules=rules,
        include_touch_paths=True,
        include_required_artifact_paths=True,
    )


def _derive_touched_phase3_surface_ids(
    *,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    rules: dict[str, Phase3ContractSurfaceRule],
    include_touch_paths: bool,
    include_required_artifact_paths: bool,
) -> tuple[str, ...]:
    touched: list[str] = []
    for rule_id, rule in sorted(rules.items()):
        if _detections_match(
            changed_paths=changed_paths,
            changed_lines_by_path=changed_lines_by_path,
            detections=rule.detection,
        ) or (
            include_touch_paths
            and _touch_paths_touched(changed_paths=changed_paths, rule=rule)
        ) or (
            include_required_artifact_paths
            and _required_artifact_paths_touched(changed_paths=changed_paths, rule=rule)
        ):
            touched.append(rule_id)
    return tuple(touched)


def _touch_paths_touched(
    *,
    changed_paths: tuple[str, ...],
    rule: Phase3ContractSurfaceRule,
) -> bool:
    return any(path in rule.touch_paths for path in changed_paths)


def _required_artifact_paths_touched(
    *,
    changed_paths: tuple[str, ...],
    rule: Phase3ContractSurfaceRule,
) -> bool:
    required_paths = {
        path
        for paths in rule.required_artifact_paths.values()
        for path in paths
    }
    return any(path in required_paths for path in changed_paths)


def derive_touched_optional_track_ids(
    *,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    rules: dict[str, OptionalTrackRule],
) -> tuple[str, ...]:
    touched: list[str] = []
    for rule_id, rule in sorted(rules.items()):
        if _detections_match(
            changed_paths=changed_paths,
            changed_lines_by_path=changed_lines_by_path,
            detections=rule.touch_detection,
            honor_line_tokens=True,
        ):
            touched.append(rule_id)
    return tuple(touched)


def _parse_detection_patterns(
    value: object,
    *,
    field_name: str,
) -> tuple[DetectionPattern, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    patterns: list[DetectionPattern] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name} entries must be mappings")
        path_glob = entry.get("path_glob")
        if not isinstance(path_glob, str) or not path_glob:
            raise ValueError(f"{field_name}.path_glob must be a non-empty string")
        line_tokens_any = entry.get("line_tokens_any", [])
        if not isinstance(line_tokens_any, list):
            raise ValueError(f"{field_name}.line_tokens_any must be a string list")
        tokens: list[str] = []
        for token in line_tokens_any:
            if not isinstance(token, str) or not token:
                raise ValueError(f"{field_name}.line_tokens_any entries must be non-empty strings")
            tokens.append(token)
        patterns.append(DetectionPattern(path_glob=path_glob, line_tokens_any=tuple(tokens)))
    return tuple(patterns)


def _parse_phase3_required_evidence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of evidence keys")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        if entry not in _PHASE3_ALLOWED_EVIDENCE_ITEMS:
            raise ValueError(f"{field_name} contains unsupported evidence key: {entry}")
        items.append(entry)
    return tuple(sorted(set(items)))


def _parse_phase3_touch_paths(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(sorted(set(_coerce_string_sequence(value, field_name=field_name))))


def _parse_phase3_required_artifact_paths(
    value: object,
    *,
    field_name: str,
    required_evidence: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")

    key_set = set(value)
    required_set = set(required_evidence)
    if key_set != required_set:
        details: list[str] = []
        missing = sorted(required_set - key_set)
        extras = sorted(key_set - required_set)
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extras:
            details.append("unsupported keys: " + ", ".join(extras))
        raise ValueError(
            f"{field_name} must declare exact required_evidence keys ({'; '.join(details)})"
        )

    parsed: dict[str, tuple[str, ...]] = {}
    for evidence_key in required_evidence:
        parsed[evidence_key] = tuple(
            sorted(
                set(
                    _coerce_string_sequence(
                        value[evidence_key],
                        field_name=f"{field_name}.{evidence_key}",
                    )
                )
            )
        )
    return parsed


def _parse_required_fragments(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of fragments")
    fragments: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        fragments.append(entry)
    return tuple(fragments)


def _parse_required_properties(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of property names")
    properties: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        properties.append(entry)
    if len(set(properties)) != len(properties):
        raise ValueError(f"{field_name} entries must be unique")
    return tuple(properties)


def _validate_phase3_rule_overlap_against_frozen_rules(
    *,
    source: str,
    surface_rules: dict[str, Phase3ContractSurfaceRule],
    frozen_rules: Mapping[int, FrozenRule] | None,
) -> None:
    if frozen_rules is None:
        return
    for rule in frozen_rules.values():
        for frozen_detection in rule.detection:
            for surface_id, surface_rule in surface_rules.items():
                for detection in surface_rule.detection:
                    if _detections_overlap(detection, frozen_detection):
                        raise ValueError(
                            "phase3 contract surface rule overlaps frozen detection surface: "
                            f"{surface_id} -> {detection.path_glob} ({source})"
                        )
                for touch_path in surface_rule.touch_paths:
                    if _detections_overlap(
                        DetectionPattern(path_glob=touch_path, line_tokens_any=()),
                        frozen_detection,
                    ):
                        raise ValueError(
                            "phase3 contract surface touch_path overlaps frozen detection surface: "
                            f"{surface_id} -> {touch_path} ({source})"
                        )


def _parse_canonical_policy_requirements(
    value: object,
    *,
    source: str,
) -> dict[str, CanonicalPolicyRequirement]:
    if not isinstance(value, list) or not value:
        raise ValueError("canonical_policy_requirements must be a non-empty list")
    requirements: dict[str, CanonicalPolicyRequirement] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("canonical_policy_requirements entries must be mappings")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("canonical_policy_requirements.path must be a non-empty string")
        if path in requirements:
            raise ValueError(f"duplicate canonical_policy_requirements path: {path}")
        requirements[path] = CanonicalPolicyRequirement(
            path=path,
            required_fragments=_parse_required_fragments(
                entry.get("required_fragments"),
                field_name=f"canonical_policy_requirements[{path}].required_fragments",
            ),
        )
    if set(requirements) != set(_REQUIRED_CANONICAL_POLICY_PATHS):
        raise ValueError(
            "canonical_policy_requirements must declare each canonical policy path exactly once: "
            + ", ".join(_REQUIRED_CANONICAL_POLICY_PATHS)
            + f" ({source})"
        )
    return requirements


def _parse_canonical_schema_requirements(
    value: object,
    *,
    source: str,
    required_paths: tuple[str, ...] | None,
) -> dict[str, CanonicalSchemaRequirement]:
    if not isinstance(value, list) or not value:
        raise ValueError("canonical_schema_requirements must be a non-empty list")
    requirements: dict[str, CanonicalSchemaRequirement] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("canonical_schema_requirements entries must be mappings")
        path = entry.get("path")
        title = entry.get("title")
        if not isinstance(path, str) or not path:
            raise ValueError("canonical_schema_requirements.path must be a non-empty string")
        if not isinstance(title, str) or not title:
            raise ValueError(
                f"canonical_schema_requirements[{path}].title must be a non-empty string"
            )
        if path in requirements:
            raise ValueError(f"duplicate canonical_schema_requirements path: {path}")
        requirements[path] = CanonicalSchemaRequirement(
            path=path,
            title=title,
            required_properties=_parse_required_properties(
                entry.get("required_properties"),
                field_name=f"canonical_schema_requirements[{path}].required_properties",
            ),
            required_fields=_parse_required_properties(
                entry.get("required_fields"),
                field_name=f"canonical_schema_requirements[{path}].required_fields",
            ),
        )
    if required_paths is not None and set(requirements) != set(required_paths):
        raise ValueError(
            "canonical_schema_requirements must declare each canonical schema path exactly once: "
            + ", ".join(required_paths)
            + f" ({source})"
        )
    return requirements


def _load_phase3_rule_table_payload(  # noqa: PLR0912,PLR0915
    payload: dict[str, object],
    *,
    parse_context: _RuleTableParseContext,
) -> Phase3RuleTableData:
    if payload.get("schema_version") != 1:
        raise ValueError("phase3 governance rule table must use schema_version=1")

    raw_surface_rules = payload.get("phase3_contract_surface_rules")
    if not isinstance(raw_surface_rules, list) or not raw_surface_rules:
        raise ValueError("phase3_contract_surface_rules must be a non-empty list")

    surface_rules: dict[str, Phase3ContractSurfaceRule] = {}
    for entry in raw_surface_rules:
        if not isinstance(entry, dict):
            raise ValueError("phase3_contract_surface_rules entries must be mappings")
        surface_id = entry.get("surface_id")
        label = entry.get("label")
        if not isinstance(surface_id, str) or not surface_id:
            raise ValueError("phase3 contract surface rule surface_id must be non-empty")
        if not isinstance(label, str) or not label:
            raise ValueError(f"phase3 contract surface rule {surface_id} label must be non-empty")
        if surface_id in surface_rules:
            raise ValueError(f"duplicate phase3 contract surface_id: {surface_id}")
        required_evidence = _parse_phase3_required_evidence(
            entry.get("required_evidence"),
            field_name=f"phase3_contract_surface_rules[{surface_id}].required_evidence",
        )
        surface_rules[surface_id] = Phase3ContractSurfaceRule(
            surface_id=surface_id,
            label=label,
            detection=_parse_detection_patterns(
                entry.get("detection"),
                field_name=f"phase3_contract_surface_rules[{surface_id}].detection",
            ),
            touch_paths=_parse_phase3_touch_paths(
                entry.get("touch_paths"),
                field_name=f"phase3_contract_surface_rules[{surface_id}].touch_paths",
            ),
            required_evidence=required_evidence,
            required_artifact_paths=_parse_phase3_required_artifact_paths(
                entry.get("required_artifact_paths"),
                field_name=f"phase3_contract_surface_rules[{surface_id}].required_artifact_paths",
                required_evidence=required_evidence,
            ),
        )

    _validate_phase3_rule_overlap_against_frozen_rules(
        source=parse_context.source,
        surface_rules=surface_rules,
        frozen_rules=parse_context.frozen_rules,
    )
    policy_requirements = _parse_canonical_policy_requirements(
        payload.get("canonical_policy_requirements"),
        source=parse_context.source,
    )
    schema_requirements = _parse_canonical_schema_requirements(
        payload.get("canonical_schema_requirements"),
        source=parse_context.source,
        required_paths=parse_context.required_canonical_schema_paths,
    )

    raw_optional_policy = payload.get("optional_track_policy")
    if not isinstance(raw_optional_policy, dict):
        raise ValueError("optional_track_policy must be a mapping")
    freshness_window_days = raw_optional_policy.get("freshness_window_days")
    usage_evidence_date_format = raw_optional_policy.get("usage_evidence_date_format")
    required_approval_status = raw_optional_policy.get("required_approval_status")
    if not isinstance(freshness_window_days, int) or freshness_window_days < 1:
        raise ValueError("optional_track_policy.freshness_window_days must be a positive integer")
    if usage_evidence_date_format != "YYYY-MM-DD":
        raise ValueError("optional_track_policy.usage_evidence_date_format must be YYYY-MM-DD")
    if required_approval_status not in _OPTIONAL_TRACK_APPROVAL_STATUSES:
        raise ValueError("optional_track_policy.required_approval_status must be approved")
    optional_policy = OptionalTrackPolicy(
        freshness_window_days=freshness_window_days,
        usage_evidence_date_format=usage_evidence_date_format,
        required_approval_status=required_approval_status,
    )

    raw_optional_rules = payload.get("optional_track_rules")
    if not isinstance(raw_optional_rules, list) or not raw_optional_rules:
        raise ValueError("optional_track_rules must be a non-empty list")
    optional_track_rules: dict[str, OptionalTrackRule] = {}
    for entry in raw_optional_rules:
        if not isinstance(entry, dict):
            raise ValueError("optional_track_rules entries must be mappings")
        track_id = entry.get("track_id")
        label = entry.get("label")
        activation_key = entry.get("activation_key")
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("optional_track_rules.track_id must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise ValueError(f"optional_track rule {track_id} label must be non-empty")
        if not isinstance(activation_key, str) or not activation_key:
            raise ValueError(f"optional_track rule {track_id} activation_key must be non-empty")
        if track_id in optional_track_rules:
            raise ValueError(f"duplicate optional_track_rules.track_id: {track_id}")
        allowed_types = _coerce_string_sequence(
            entry.get("allowed_usage_evidence_types"),
            field_name=f"optional_track_rules[{track_id}].allowed_usage_evidence_types",
        )
        optional_track_rules[track_id] = OptionalTrackRule(
            track_id=track_id,
            label=label,
            activation_key=activation_key,
            touch_detection=_parse_detection_patterns(
                entry.get("touch_detection"),
                field_name=f"optional_track_rules[{track_id}].touch_detection",
            ),
            allowed_usage_evidence_types=allowed_types,
        )
    return Phase3RuleTableData(
        surface_rules=surface_rules,
        policy_requirements=policy_requirements,
        schema_requirements=schema_requirements,
        optional_track_rules=optional_track_rules,
        optional_track_policy=optional_policy,
    )


def _load_phase3_rule_table(
    path: Path,
    *,
    frozen_rules: Mapping[int, FrozenRule] | None,
) -> Phase3RuleTableData:
    return _load_phase3_rule_table_payload(
        _load_structured_dict(path),
        parse_context=_RuleTableParseContext(
            source=path.as_posix(),
            frozen_rules=frozen_rules,
            required_canonical_schema_paths=_REQUIRED_CANONICAL_SCHEMA_PATHS,
        ),
    )


def _load_phase3_rule_table_from_git_ref(
    *,
    repo_root: Path,
    ref: str,
    rel_path: str,
    frozen_rules: Mapping[int, FrozenRule] | None,
) -> Phase3RuleTableData:
    payload_text = _git_command(repo_root, "show", f"{ref}:{rel_path}")
    payload = _load_structured_dict_from_text(source=f"{ref}:{rel_path}", payload_text=payload_text)
    return _load_phase3_rule_table_payload(
        payload,
        parse_context=_RuleTableParseContext(
            source=f"{ref}:{rel_path}",
            frozen_rules=frozen_rules,
            required_canonical_schema_paths=None,
        ),
    )


def _parse_phase3_change_surface_ids(value: object, *, valid_ids: set[str]) -> tuple[str, ...]:
    if value == "none":
        return ()
    if not isinstance(value, list):
        raise ValueError("declared_surface_ids must be 'none' or a list of strings")
    surface_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("declared_surface_ids entries must be non-empty strings")
        if item not in valid_ids:
            raise ValueError(f"declared_surface_ids contains unknown phase3 surface id: {item}")
        surface_ids.append(item)
    if sorted(surface_ids) != surface_ids or len(set(surface_ids)) != len(surface_ids):
        raise ValueError("declared_surface_ids must be sorted ascending with unique values")
    return tuple(surface_ids)


def _parse_phase3_change_surface_evidence(
    payload: dict[str, object],
) -> Phase3ChangeSurfaceEvidence:
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError("phase3_change_surface.evidence must be a mapping")
    _validate_mapping_keys(
        mapping=raw_evidence,
        required_keys=_PHASE3_CHANGE_SURFACE_EVIDENCE_KEYS,
        allowed_keys=_PHASE3_CHANGE_SURFACE_EVIDENCE_KEYS,
        field_name="phase3_change_surface.evidence",
    )
    return Phase3ChangeSurfaceEvidence(
        policy_docs=_coerce_string_sequence(
            raw_evidence["policy_docs"],
            field_name="phase3_change_surface.evidence.policy_docs",
        ),
        schema_artifacts=_coerce_string_sequence(
            raw_evidence["schema_artifacts"],
            field_name="phase3_change_surface.evidence.schema_artifacts",
        ),
        conformance_updates=_coerce_string_sequence(
            raw_evidence["conformance_updates"],
            field_name="phase3_change_surface.evidence.conformance_updates",
        ),
        ci_enforcement=_coerce_string_sequence(
            raw_evidence["ci_enforcement"],
            field_name="phase3_change_surface.evidence.ci_enforcement",
        ),
        process_traceability=_coerce_string_sequence(
            raw_evidence["process_traceability"],
            field_name="phase3_change_surface.evidence.process_traceability",
        ),
    )


def _load_phase3_change_surface(
    path: Path,
    *,
    valid_surface_ids: set[str],
) -> tuple[tuple[str, ...], Phase3ChangeSurfaceEvidence]:
    payload = _load_structured_dict(path)
    _validate_mapping_keys(
        mapping=payload,
        required_keys=_PHASE3_CHANGE_SURFACE_REQUIRED_KEYS,
        allowed_keys=_PHASE3_CHANGE_SURFACE_ALLOWED_KEYS,
        field_name="phase3_change_surface",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("phase3_change_surface schema_version must be 1")
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("phase3_change_surface.notes must be a string when present")
    return (
        _parse_phase3_change_surface_ids(payload.get("declared_surface_ids"), valid_ids=valid_surface_ids),
        _parse_phase3_change_surface_evidence(payload),
    )


def _parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc


def _parse_optional_track_evidence_source(
    value: object,
    *,
    field_name: str,
) -> OptionalTrackEvidenceSource | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping or null")
    _validate_mapping_keys(
        mapping=value,
        required_keys=("evidence_type", "reference"),
        allowed_keys=("evidence_type", "reference"),
        field_name=field_name,
    )
    evidence_type = value["evidence_type"]
    reference = value["reference"]
    if not isinstance(evidence_type, str) or not evidence_type:
        raise ValueError(f"{field_name}.evidence_type must be a non-empty string")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{field_name}.reference must be a non-empty string")
    return OptionalTrackEvidenceSource(evidence_type=evidence_type, reference=reference)


def _parse_optional_track_approval_record(
    value: object,
    *,
    field_name: str,
) -> OptionalTrackApprovalRecord | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping or null")
    _validate_mapping_keys(
        mapping=value,
        required_keys=("status", "approved_by", "decision_date", "decision_ref"),
        allowed_keys=("status", "approved_by", "decision_date", "decision_ref"),
        field_name=field_name,
    )
    status = value["status"]
    approved_by = value["approved_by"]
    decision_date = value["decision_date"]
    decision_ref = value["decision_ref"]
    if not isinstance(status, str) or status not in _OPTIONAL_TRACK_APPROVAL_STATUSES:
        raise ValueError(f"{field_name}.status must be approved")
    if not isinstance(approved_by, str) or not approved_by:
        raise ValueError(f"{field_name}.approved_by must be a non-empty string")
    if not isinstance(decision_date, str) or not decision_date:
        raise ValueError(f"{field_name}.decision_date must be a non-empty string")
    _parse_iso_date(decision_date, field_name=f"{field_name}.decision_date")
    if not isinstance(decision_ref, str) or not decision_ref:
        raise ValueError(f"{field_name}.decision_ref must be a non-empty string")
    return OptionalTrackApprovalRecord(
        status=status,
        approved_by=approved_by,
        decision_date=decision_date,
        decision_ref=decision_ref,
    )


def _parse_impacted_frozen_ids(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of ints")
    ids: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{field_name} entries must be integers")
        if item < 1 or item > _MAX_FROZEN_ID:
            raise ValueError(f"{field_name} entries must be within 1..12")
        ids.append(item)
    if sorted(ids) != ids or len(set(ids)) != len(ids):
        raise ValueError(f"{field_name} must be sorted ascending with unique values")
    return tuple(ids)


def _load_optional_track_activation(
    path: Path,
    *,
    expected_track_ids: tuple[str, ...],
) -> OptionalTrackActivationData:
    payload = _load_structured_dict(path)
    return _load_optional_track_activation_payload(payload, expected_track_ids=expected_track_ids)


def _load_optional_track_activation_payload(
    payload: dict[str, object],
    *,
    expected_track_ids: tuple[str, ...],
) -> OptionalTrackActivationData:
    _validate_mapping_keys(
        mapping=payload,
        required_keys=("schema_version", "tracks"),
        allowed_keys=("schema_version", "tracks", "notes"),
        field_name="optional_track_activation",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("optional_track_activation schema_version must be 1")
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("optional_track_activation.notes must be a string when present")

    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("optional_track_activation.tracks must be a non-empty list")

    records: dict[str, OptionalTrackActivationRecord] = {}
    for entry in raw_tracks:
        if not isinstance(entry, dict):
            raise ValueError("optional_track_activation.tracks entries must be mappings")
        _validate_mapping_keys(
            mapping=entry,
            required_keys=(
                "track_id",
                "state",
                "usage_evidence_source",
                "usage_evidence_date",
                "activation_rationale",
                "impacted_frozen_ids",
                "approval_record",
            ),
            allowed_keys=(
                "track_id",
                "state",
                "usage_evidence_source",
                "usage_evidence_date",
                "activation_rationale",
                "impacted_frozen_ids",
                "approval_record",
            ),
            field_name="optional_track_activation.tracks[]",
        )
        track_id = entry["track_id"]
        state = entry["state"]
        usage_evidence_date = entry["usage_evidence_date"]
        activation_rationale = entry["activation_rationale"]
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("optional_track_activation.tracks[].track_id must be non-empty")
        if track_id in records:
            raise ValueError(f"duplicate optional_track_activation track_id: {track_id}")
        if state not in _OPTIONAL_TRACK_ALLOWED_STATES:
            raise ValueError(
                f"optional_track_activation track {track_id} state must be one of {sorted(_OPTIONAL_TRACK_ALLOWED_STATES)}"
            )
        if usage_evidence_date is not None and not isinstance(usage_evidence_date, str):
            raise ValueError(
                f"optional_track_activation track {track_id} usage_evidence_date must be a string or null"
            )
        if not isinstance(activation_rationale, str) or not activation_rationale:
            raise ValueError(
                f"optional_track_activation track {track_id} activation_rationale must be non-empty"
            )
        if isinstance(usage_evidence_date, str):
            _parse_iso_date(
                usage_evidence_date,
                field_name=f"optional_track_activation track {track_id} usage_evidence_date",
            )
        records[track_id] = OptionalTrackActivationRecord(
            track_id=track_id,
            state=state,
            usage_evidence_source=_parse_optional_track_evidence_source(
                entry["usage_evidence_source"],
                field_name=f"optional_track_activation track {track_id} usage_evidence_source",
            ),
            usage_evidence_date=usage_evidence_date,
            activation_rationale=activation_rationale,
            impacted_frozen_ids=_parse_impacted_frozen_ids(
                entry["impacted_frozen_ids"],
                field_name=f"optional_track_activation track {track_id} impacted_frozen_ids",
            ),
            approval_record=_parse_optional_track_approval_record(
                entry["approval_record"],
                field_name=f"optional_track_activation track {track_id} approval_record",
            ),
        )

    expected = tuple(sorted(expected_track_ids))
    if tuple(sorted(records)) != expected:
        raise ValueError(
            "optional_track_activation.tracks must declare each track exactly once: "
            + ", ".join(expected)
        )
    return OptionalTrackActivationData(tracks=records)


def _load_optional_track_activation_from_git_ref(
    *,
    repo_root: Path,
    ref: str,
    rel_path: str,
    expected_track_ids: tuple[str, ...],
) -> OptionalTrackActivationData:
    payload_text = _git_command(repo_root, "show", f"{ref}:{rel_path}")
    payload = _load_structured_dict_from_text(source=f"{ref}:{rel_path}", payload_text=payload_text)
    return _load_optional_track_activation_payload(payload, expected_track_ids=expected_track_ids)


def _git_ref_contains_path(*, repo_root: Path, ref: str, rel_path: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{ref}:{rel_path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _validate_repo_relative_paths(
    *,
    repo_root: Path,
    paths: tuple[str, ...],
    list_name: str,
    allowed_prefixes: tuple[str, ...],
    required_suffix: str | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    repo_root_resolved = repo_root.resolve()
    for rel_path in paths:
        if Path(rel_path).is_absolute():
            errors.append(f"{list_name} path must be repo-relative: {rel_path}")
            continue
        if not rel_path.startswith(allowed_prefixes):
            prefix_text = ", ".join(allowed_prefixes)
            errors.append(f"{list_name} path must be under [{prefix_text}]: {rel_path}")
            continue
        if required_suffix is not None and not rel_path.endswith(required_suffix):
            errors.append(f"{list_name} path must end with {required_suffix}: {rel_path}")
            continue
        candidate = (repo_root / rel_path).resolve(strict=False)
        if not candidate.is_relative_to(repo_root_resolved):
            errors.append(f"{list_name} path must remain within repository root: {rel_path}")
            continue
        if not candidate.exists() or not candidate.is_file():
            errors.append(f"{list_name} path does not exist: {rel_path}")
    return tuple(errors)


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {path.as_posix()}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON: {path.as_posix()}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path.as_posix()}")
    return payload


def _expected_schema_artifact_id(rel_path: str) -> str:
    if rel_path == _PACKAGED_DESIGN_BUNDLE_SCHEMA_DOC:
        return _DESIGN_BUNDLE_SCHEMA_DOC
    return rel_path


def _validate_json_schema_document_shape(
    payload: Mapping[str, object],
    *,
    rel_path: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append(f"schema artifact must declare draft 2020-12 JSON Schema: {rel_path}")
    expected_schema_id = _expected_schema_artifact_id(rel_path)
    if payload.get("$id") != expected_schema_id:
        errors.append(
            "schema artifact $id must match canonical schema id: "
            f"{rel_path} -> {expected_schema_id}"
        )
    title = payload.get("title")
    if not isinstance(title, str) or not title:
        errors.append(f"schema artifact title must be a non-empty string: {rel_path}")
    if payload.get("type") != "object":
        errors.append(f"schema artifact top-level type must be object: {rel_path}")
    if payload.get("additionalProperties") is not False:
        errors.append(f"schema artifact must set additionalProperties to false: {rel_path}")
    required = payload.get("required")
    if not isinstance(required, list) or not required or any(
        not isinstance(item, str) or not item for item in required
    ):
        errors.append(f"schema artifact required must be a non-empty string list: {rel_path}")
    properties = payload.get("properties")
    if not isinstance(properties, dict) or not properties:
        errors.append(f"schema artifact properties must be a non-empty mapping: {rel_path}")
    elif isinstance(required, list):
        missing_required = sorted(
            item
            for item in required
            if isinstance(item, str) and item and item not in properties
        )
        if missing_required:
            errors.append(
                f"schema artifact required keys must exist under properties: {rel_path}: "
                + ", ".join(missing_required)
            )
    return tuple(errors)


def _validate_canonical_schema_artifact(
    *,
    artifact_type_path: str,
    payload: Mapping[str, object],
    requirement: CanonicalSchemaRequirement,
) -> tuple[str, ...]:
    errors: list[str] = []
    required = payload.get("required")
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        return ()
    if payload.get("title") != requirement.title:
        errors.append(f"schema artifact title mismatch: {artifact_type_path}")
    missing = sorted(set(requirement.required_properties) - set(properties))
    if missing:
        errors.append(
            f"schema artifact is missing canonical required properties: {artifact_type_path}: "
            + ", ".join(missing)
        )
    if not isinstance(required, list):
        return tuple(errors)
    required_field_set = {item for item in required if isinstance(item, str) and item}
    missing_required_fields = sorted(set(requirement.required_fields) - required_field_set)
    if missing_required_fields:
        errors.append(
            f"schema artifact is missing canonical required fields: {artifact_type_path}: "
            + ", ".join(missing_required_fields)
        )
    return tuple(errors)


def _validate_schema_artifact_document(
    *,
    repo_root: Path,
    rel_path: str,
    artifact_type_path: str,
    requirement: CanonicalSchemaRequirement,
) -> tuple[str, ...]:
    path = repo_root / rel_path
    try:
        payload = _load_json_mapping(path, label="schema artifact")
    except ValueError as exc:
        return (str(exc),)
    return _validate_json_schema_document_shape(
        payload,
        rel_path=rel_path,
    ) + _validate_canonical_schema_artifact(
        artifact_type_path=artifact_type_path,
        payload=payload,
        requirement=requirement,
    )


def _validate_policy_document(
    *,
    repo_root: Path,
    rel_path: str,
    required_fragments: tuple[str, ...],
) -> tuple[str, ...]:
    path = repo_root / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return (f"policy document could not be read: {rel_path}",)
    if not text.strip():
        return (f"policy document must be non-empty: {rel_path}",)
    errors: list[str] = []
    for fragment in required_fragments:
        if fragment not in text:
            errors.append(f"policy document is missing required content fragment: {rel_path}: {fragment}")
    return tuple(errors)


def _validate_phase3_surface_evidence(  # noqa: PLR0913
    *,
    repo_root: Path,
    evidence: Phase3ChangeSurfaceEvidence,
    required_items: tuple[str, ...],
    required_artifact_paths: dict[str, tuple[str, ...]],
    policy_requirements: dict[str, CanonicalPolicyRequirement],
    schema_requirements: dict[str, CanonicalSchemaRequirement],
) -> tuple[str, ...]:
    errors: list[str] = []
    for item in _PHASE3_CHANGE_SURFACE_EVIDENCE_KEYS:
        values = getattr(evidence, item)
        if item in required_items and not values:
            errors.append(f"missing phase3_change_surface evidence: {item}")
            continue
        if not values:
            continue
        errors.extend(
            _validate_repo_relative_paths(
                repo_root=repo_root,
                paths=values,
                list_name=f"phase3_change_surface.evidence.{item}",
                allowed_prefixes=_PHASE3_EVIDENCE_ALLOWED_PREFIXES[item],
                required_suffix=".json" if item == "schema_artifacts" else None,
            )
        )
        missing_required_paths = sorted(set(required_artifact_paths.get(item, ())) - set(values))
        if item in required_items and missing_required_paths:
            errors.append(
                f"phase3_change_surface.evidence.{item} missing required artifact paths: "
                + ", ".join(missing_required_paths)
            )
        if item == "schema_artifacts":
            for rel_path in values:
                if rel_path in schema_requirements:
                    errors.extend(
                        _validate_schema_artifact_document(
                            repo_root=repo_root,
                            rel_path=rel_path,
                            artifact_type_path=rel_path,
                            requirement=schema_requirements[rel_path],
                        )
                    )
                else:
                    path = repo_root / rel_path
                    try:
                        payload = _load_json_mapping(path, label="schema artifact")
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    errors.extend(
                        _validate_json_schema_document_shape(
                            payload,
                            rel_path=rel_path,
                        )
                    )
        if item == "policy_docs":
            for rel_path in values:
                errors.extend(
                    _validate_policy_document(
                        repo_root=repo_root,
                        rel_path=rel_path,
                        required_fragments=policy_requirements[rel_path].required_fragments
                        if rel_path in policy_requirements
                        else (),
                    )
                )
    return tuple(errors)


def _validate_contract_surface_cross_consistency(  # noqa: PLR0913
    *,
    repo_root: Path,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]],
    surface_rules: dict[str, Phase3ContractSurfaceRule],
    change_scope_path: str,
    frozen_rules: Mapping[int, FrozenRule],
) -> tuple[str, ...]:
    errors: list[str] = []
    frozen_rule_map = dict(frozen_rules)
    try:
        _load_change_scope(repo_root / change_scope_path)
    except ValueError as exc:
        return (f"phase3 cross-consistency requires valid change_scope authority: {exc}",)

    overlapping_paths: list[str] = []
    for path in changed_paths:
        lines_for_path = (
            {path: changed_lines_by_path[path]}
            if path in changed_lines_by_path
            else {}
        )
        path_surface_ids = derive_touched_phase3_surface_ids(
            changed_paths=(path,),
            changed_lines_by_path=lines_for_path,
            rules=surface_rules,
        )
        if not path_surface_ids:
            continue
        path_surface_detection_ids = _derive_touched_phase3_surface_ids(
            changed_paths=(path,),
            changed_lines_by_path=lines_for_path,
            rules=surface_rules,
            include_touch_paths=False,
            include_required_artifact_paths=False,
        )
        if not path_surface_detection_ids:
            continue
        path_frozen_ids = derive_touched_frozen_ids(
            changed_paths=(path,),
            changed_lines_by_path=lines_for_path,
            rules=frozen_rule_map,
        )
        if path_frozen_ids:
            overlapping_paths.append(path)
    if overlapping_paths:
        errors.append(
            "phase3/non-frozen declaration overlaps frozen declaration authority on paths: "
            + ", ".join(sorted(overlapping_paths))
        )
    return tuple(errors)


def _load_contract_gate_frozen_rules(
    *,
    repo_root: Path,
    artifact_paths: Phase3ArtifactPaths,
    baseline_ref: str | None,
) -> dict[int, FrozenRule]:
    if baseline_ref is None:
        return _load_rule_table(repo_root / artifact_paths.frozen_rule_table_path).rules
    return _load_rule_table_from_git_ref(
        repo_root=repo_root,
        ref=baseline_ref,
        rel_path=artifact_paths.frozen_rule_table_path,
    ).rules


def _validate_optional_track_activation_record(  # noqa: PLR0913
    *,
    track_rule: OptionalTrackRule,
    record: OptionalTrackActivationRecord,
    policy: OptionalTrackPolicy,
    enforce_freshness_checks: bool,
    enforce_change_scope_match: bool,
    base_ref_utc_date: date | None = None,
    declared_frozen_ids: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    if record.state != "activated":
        errors.append(
            f"optional track {record.track_id} is active by scope detection but state is {record.state}"
        )
        return tuple(errors)

    if record.usage_evidence_source is None:
        errors.append(f"optional track {record.track_id} missing usage_evidence_source")
    elif record.usage_evidence_source.evidence_type not in set(track_rule.allowed_usage_evidence_types):
        errors.append(
            f"optional track {record.track_id} usage_evidence_source.evidence_type must be one of "
            + ", ".join(track_rule.allowed_usage_evidence_types)
        )
    if record.usage_evidence_date is None:
        errors.append(f"optional track {record.track_id} missing usage_evidence_date")
    elif enforce_freshness_checks:
        assert base_ref_utc_date is not None
        usage_date = _parse_iso_date(
            record.usage_evidence_date,
            field_name=f"optional track {record.track_id} usage_evidence_date",
        )
        age_days = (base_ref_utc_date - usage_date).days
        if age_days < 0:
            errors.append(
                f"optional track {record.track_id} usage_evidence_date cannot be after base-ref UTC date"
            )
        elif age_days > policy.freshness_window_days:
            errors.append(
                f"optional track {record.track_id} usage evidence is stale: {age_days}d > {policy.freshness_window_days}d"
            )
    if record.approval_record is None:
        errors.append(f"optional track {record.track_id} missing approval_record")
    elif record.approval_record.status != policy.required_approval_status:
        errors.append(
            f"optional track {record.track_id} approval_record.status must be {policy.required_approval_status}"
        )
    if not record.impacted_frozen_ids:
        errors.append(f"optional track {record.track_id} impacted_frozen_ids must be non-empty when activated")
    if enforce_change_scope_match:
        assert declared_frozen_ids is not None
        if tuple(record.impacted_frozen_ids) != tuple(declared_frozen_ids):
            errors.append(
                f"optional track {record.track_id} impacted_frozen_ids mismatch: "
                f"activation={list(record.impacted_frozen_ids)} change_scope={list(declared_frozen_ids)}"
            )
    return tuple(errors)


def _derive_activation_decision_track_ids(
    *,
    current_activation: OptionalTrackActivationData,
    baseline_activation: OptionalTrackActivationData | None,
) -> tuple[str, ...]:
    activation_decisions: list[str] = []
    for track_id in sorted(current_activation.tracks):
        current_record = current_activation.tracks[track_id]
        baseline_record = (
            baseline_activation.tracks.get(track_id)
            if baseline_activation is not None
            else None
        )
        if current_record.state != "activated":
            continue
        if baseline_record is None or baseline_record.state != "activated":
            activation_decisions.append(track_id)
    return tuple(activation_decisions)


def _detect_surface_rule_relaxations(
    *,
    current_rules: dict[str, Phase3ContractSurfaceRule],
    baseline_rules: dict[str, Phase3ContractSurfaceRule],
) -> tuple[str, ...]:
    errors: list[str] = []
    for surface_id, baseline_surface_rule in baseline_rules.items():
        current_surface_rule = current_rules.get(surface_id)
        if current_surface_rule is None:
            errors.append(f"phase3 rule table removed baseline surface rule: {surface_id}")
            continue
        if not set(current_surface_rule.required_evidence).issuperset(
            baseline_surface_rule.required_evidence
        ):
            errors.append(f"phase3 rule table weakens required_evidence for surface: {surface_id}")
        if not set(current_surface_rule.detection).issuperset(baseline_surface_rule.detection):
            errors.append(f"phase3 rule table weakens detection for surface: {surface_id}")
        if not set(current_surface_rule.touch_paths).issuperset(baseline_surface_rule.touch_paths):
            errors.append(f"phase3 rule table weakens touch_paths for surface: {surface_id}")
        for evidence_key, baseline_paths in baseline_surface_rule.required_artifact_paths.items():
            current_paths = current_surface_rule.required_artifact_paths.get(evidence_key, ())
            if not set(current_paths).issuperset(baseline_paths):
                errors.append(
                    "phase3 rule table weakens required artifact paths for surface: "
                    f"{surface_id}.{evidence_key}"
                )
    return tuple(errors)


def _detect_policy_requirement_relaxations(
    *,
    current_requirements: dict[str, CanonicalPolicyRequirement],
    baseline_requirements: dict[str, CanonicalPolicyRequirement],
) -> tuple[str, ...]:
    errors: list[str] = []
    for path, baseline_policy_requirement in baseline_requirements.items():
        current_policy_requirement = current_requirements.get(path)
        if current_policy_requirement is None:
            errors.append(f"phase3 rule table removed canonical policy requirement: {path}")
            continue
        if not set(current_policy_requirement.required_fragments).issuperset(
            baseline_policy_requirement.required_fragments
        ):
            errors.append(f"phase3 rule table weakens canonical policy requirement: {path}")
    return tuple(errors)


def _detect_schema_requirement_relaxations(
    *,
    current_requirements: dict[str, CanonicalSchemaRequirement],
    baseline_requirements: dict[str, CanonicalSchemaRequirement],
) -> tuple[str, ...]:
    errors: list[str] = []
    for path, baseline_schema_requirement in baseline_requirements.items():
        current_schema_requirement = current_requirements.get(path)
        if current_schema_requirement is None:
            errors.append(f"phase3 rule table removed canonical schema requirement: {path}")
            continue
        if current_schema_requirement.title != baseline_schema_requirement.title:
            errors.append(f"phase3 rule table changes canonical schema title: {path}")
        if not set(current_schema_requirement.required_properties).issuperset(
            baseline_schema_requirement.required_properties
        ):
            errors.append(f"phase3 rule table weakens canonical schema requirement: {path}")
        if not set(current_schema_requirement.required_fields).issuperset(
            baseline_schema_requirement.required_fields
        ):
            errors.append(f"phase3 rule table weakens canonical schema required fields: {path}")
    return tuple(errors)


def _detect_optional_track_rule_relaxations(
    *,
    current_rules: dict[str, OptionalTrackRule],
    baseline_rules: dict[str, OptionalTrackRule],
) -> tuple[str, ...]:
    errors: list[str] = []
    for track_id, baseline_track_rule in baseline_rules.items():
        current_track_rule = current_rules.get(track_id)
        if current_track_rule is None:
            errors.append(f"phase3 rule table removed baseline optional-track rule: {track_id}")
            continue
        if not set(current_track_rule.touch_detection).issuperset(
            baseline_track_rule.touch_detection
        ):
            errors.append(f"phase3 rule table weakens optional-track detection: {track_id}")
        if not set(current_track_rule.allowed_usage_evidence_types).issubset(
            baseline_track_rule.allowed_usage_evidence_types
        ):
            errors.append(
                f"phase3 rule table weakens optional-track evidence-type policy: {track_id}"
            )
    return tuple(errors)


def _detect_optional_track_policy_relaxations(
    *,
    current_policy: OptionalTrackPolicy,
    baseline_policy: OptionalTrackPolicy,
) -> tuple[str, ...]:
    errors: list[str] = []
    if current_policy.freshness_window_days > baseline_policy.freshness_window_days:
        errors.append("phase3 rule table weakens optional-track freshness_window_days")
    if current_policy.usage_evidence_date_format != baseline_policy.usage_evidence_date_format:
        errors.append("phase3 rule table changes optional-track usage_evidence_date_format")
    if current_policy.required_approval_status != baseline_policy.required_approval_status:
        errors.append("phase3 rule table changes optional-track required_approval_status")
    return tuple(errors)


def _detect_rule_table_relaxations(
    *,
    current_rule_table: Phase3RuleTableData,
    baseline_rule_table: Phase3RuleTableData,
) -> tuple[str, ...]:
    errors: list[str] = []
    errors.extend(
        _detect_surface_rule_relaxations(
            current_rules=current_rule_table.surface_rules,
            baseline_rules=baseline_rule_table.surface_rules,
        )
    )
    errors.extend(
        _detect_policy_requirement_relaxations(
            current_requirements=current_rule_table.policy_requirements,
            baseline_requirements=baseline_rule_table.policy_requirements,
        )
    )
    errors.extend(
        _detect_schema_requirement_relaxations(
            current_requirements=current_rule_table.schema_requirements,
            baseline_requirements=baseline_rule_table.schema_requirements,
        )
    )
    errors.extend(
        _detect_optional_track_rule_relaxations(
            current_rules=current_rule_table.optional_track_rules,
            baseline_rules=baseline_rule_table.optional_track_rules,
        )
    )
    errors.extend(
        _detect_optional_track_policy_relaxations(
            current_policy=current_rule_table.optional_track_policy,
            baseline_policy=baseline_rule_table.optional_track_policy,
        )
    )
    return tuple(errors)


def _detect_activation_relaxations(
    *,
    current_activation: OptionalTrackActivationData,
    baseline_activation: OptionalTrackActivationData | None,
) -> tuple[str, ...]:
    if baseline_activation is None:
        return ()
    errors: list[str] = []
    for track_id, baseline_record in baseline_activation.tracks.items():
        if baseline_record.state != "activated":
            continue
        current_record = current_activation.tracks[track_id]
        if current_record.state != "activated":
            errors.append(
                "optional-track activation artifact relaxes active track state: "
                f"{track_id} baseline={baseline_record.state} current={current_record.state}"
            )
            continue
        if current_record.usage_evidence_source != baseline_record.usage_evidence_source:
            errors.append(
                "optional-track activation artifact mutates active track usage_evidence_source: "
                f"{track_id}"
            )
        if current_record.usage_evidence_date != baseline_record.usage_evidence_date:
            errors.append(
                "optional-track activation artifact mutates active track usage_evidence_date: "
                f"{track_id}"
            )
        if current_record.activation_rationale != baseline_record.activation_rationale:
            errors.append(
                "optional-track activation artifact mutates active track activation_rationale: "
                f"{track_id}"
            )
        if current_record.impacted_frozen_ids != baseline_record.impacted_frozen_ids:
            errors.append(
                "optional-track activation artifact mutates active track impacted_frozen_ids: "
                f"{track_id}"
            )
        if current_record.approval_record != baseline_record.approval_record:
            errors.append(
                "optional-track activation artifact mutates active track approval_record: "
                f"{track_id}"
            )
    return tuple(errors)


def _is_zero_or_missing_ref(value: str | None) -> bool:
    return value is None or not value.strip() or value == _ZERO_GIT_SHA


def _is_probably_unambiguous_commit_ref(value: str) -> bool:
    if len(value) == _GIT_SHA_LENGTH and all(char in "0123456789abcdef" for char in value.lower()):
        return True
    return "/" in value or value.startswith(("refs/", "origin/", "HEAD"))


def _resolve_strict_git_refs(*, repo_root: Path, base_ref: str | None, head_ref: str | None) -> StrictGitRefs:
    if _is_zero_or_missing_ref(base_ref):
        raise ValueError("phase3 governance requires a non-empty, non-zero base-ref")
    if _is_zero_or_missing_ref(head_ref):
        raise ValueError("phase3 governance requires a non-empty, non-zero head-ref")
    assert base_ref is not None
    assert head_ref is not None
    if not _is_probably_unambiguous_commit_ref(base_ref):
        raise ValueError(
            f"phase3 governance base-ref must be an unambiguous commit-ish or fully qualified ref: {base_ref}"
        )
    if not _is_probably_unambiguous_commit_ref(head_ref):
        raise ValueError(
            f"phase3 governance head-ref must be an unambiguous commit-ish or fully qualified ref: {head_ref}"
        )

    for label, ref in (("base-ref", base_ref), ("head-ref", head_ref)):
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError(f"phase3 governance {label} is unresolvable: {ref}")
    return StrictGitRefs(base_ref=base_ref, head_ref=head_ref)


def _resolve_changed_inputs_strict(
    *,
    repo_root: Path,
    base_ref: str | None,
    head_ref: str | None,
    changed_paths_file: str | None,
    changed_lines_file: str | None,
) -> tuple[ChangedInputs, StrictGitRefs]:
    if changed_paths_file is not None:
        refs = _resolve_strict_git_refs(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
        paths = _read_nonempty_lines(repo_root / changed_paths_file)
        lines = (
            _parse_changed_lines_file(repo_root / changed_lines_file)
            if changed_lines_file is not None
            else {}
        )
        return ChangedInputs(changed_paths=paths, changed_lines_by_path=lines), refs

    refs = _resolve_strict_git_refs(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
    changed_paths = tuple(
        sorted(
            set(
                line.strip()
                for line in _git_command(repo_root, "diff", "--name-only", refs.base_ref, refs.head_ref).splitlines()
                if line.strip()
            )
        )
    )
    changed_lines_output = _git_command(
        repo_root,
        "diff",
        "--unified=0",
        "--no-color",
        refs.base_ref,
        refs.head_ref,
    )
    current_path: str | None = None
    changed_lines: dict[str, list[str]] = {}
    for raw_line in changed_lines_output.splitlines():
        if raw_line.startswith("diff --git "):
            parts = raw_line.split(" ")
            current_path = (
                parts[3][2:]
                if len(parts) >= _GIT_DIFF_HEADER_PART_COUNT_MIN and parts[3].startswith("b/")
                else None
            )
            continue
        if current_path is None or raw_line.startswith(("+++", "---")):
            continue
        if raw_line.startswith(("+", "-")):
            changed_lines.setdefault(current_path, []).append(raw_line[1:])
    return (
        ChangedInputs(
            changed_paths=changed_paths,
            changed_lines_by_path={path: tuple(lines) for path, lines in changed_lines.items()},
        ),
        refs,
    )


def _resolve_changed_inputs_with_refs(
    *,
    repo_root: Path,
    refs: StrictGitRefs,
    changed_paths_file: str | None,
    changed_lines_file: str | None,
) -> ChangedInputs:
    if changed_paths_file is not None:
        paths = _read_nonempty_lines(repo_root / changed_paths_file)
        lines = (
            _parse_changed_lines_file(repo_root / changed_lines_file)
            if changed_lines_file is not None
            else {}
        )
        return ChangedInputs(changed_paths=paths, changed_lines_by_path=lines)

    changed_paths = tuple(
        sorted(
            set(
                line.strip()
                for line in _git_command(repo_root, "diff", "--name-only", refs.base_ref, refs.head_ref).splitlines()
                if line.strip()
            )
        )
    )
    changed_lines_output = _git_command(
        repo_root,
        "diff",
        "--unified=0",
        "--no-color",
        refs.base_ref,
        refs.head_ref,
    )
    current_path: str | None = None
    changed_lines: dict[str, list[str]] = {}
    for raw_line in changed_lines_output.splitlines():
        if raw_line.startswith("diff --git "):
            parts = raw_line.split(" ")
            current_path = (
                parts[3][2:]
                if len(parts) >= _GIT_DIFF_HEADER_PART_COUNT_MIN and parts[3].startswith("b/")
                else None
            )
            continue
        if current_path is None or raw_line.startswith(("+++", "---")):
            continue
        if raw_line.startswith(("+", "-")):
            changed_lines.setdefault(current_path, []).append(raw_line[1:])
    return ChangedInputs(
        changed_paths=changed_paths,
        changed_lines_by_path={path: tuple(lines) for path, lines in changed_lines.items()},
    )


def _base_ref_commit_utc_date(*, repo_root: Path, base_ref: str) -> date:
    timestamp = _git_command(repo_root, "show", "-s", "--format=%cI", base_ref).strip()
    if not timestamp:
        raise ValueError(f"failed to resolve UTC commit date for base-ref: {base_ref}")
    try:
        instant = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"invalid commit timestamp returned for base-ref {base_ref}: {timestamp}") from exc
    return instant.astimezone(UTC).date()


def _load_phase3_baseline_artifacts(
    *,
    repo_root: Path,
    refs: StrictGitRefs | None,
    artifact_paths: Phase3ArtifactPaths,
    changed_paths: tuple[str, ...],
    frozen_rules: Mapping[int, FrozenRule],
) -> Phase3BaselineArtifacts:
    current_rule_table = _load_phase3_rule_table(
        repo_root / artifact_paths.phase3_rule_table_path,
        frozen_rules=frozen_rules,
    )
    if refs is None:
        return Phase3BaselineArtifacts(
            rule_table_data=current_rule_table,
            bootstrap_mode=False,
            baseline_ref=None,
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "cat-file",
            "-e",
            f"{refs.base_ref}:{artifact_paths.phase3_rule_table_path}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        baseline_rule_table = _load_phase3_rule_table_from_git_ref(
            repo_root=repo_root,
            ref=refs.base_ref,
            rel_path=artifact_paths.phase3_rule_table_path,
            frozen_rules=frozen_rules,
        )
        return Phase3BaselineArtifacts(
            rule_table_data=baseline_rule_table,
            bootstrap_mode=False,
            baseline_ref=refs.base_ref,
        )

    missing_bootstrap_paths = sorted(set(_BOOTSTRAP_REQUIRED_CHANGED_PATHS) - set(changed_paths))
    if missing_bootstrap_paths:
        raise ValueError(
            "phase3 governance baseline rule table is missing at base-ref and bootstrap prerequisites are not met: "
            + ", ".join(missing_bootstrap_paths)
        )
    return Phase3BaselineArtifacts(
        rule_table_data=current_rule_table,
        bootstrap_mode=True,
        baseline_ref=refs.base_ref,
    )


def evaluate_contract_surface_governance_gate(  # noqa: PLR0913
    *,
    repo_root: Path,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]] | None = None,
    artifact_paths: Phase3ArtifactPaths | None = None,
    baseline_rule_table: Phase3RuleTableData | None = None,
    baseline_frozen_rules: Mapping[int, FrozenRule] | None = None,
) -> Phase3GateResult:
    errors: list[str] = []
    resolved_artifact_paths = artifact_paths or Phase3ArtifactPaths()
    frozen_rules_to_use = (
        dict(baseline_frozen_rules)
        if baseline_frozen_rules is not None
        else _load_rule_table(repo_root / resolved_artifact_paths.frozen_rule_table_path).rules
    )
    current_rule_table = _load_phase3_rule_table(
        repo_root / resolved_artifact_paths.phase3_rule_table_path,
        frozen_rules=frozen_rules_to_use,
    )
    merged_policy_requirements = _merge_policy_requirements(
        current_requirements=current_rule_table.policy_requirements,
        baseline_requirements=baseline_rule_table.policy_requirements
        if baseline_rule_table is not None
        else None,
    )
    merged_schema_requirements = _merge_schema_requirements(
        current_requirements=current_rule_table.schema_requirements,
        baseline_requirements=baseline_rule_table.schema_requirements
        if baseline_rule_table is not None
        else None,
    )
    errors.extend(
        _validate_repo_relative_paths(
            repo_root=repo_root,
            paths=(resolved_artifact_paths.phase3_change_surface_schema_path,),
            list_name="phase3 canonical schema artifact",
            allowed_prefixes=("docs/dev/",),
            required_suffix=".json",
        )
    )
    errors.extend(
        _validate_schema_artifact_document(
            repo_root=repo_root,
            rel_path=resolved_artifact_paths.phase3_change_surface_schema_path,
            artifact_type_path=_PHASE3_CHANGE_SURFACE_SCHEMA_DOC,
            requirement=merged_schema_requirements[_PHASE3_CHANGE_SURFACE_SCHEMA_DOC],
        )
    )
    errors.extend(
        _validate_repo_relative_paths(
            repo_root=repo_root,
            paths=(resolved_artifact_paths.phase3_change_surface_policy_path,),
            list_name="phase3 canonical policy document",
            allowed_prefixes=("docs/dev/",),
        )
    )
    errors.extend(
        _validate_policy_document(
            repo_root=repo_root,
            rel_path=resolved_artifact_paths.phase3_change_surface_policy_path,
            required_fragments=merged_policy_requirements[_PHASE3_POLICY_DOC].required_fragments,
        )
    )
    merged_surface_rules = _merge_surface_rules(
        current_rules=current_rule_table.surface_rules,
        baseline_rules=baseline_rule_table.surface_rules if baseline_rule_table is not None else None,
    )
    declared_surface_ids, evidence = _load_phase3_change_surface(
        repo_root / resolved_artifact_paths.phase3_change_surface_path,
        valid_surface_ids=set(merged_surface_rules),
    )
    touched_surface_ids = derive_touched_phase3_surface_ids(
        changed_paths=changed_paths,
        changed_lines_by_path=changed_lines_by_path or {},
        rules=merged_surface_rules,
    )
    declaration_path_touched = _PHASE3_CHANGE_SURFACE_PATH in changed_paths
    if (touched_surface_ids or declaration_path_touched) and declared_surface_ids != touched_surface_ids:
        errors.append(
            "declared_surface_ids mismatch: "
            f"declared={list(declared_surface_ids)} detected={list(touched_surface_ids)}"
        )
    errors.extend(
        _validate_contract_surface_cross_consistency(
            repo_root=repo_root,
            changed_paths=changed_paths,
            changed_lines_by_path=changed_lines_by_path or {},
            surface_rules=merged_surface_rules,
            change_scope_path=resolved_artifact_paths.change_scope_path,
            frozen_rules=frozen_rules_to_use,
        )
    )
    required_evidence = tuple(
        sorted(
            {
                item
                for surface_id in touched_surface_ids
                for item in merged_surface_rules[surface_id].required_evidence
            }
        )
    )
    required_artifact_paths: dict[str, tuple[str, ...]] = {
        item: tuple(
            sorted(
                {
                    path
                    for surface_id in touched_surface_ids
                    for path in merged_surface_rules[surface_id].required_artifact_paths.get(item, ())
                }
            )
        )
        for item in required_evidence
    }
    errors.extend(
        _validate_phase3_surface_evidence(
            repo_root=repo_root,
            evidence=evidence,
            required_items=required_evidence,
            required_artifact_paths=required_artifact_paths,
            policy_requirements=merged_policy_requirements,
            schema_requirements=merged_schema_requirements,
        )
    )
    if errors:
        errors = [
            error if "phase3_change_surface evidence" not in error else "phase3 contract surface change requires evidence: " + error
            for error in errors
        ]
    return Phase3GateResult(
        sub_gate="contract-surface-governance",
        passed=not errors,
        touched_surface_ids=touched_surface_ids,
        active_optional_track_ids=(),
        errors=tuple(errors),
    )


def evaluate_anti_tamper_gate(
    *,
    repo_root: Path,
    base_ref: str | None,
    head_ref: str | None,
    changed_paths: tuple[str, ...],
    artifact_paths: Phase3ArtifactPaths | None = None,
) -> Phase3GateResult:
    errors: list[str] = []
    resolved_artifact_paths = artifact_paths or Phase3ArtifactPaths()
    refs: StrictGitRefs | None = None
    try:
        refs = _resolve_strict_git_refs(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
    except ValueError as exc:
        errors.append(str(exc))

    if refs is not None:
        frozen_rules = _load_contract_gate_frozen_rules(
            repo_root=repo_root,
            artifact_paths=resolved_artifact_paths,
            baseline_ref=refs.base_ref,
        )
        try:
            baseline_artifacts = _load_phase3_baseline_artifacts(
                repo_root=repo_root,
                refs=refs,
                artifact_paths=resolved_artifact_paths,
                changed_paths=changed_paths,
                frozen_rules=frozen_rules,
            )
        except ValueError as exc:
            errors.append(str(exc))
        else:
            current_rule_table = _load_phase3_rule_table(
                repo_root / resolved_artifact_paths.phase3_rule_table_path,
                frozen_rules=frozen_rules,
            )
            errors.extend(
                _detect_rule_table_relaxations(
                    current_rule_table=current_rule_table,
                    baseline_rule_table=baseline_artifacts.rule_table_data,
                )
            )
            if _git_ref_contains_path(
                repo_root=repo_root,
                ref=refs.base_ref,
                rel_path=_OPTIONAL_TRACK_ACTIVATION_PATH,
            ):
                baseline_activation = _load_optional_track_activation_from_git_ref(
                    repo_root=repo_root,
                    ref=refs.base_ref,
                    rel_path=_OPTIONAL_TRACK_ACTIVATION_PATH,
                    expected_track_ids=tuple(
                        sorted(baseline_artifacts.rule_table_data.optional_track_rules)
                    ),
                )
                current_activation = _load_optional_track_activation(
                    repo_root / resolved_artifact_paths.optional_track_activation_path,
                    expected_track_ids=tuple(
                        sorted(baseline_artifacts.rule_table_data.optional_track_rules)
                    ),
                )
                errors.extend(
                    _detect_activation_relaxations(
                        current_activation=current_activation,
                        baseline_activation=baseline_activation,
                    )
                )
    return Phase3GateResult(
        sub_gate="anti-tamper",
        passed=not errors,
        touched_surface_ids=(),
        active_optional_track_ids=(),
        errors=tuple(errors),
    )


def evaluate_optional_track_gate(  # noqa: PLR0911,PLR0912,PLR0913,PLR0915
    *,
    repo_root: Path,
    changed_paths: tuple[str, ...],
    changed_lines_by_path: dict[str, tuple[str, ...]] | None = None,
    base_ref: str | None,
    head_ref: str | None,
    artifact_paths: Phase3ArtifactPaths | None = None,
    baseline_rule_table: Phase3RuleTableData | None = None,
) -> Phase3GateResult:
    errors: list[str] = []
    resolved_artifact_paths = artifact_paths or Phase3ArtifactPaths()
    activation_artifact_path = resolved_artifact_paths.optional_track_activation_path
    activation_artifact_rel_path = _OPTIONAL_TRACK_ACTIVATION_PATH
    current_rule_table = _load_phase3_rule_table(
        repo_root / resolved_artifact_paths.phase3_rule_table_path,
        frozen_rules=_load_rule_table(repo_root / resolved_artifact_paths.frozen_rule_table_path).rules,
    )
    merged_policy_requirements = _merge_policy_requirements(
        current_requirements=current_rule_table.policy_requirements,
        baseline_requirements=baseline_rule_table.policy_requirements
        if baseline_rule_table is not None
        else None,
    )
    merged_schema_requirements = _merge_schema_requirements(
        current_requirements=current_rule_table.schema_requirements,
        baseline_requirements=baseline_rule_table.schema_requirements
        if baseline_rule_table is not None
        else None,
    )
    rules_to_use = _merge_optional_track_rules(
        current_rules=current_rule_table.optional_track_rules,
        baseline_rules=baseline_rule_table.optional_track_rules if baseline_rule_table is not None else None,
    )
    policy_to_use = (
        baseline_rule_table.optional_track_policy
        if baseline_rule_table is not None
        else current_rule_table.optional_track_policy
    )
    activation_artifact_touched = activation_artifact_rel_path in changed_paths
    touched_track_ids = derive_touched_optional_track_ids(
        changed_paths=changed_paths,
        changed_lines_by_path=changed_lines_by_path or {},
        rules=rules_to_use,
    )
    activation: OptionalTrackActivationData | None = None
    activation_load_error: str | None = None
    try:
        activation = _load_optional_track_activation(
            repo_root / activation_artifact_path,
            expected_track_ids=tuple(sorted(rules_to_use)),
        )
    except ValueError as exc:
        activation_load_error = str(exc)
        if activation_artifact_touched:
            return Phase3GateResult(
                sub_gate="optional-track-activation",
                passed=False,
                touched_surface_ids=(),
                active_optional_track_ids=tuple(sorted(set(touched_track_ids))),
                errors=(activation_load_error,),
            )
        if not touched_track_ids:
            try:
                load_failure_refs = _resolve_strict_git_refs(
                    repo_root=repo_root,
                    base_ref=base_ref,
                    head_ref=head_ref,
                )
            except ValueError:
                load_failure_refs = None
            if load_failure_refs is not None and _git_ref_contains_path(
                repo_root=repo_root,
                ref=load_failure_refs.base_ref,
                rel_path=activation_artifact_rel_path,
            ):
                baseline_activation_at_base_ref = _load_optional_track_activation_from_git_ref(
                    repo_root=repo_root,
                    ref=load_failure_refs.base_ref,
                    rel_path=activation_artifact_rel_path,
                    expected_track_ids=tuple(sorted(rules_to_use)),
                )
                baseline_active_track_ids = tuple(
                    sorted(
                        track_id
                        for track_id, record in baseline_activation_at_base_ref.tracks.items()
                        if record.state == "activated"
                    )
                )
                if baseline_active_track_ids:
                    return Phase3GateResult(
                        sub_gate="optional-track-activation",
                        passed=False,
                        touched_surface_ids=(),
                        active_optional_track_ids=baseline_active_track_ids,
                        errors=(activation_load_error,),
                    )
            return Phase3GateResult(
                sub_gate="optional-track-activation",
                passed=True,
                touched_surface_ids=(),
                active_optional_track_ids=(),
                errors=(),
            )

    if activation is None:
        assert activation_load_error is not None
        return Phase3GateResult(
            sub_gate="optional-track-activation",
            passed=False,
            touched_surface_ids=(),
            active_optional_track_ids=tuple(sorted(set(touched_track_ids))),
            errors=(activation_load_error,),
        )

    activated_track_ids = tuple(
        sorted(track_id for track_id, record in activation.tracks.items() if record.state == "activated")
    )
    active_track_ids = tuple(sorted(set(touched_track_ids) | set(activated_track_ids)))
    if not active_track_ids:
        return Phase3GateResult(
            sub_gate="optional-track-activation",
            passed=True,
            touched_surface_ids=(),
            active_optional_track_ids=(),
            errors=(),
        )

    baseline_activation: OptionalTrackActivationData | None = None
    refs: StrictGitRefs | None = None
    try:
        refs = _resolve_strict_git_refs(repo_root=repo_root, base_ref=base_ref, head_ref=head_ref)
    except ValueError as exc:
        errors.append(str(exc))
        return Phase3GateResult(
            sub_gate="optional-track-activation",
            passed=False,
            touched_surface_ids=(),
            active_optional_track_ids=active_track_ids,
            errors=tuple(errors),
        )
    else:
        if _git_ref_contains_path(
            repo_root=repo_root,
            ref=refs.base_ref,
            rel_path=activation_artifact_rel_path,
        ):
            baseline_activation = _load_optional_track_activation_from_git_ref(
                repo_root=repo_root,
                ref=refs.base_ref,
                rel_path=activation_artifact_rel_path,
                expected_track_ids=tuple(sorted(rules_to_use)),
            )
    activation_decision_track_ids = _derive_activation_decision_track_ids(
        current_activation=activation,
        baseline_activation=baseline_activation,
    )

    errors.extend(
        _validate_repo_relative_paths(
            repo_root=repo_root,
            paths=(resolved_artifact_paths.optional_track_activation_schema_path,),
            list_name="optional-track canonical schema artifact",
            allowed_prefixes=("docs/dev/",),
            required_suffix=".json",
        )
    )
    errors.extend(
        _validate_schema_artifact_document(
            repo_root=repo_root,
            rel_path=resolved_artifact_paths.optional_track_activation_schema_path,
            artifact_type_path=_OPTIONAL_TRACK_SCHEMA_DOC,
            requirement=merged_schema_requirements[_OPTIONAL_TRACK_SCHEMA_DOC],
        )
    )
    errors.extend(
        _validate_repo_relative_paths(
            repo_root=repo_root,
            paths=(resolved_artifact_paths.optional_track_policy_path,),
            list_name="optional-track canonical policy document",
            allowed_prefixes=("docs/dev/",),
        )
    )
    errors.extend(
        _validate_policy_document(
            repo_root=repo_root,
            rel_path=resolved_artifact_paths.optional_track_policy_path,
            required_fragments=merged_policy_requirements[
                _OPTIONAL_TRACK_POLICY_DOC
            ].required_fragments,
        )
    )
    if errors:
        return Phase3GateResult(
            sub_gate="optional-track-activation",
            passed=False,
            touched_surface_ids=(),
            active_optional_track_ids=active_track_ids,
            errors=tuple(errors),
        )

    assert refs is not None

    base_ref_utc_date = _base_ref_commit_utc_date(repo_root=repo_root, base_ref=refs.base_ref)
    declared_frozen_ids = None
    if activation_decision_track_ids:
        declared_frozen_ids, _ = _load_change_scope(repo_root / resolved_artifact_paths.change_scope_path)

    active_track_id_set = set(active_track_ids)
    activation_decision_track_id_set = set(activation_decision_track_ids)
    for track_id in active_track_ids:
        track_rule = rules_to_use[track_id]
        record = activation.tracks[track_id]
        errors.extend(
            _validate_optional_track_activation_record(
                track_rule=track_rule,
                record=record,
                policy=policy_to_use,
                enforce_freshness_checks=track_id in active_track_id_set,
                enforce_change_scope_match=track_id in activation_decision_track_id_set,
                base_ref_utc_date=base_ref_utc_date,
                declared_frozen_ids=declared_frozen_ids,
            )
        )
    return Phase3GateResult(
        sub_gate="optional-track-activation",
        passed=not errors,
        touched_surface_ids=(),
        active_optional_track_ids=active_track_ids,
        errors=tuple(errors),
    )


def _serialize_report(result: Phase3GateResult) -> str:
    lines = [
        f"sub_gate={result.sub_gate}",
        f"status={'pass' if result.passed else 'fail'}",
        "touched_surface_ids=" + (",".join(result.touched_surface_ids) or "none"),
        "active_optional_track_ids=" + (",".join(result.active_optional_track_ids) or "none"),
        f"error_count={len(result.errors)}",
    ]
    for index, error in enumerate(result.errors, start=1):
        lines.append(f"error_{index}={error}")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 governance gate checks")
    parser.add_argument(
        "--sub-gate",
        choices=("contract-surface", "anti-tamper", "optional-track", "all"),
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
        refs = _resolve_strict_git_refs(
            repo_root=repo_root,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
        )
        changed_inputs = _resolve_changed_inputs_with_refs(
            repo_root=repo_root,
            refs=refs,
            changed_paths_file=args.changed_paths_file,
            changed_lines_file=args.changed_lines_file,
        )
        artifact_paths = Phase3ArtifactPaths()
        baseline_frozen_rules = _load_contract_gate_frozen_rules(
            repo_root=repo_root,
            artifact_paths=artifact_paths,
            baseline_ref=refs.base_ref,
        )
        results: list[Phase3GateResult] = []
        if args.sub_gate in {"contract-surface", "all"}:
            baseline_artifacts = _load_phase3_baseline_artifacts(
                repo_root=repo_root,
                refs=refs,
                artifact_paths=artifact_paths,
                changed_paths=changed_inputs.changed_paths,
                frozen_rules=baseline_frozen_rules,
            )
            results.append(
                evaluate_contract_surface_governance_gate(
                    repo_root=repo_root,
                    changed_paths=changed_inputs.changed_paths,
                    changed_lines_by_path=changed_inputs.changed_lines_by_path,
                    artifact_paths=artifact_paths,
                    baseline_rule_table=baseline_artifacts.rule_table_data,
                    baseline_frozen_rules=baseline_frozen_rules,
                )
            )
        if args.sub_gate in {"anti-tamper", "all"}:
            results.append(
                evaluate_anti_tamper_gate(
                    repo_root=repo_root,
                    base_ref=refs.base_ref,
                    head_ref=refs.head_ref,
                    changed_paths=changed_inputs.changed_paths,
                    artifact_paths=artifact_paths,
                )
            )
        if args.sub_gate in {"optional-track", "all"}:
            baseline_artifacts = _load_phase3_baseline_artifacts(
                repo_root=repo_root,
                refs=refs,
                artifact_paths=artifact_paths,
                changed_paths=changed_inputs.changed_paths,
                frozen_rules=baseline_frozen_rules,
            )
            results.append(
                evaluate_optional_track_gate(
                    repo_root=repo_root,
                    changed_paths=changed_inputs.changed_paths,
                    changed_lines_by_path=changed_inputs.changed_lines_by_path,
                    base_ref=refs.base_ref,
                    head_ref=refs.head_ref,
                    artifact_paths=artifact_paths,
                    baseline_rule_table=baseline_artifacts.rule_table_data,
                )
            )
    except (RuntimeError, ValueError) as exc:
        results = [
            Phase3GateResult(
                sub_gate=args.sub_gate,
                passed=False,
                touched_surface_ids=(),
                active_optional_track_ids=(),
                errors=(str(exc),),
            )
        ]

    exit_code = 0
    serialized = []
    for result in results:
        if not result.passed:
            exit_code = 1
        report = _serialize_report(result)
        serialized.append(report.rstrip())
        print(report, end="")
    if args.report_file:
        Path(args.report_file).write_text("\n\n".join(serialized) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from random import Random

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.diagnostics import (
    CANONICAL_DIAGNOSTIC_CATALOG,
    REQUIRED_CATALOG_FIELDS,
    Severity,
    SolverStage,
    adapt_validation_issue,
    adapt_validation_issues,
    canonical_witness_json,
)
from rfmna.diagnostics.adapters import ValidationIssueAdapterError
from rfmna.elements import (
    CapacitorStamp,
    ConductanceStamp,
    CurrentSourceStamp,
    InductorStamp,
    ResistorStamp,
    StampContext,
    ValidationIssue,
    VCCSStamp,
    VCVSStamp,
    VoltageSourceStamp,
)

pytestmark = pytest.mark.unit

_REPEATS = 25
EXPECTED_VALIDATOR_CODES = (
    "E_MODEL_C_NEGATIVE",
    "E_MODEL_G_NEGATIVE",
    "E_MODEL_ISRC_INVALID",
    "E_MODEL_L_NONPOSITIVE",
    "E_MODEL_R_NONPOSITIVE",
    "E_MODEL_VCCS_INVALID",
    "E_MODEL_VCVS_INVALID",
    "E_MODEL_VSRC_INVALID",
)


def _ctx() -> StampContext:
    return StampContext(omega_rad_s=7.0, resolved_params={"alpha": 1.0})


def _invalid_validation_issues() -> tuple[tuple[str, ValidationIssue], ...]:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("L1:i", "V1:i", "E1:i"))
    invalid_stamps = (
        ResistorStamp("Rbad", "n1", "0", 0.0, indexing),
        CapacitorStamp("Cbad", "n1", "0", -1.0, indexing),
        ConductanceStamp("Gbad", "n1", "0", -1.0, indexing),
        InductorStamp("Lbad", "n1", "0", "L1:i", 0.0, indexing),
        CurrentSourceStamp("Ibad", "n1", "0", float("nan"), indexing),
        VoltageSourceStamp("Vbad", "n1", "0", "V1:i", float("inf"), indexing),
        VCCSStamp("GMbad", "n1", "0", "nc", "nd", float("nan"), indexing),
        VCVSStamp("Ebad", "n1", "0", "nc", "nd", "E1:i", float("inf"), indexing),
    )
    issues: list[tuple[str, ValidationIssue]] = []
    for stamp in invalid_stamps:
        issue = stamp.validate(_ctx())[0]
        issues.append((stamp.element_id, issue))
    return tuple(issues)


def test_all_current_element_validators_map_to_schema_valid_diagnostics() -> None:
    events = [
        adapt_validation_issue(issue, element_id=element_id)
        for element_id, issue in _invalid_validation_issues()
    ]
    assert tuple(sorted(event.code for event in events)) == EXPECTED_VALIDATOR_CODES
    for event in events:
        assert event.severity == Severity.ERROR
        assert event.solver_stage == SolverStage.ASSEMBLE
        assert event.element_id is not None
        assert isinstance(event.suggested_action, str) and event.suggested_action


def test_adapter_preserves_overrides_context_and_stable_witness_ordering() -> None:
    issue = ValidationIssue(
        code="E_MODEL_R_NONPOSITIVE",
        message="resistance_ohm must be > 0",
        context={"z": 1, "node_id": "n1", "element_id": "R1", "a": 2},
    )
    event = adapt_validation_issue(
        issue,
        element_id=None,
        solver_stage=SolverStage.POSTPROCESS,
        severity=Severity.WARNING,
        suggested_action="custom action",
    )

    assert event.element_id == "R1"
    assert event.node_context is not None
    assert event.node_context.node_id == "n1"
    assert event.severity == Severity.WARNING
    assert event.solver_stage == SolverStage.POSTPROCESS
    assert event.suggested_action == "custom action"
    assert canonical_witness_json(event.witness) == (
        '{"issue_code":"E_MODEL_R_NONPOSITIVE","validation_context":{"a":2,"element_id":"R1","node_id":"n1","z":1}}'
    )


def test_mapping_is_deterministic_under_input_permutations() -> None:
    issues = tuple(issue for _, issue in _invalid_validation_issues())
    baseline = adapt_validation_issues(issues, element_id="E_SHARED")
    rng = Random(0)
    permutable = list(issues)

    for _ in range(_REPEATS):
        rng.shuffle(permutable)
        assert adapt_validation_issues(tuple(permutable), element_id="E_SHARED") == baseline


def test_adapter_raises_structured_error_not_string_only() -> None:
    issue = ValidationIssue(code="E_MODEL_R_NONPOSITIVE", message="bad", context=None)
    with pytest.raises(ValidationIssueAdapterError) as exc_info:
        adapt_validation_issue(issue, element_id=None)
    detail = exc_info.value.detail
    assert detail.code == "ADAPTER_CONTEXT_MISSING"
    assert detail.issue_code == "E_MODEL_R_NONPOSITIVE"
    assert detail.witness == {"issue_code": "E_MODEL_R_NONPOSITIVE"}


def test_catalog_codes_are_unique_and_schema_complete() -> None:
    assert len(CANONICAL_DIAGNOSTIC_CATALOG) == len(set(CANONICAL_DIAGNOSTIC_CATALOG))
    for code, metadata in CANONICAL_DIAGNOSTIC_CATALOG.items():
        assert code == metadata.code
        assert code.startswith("E_")
        assert (
            tuple(field.name for field in metadata.__dataclass_fields__.values())
            == REQUIRED_CATALOG_FIELDS
        )
        assert metadata.suggested_action

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rfmna.elements.base import ValidationIssue

from .catalog import CANONICAL_DIAGNOSTIC_CATALOG
from .models import DiagnosticEvent, NodeContext, PortContext, Severity, SolverStage
from .sort import sort_diagnostics

_DEFAULT_SUGGESTED_ACTION = "resolve element validation issue and retry"


@dataclass(frozen=True, slots=True)
class ValidationIssueAdapterErrorDetail:
    code: str
    message: str
    issue_code: str
    witness: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("adapter error code must be non-empty")
        if not self.message:
            raise ValueError("adapter error message must be non-empty")
        if not self.issue_code:
            raise ValueError("adapter error issue_code must be non-empty")
        canonical_witness = {key: self.witness[key] for key in sorted(self.witness)}
        object.__setattr__(self, "witness", MappingProxyType(canonical_witness))


class ValidationIssueAdapterError(ValueError):
    def __init__(self, detail: ValidationIssueAdapterErrorDetail) -> None:
        super().__init__(f"{detail.code}: {detail.message}")
        self.detail = detail


def adapt_validation_issue(
    issue: ValidationIssue,
    *,
    element_id: str | None,
    solver_stage: SolverStage | None = None,
    severity: Severity | None = None,
    suggested_action: str | None = None,
) -> DiagnosticEvent:
    if element_id is not None and not element_id:
        raise _adapter_error(
            code="ADAPTER_CONTEXT_INVALID",
            message="element_id must be non-empty when provided",
            issue=issue,
        )

    catalog_entry = CANONICAL_DIAGNOSTIC_CATALOG.get(issue.code)
    resolved_severity = (
        severity
        if severity is not None
        else (catalog_entry.severity if catalog_entry is not None else Severity.ERROR)
    )
    resolved_stage = (
        solver_stage
        if solver_stage is not None
        else (catalog_entry.solver_stage if catalog_entry is not None else SolverStage.ASSEMBLE)
    )
    resolved_action = (
        suggested_action
        if suggested_action is not None
        else (
            catalog_entry.suggested_action
            if catalog_entry is not None
            else _DEFAULT_SUGGESTED_ACTION
        )
    )
    if not resolved_action:
        raise _adapter_error(
            code="ADAPTER_SUGGESTED_ACTION_INVALID",
            message="suggested_action must be non-empty",
            issue=issue,
        )

    issue_context = _normalized_issue_context(issue.context)
    event_element_id = _event_element_id(element_id=element_id, issue_context=issue_context)
    node_context = _node_context(issue_context)
    port_context = _port_context(issue_context)
    if event_element_id is None and node_context is None and port_context is None:
        raise _adapter_error(
            code="ADAPTER_CONTEXT_MISSING",
            message="mapped diagnostic requires element/node/port context",
            issue=issue,
        )

    witness: dict[str, object] = {"issue_code": issue.code}
    if issue_context:
        witness["validation_context"] = issue_context

    return DiagnosticEvent(
        code=issue.code,
        severity=resolved_severity,
        message=issue.message,
        suggested_action=resolved_action,
        solver_stage=resolved_stage,
        element_id=event_element_id,
        node_context=node_context,
        port_context=port_context,
        witness=witness,
    )


def adapt_validation_issues(
    issues: Sequence[ValidationIssue],
    *,
    element_id: str | None,
    solver_stage: SolverStage | None = None,
    severity: Severity | None = None,
    suggested_action: str | None = None,
) -> tuple[DiagnosticEvent, ...]:
    mapped = [
        adapt_validation_issue(
            issue,
            element_id=element_id,
            solver_stage=solver_stage,
            severity=severity,
            suggested_action=suggested_action,
        )
        for issue in issues
    ]
    ordered = sort_diagnostics(mapped)
    return tuple(ordered)


def _normalized_issue_context(context: Mapping[str, object] | None) -> dict[str, object] | None:
    if context is None:
        return None
    return {key: context[key] for key in sorted(context)}


def _event_element_id(
    *, element_id: str | None, issue_context: Mapping[str, object] | None
) -> str | None:
    if element_id is not None:
        return element_id
    if issue_context is None:
        return None
    raw = issue_context.get("element_id")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _node_context(issue_context: Mapping[str, object] | None) -> NodeContext | None:
    if issue_context is None:
        return None
    raw = issue_context.get("node_id")
    if isinstance(raw, str) and raw:
        return NodeContext(node_id=raw)
    return None


def _port_context(issue_context: Mapping[str, object] | None) -> PortContext | None:
    if issue_context is None:
        return None
    raw = issue_context.get("port_id")
    if isinstance(raw, str) and raw:
        return PortContext(port_id=raw)
    return None


def _adapter_error(
    *, code: str, message: str, issue: ValidationIssue
) -> ValidationIssueAdapterError:
    detail = ValidationIssueAdapterErrorDetail(
        code=code,
        message=message,
        issue_code=issue.code,
        witness={"issue_code": issue.code},
    )
    return ValidationIssueAdapterError(detail)

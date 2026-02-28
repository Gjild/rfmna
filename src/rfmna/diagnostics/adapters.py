from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from rfmna.elements.base import ValidationIssue

from .catalog import CANONICAL_DIAGNOSTIC_CATALOG
from .models import DiagnosticEvent, NodeContext, PortContext, Severity, SolverStage
from .sort import sort_diagnostics

_DEFAULT_SUGGESTED_ACTION = "resolve element validation issue and retry"
_WITNESS_UNSET: Final[object] = object()


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


def build_diagnostic_event(  # noqa: PLR0913
    *,
    code: str,
    message: str,
    element_id: str | None = None,
    node_id: str | None = None,
    port_id: str | None = None,
    frequency_hz: float | None = None,
    frequency_index: int | None = None,
    sweep_index: int | None = None,
    witness: object | None = None,
    severity: Severity | None = None,
    solver_stage: SolverStage | None = None,
    suggested_action: str | None = None,
) -> DiagnosticEvent:
    if not code:
        raise ValueError("diagnostic code must be non-empty")
    if not message:
        raise ValueError("diagnostic message must be non-empty")
    if element_id is not None and not element_id:
        raise ValueError("diagnostic element_id must be non-empty when provided")
    if node_id is not None and not node_id:
        raise ValueError("diagnostic node_id must be non-empty when provided")
    if port_id is not None and not port_id:
        raise ValueError("diagnostic port_id must be non-empty when provided")

    catalog_entry = CANONICAL_DIAGNOSTIC_CATALOG.get(code)
    resolved_severity = (
        severity
        if severity is not None
        else _require_catalog_field(
            code=code,
            field_name="severity",
            value=(None if catalog_entry is None else catalog_entry.severity),
        )
    )
    resolved_stage = (
        solver_stage
        if solver_stage is not None
        else _require_catalog_field(
            code=code,
            field_name="solver_stage",
            value=(None if catalog_entry is None else catalog_entry.solver_stage),
        )
    )
    resolved_action = (
        suggested_action
        if suggested_action is not None
        else _require_catalog_field(
            code=code,
            field_name="suggested_action",
            value=(None if catalog_entry is None else catalog_entry.suggested_action),
        )
    )
    if not resolved_action:
        raise ValueError("diagnostic suggested_action must be non-empty")

    return DiagnosticEvent(
        code=code,
        severity=resolved_severity,
        message=message,
        suggested_action=resolved_action,
        solver_stage=resolved_stage,
        element_id=element_id,
        node_context=NodeContext(node_id=node_id) if node_id is not None else None,
        port_context=PortContext(port_id=port_id) if port_id is not None else None,
        frequency_hz=frequency_hz,
        frequency_index=frequency_index,
        sweep_index=sweep_index,
        witness=witness,
    )


def remap_diagnostic_event(  # noqa: PLR0913
    event: DiagnosticEvent,
    *,
    element_id: str | None = None,
    node_id: str | None = None,
    port_id: str | None = None,
    frequency_hz: float | None = None,
    frequency_index: int | None = None,
    sweep_index: int | None = None,
    witness: object = _WITNESS_UNSET,
) -> DiagnosticEvent:
    return build_diagnostic_event(
        code=event.code,
        severity=event.severity,
        message=event.message,
        suggested_action=event.suggested_action,
        solver_stage=event.solver_stage,
        element_id=(event.element_id if element_id is None else element_id),
        node_id=(
            event.node_context.node_id
            if node_id is None and event.node_context is not None
            else node_id
        ),
        port_id=(
            event.port_context.port_id
            if port_id is None and event.port_context is not None
            else port_id
        ),
        frequency_hz=(event.frequency_hz if frequency_hz is None else frequency_hz),
        frequency_index=(event.frequency_index if frequency_index is None else frequency_index),
        sweep_index=(event.sweep_index if sweep_index is None else sweep_index),
        witness=(event.witness if witness is _WITNESS_UNSET else witness),
    )


def prefixed_witness(
    *,
    prefix: str,
    payload: object | None,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not prefix:
        raise ValueError("witness prefix must be non-empty")
    witness: dict[str, object] = {prefix: payload}
    if extras is None:
        return witness
    for key in sorted(extras):
        witness[key] = extras[key]
    return witness


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
    node_id = _node_id(issue_context)
    port_id = _port_id(issue_context)
    if event_element_id is None and node_id is None and port_id is None:
        raise _adapter_error(
            code="ADAPTER_CONTEXT_MISSING",
            message="mapped diagnostic requires element/node/port context",
            issue=issue,
        )

    witness: dict[str, object] = {"issue_code": issue.code}
    if issue_context:
        witness["validation_context"] = issue_context

    return build_diagnostic_event(
        code=issue.code,
        severity=resolved_severity,
        message=issue.message,
        suggested_action=resolved_action,
        solver_stage=resolved_stage,
        element_id=event_element_id,
        node_id=node_id,
        port_id=port_id,
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


def _node_id(issue_context: Mapping[str, object] | None) -> str | None:
    if issue_context is None:
        return None
    raw = issue_context.get("node_id")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _port_id(issue_context: Mapping[str, object] | None) -> str | None:
    if issue_context is None:
        return None
    raw = issue_context.get("port_id")
    if isinstance(raw, str) and raw:
        return raw
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


def _require_catalog_field[T](*, code: str, field_name: str, value: T | None) -> T:
    if value is None:
        raise ValueError(
            f"diagnostic code '{code}' is not in canonical catalog; explicit {field_name} is required"
        )
    return value

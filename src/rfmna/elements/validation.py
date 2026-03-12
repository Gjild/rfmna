from __future__ import annotations

from dataclasses import dataclass, field

from rfmna.ir import CanonicalIR, IRElement, IRNode
from rfmna.ir.models import canonicalize_element_kind

from .base import StampContext, ValidationIssue
from .factory import build_stamps_from_normalized_models
from .factory_models import normalize_canonical_ir_for_factory


@dataclass(frozen=True, slots=True)
class _ValidationIndexer:
    node_ids: tuple[str, ...]
    reference_node: str
    aux_ids: tuple[str, ...]
    _node_positions: dict[str, int] = field(init=False, repr=False)
    _aux_positions: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        node_positions = {
            node_id: index
            for index, node_id in enumerate(
                node_id
                for node_id in self.node_ids
                if node_id != self.reference_node
            )
        }
        aux_positions = {
            aux_id: len(node_positions) + index
            for index, aux_id in enumerate(self.aux_ids)
        }
        object.__setattr__(self, "_node_positions", node_positions)
        object.__setattr__(self, "_aux_positions", aux_positions)

    def node_index(self, node_id: str) -> int | None:
        return self._node_positions.get(node_id)

    def aux_index(self, aux_id: str) -> int:
        return self._aux_positions[aux_id]


class ResolvedElementValidationError(ValueError):
    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(f"{issue.code}: {issue.message}")
        self.issue_code = issue.code
        self.issue_message = issue.message
        self.issue_context = issue.context


def validate_resolved_supported_element_model(element: IRElement) -> None:
    canonical_kind = canonicalize_element_kind(element.element_type)
    if canonical_kind is None:
        return
    node_ids = tuple(sorted(set(element.nodes)))
    reference_node = node_ids[0]
    ir = CanonicalIR(
        nodes=tuple(
            IRNode(node_id=node_id, is_reference=(node_id == reference_node))
            for node_id in node_ids
        ),
        aux_unknowns=(),
        elements=(element,),
        ports=(),
        resolved_params=(),
    )
    normalized = normalize_canonical_ir_for_factory(ir)
    indexing = _ValidationIndexer(
        node_ids=node_ids,
        reference_node=reference_node,
        aux_ids=normalized.allocated_aux_ids,
    )
    stamps = build_stamps_from_normalized_models(normalized.models, indexing=indexing)
    validation_ctx = StampContext(omega_rad_s=0.0, resolved_params={}, frequency_index=0)
    for stamp in stamps:
        issues = stamp.validate(validation_ctx)
        if issues:
            first_issue = issues[0]
            raise ResolvedElementValidationError(first_issue)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rfmna.ir.models import (
    CANONICAL_ELEMENT_KINDS,
    CanonicalIR,
    IRElement,
    canonicalize_element_kind,
)

UNKNOWN_KIND_CODE = "E_IR_KIND_UNKNOWN"
_AUX_REQUIRED_KINDS = frozenset(("L", "V", "VCVS"))


@dataclass(frozen=True, slots=True)
class FactoryModelNormalizationErrorDetail:
    code: str
    message: str
    element_id: str
    witness: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("normalization error code must be non-empty")
        if not self.message:
            raise ValueError("normalization error message must be non-empty")
        if not self.element_id:
            raise ValueError("normalization error element_id must be non-empty")
        canonical_witness = {key: self.witness[key] for key in sorted(self.witness)}
        object.__setattr__(self, "witness", MappingProxyType(canonical_witness))


class FactoryModelNormalizationError(ValueError):
    def __init__(self, detail: FactoryModelNormalizationErrorDetail) -> None:
        super().__init__(f"{detail.code}: {detail.message}")
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ResistorModel:
    element_id: str
    node_a: str
    node_b: str
    resistance_ohm: float
    kind: str = "R"


@dataclass(frozen=True, slots=True)
class CapacitorModel:
    element_id: str
    node_a: str
    node_b: str
    capacitance_f: float
    kind: str = "C"


@dataclass(frozen=True, slots=True)
class ConductanceModel:
    element_id: str
    node_a: str
    node_b: str
    conductance_s: float
    kind: str = "G"


@dataclass(frozen=True, slots=True)
class InductorModel:
    element_id: str
    node_a: str
    node_b: str
    aux_id: str
    inductance_h: float
    kind: str = "L"


@dataclass(frozen=True, slots=True)
class CurrentSourceModel:
    element_id: str
    node_a: str
    node_b: str
    current_a: float
    kind: str = "I"


@dataclass(frozen=True, slots=True)
class VoltageSourceModel:
    element_id: str
    node_a: str
    node_b: str
    aux_id: str
    voltage_v: float
    kind: str = "V"


@dataclass(frozen=True, slots=True)
class VCCSModel:
    element_id: str
    node_a: str
    node_b: str
    node_c: str
    node_d: str
    transconductance_s: float
    kind: str = "VCCS"


@dataclass(frozen=True, slots=True)
class VCVSModel:
    element_id: str
    node_a: str
    node_b: str
    node_c: str
    node_d: str
    aux_id: str
    gain_mu: float
    kind: str = "VCVS"


type NormalizedElementModel = (
    ResistorModel
    | CapacitorModel
    | ConductanceModel
    | InductorModel
    | CurrentSourceModel
    | VoltageSourceModel
    | VCCSModel
    | VCVSModel
)


@dataclass(frozen=True, slots=True)
class NormalizedFactoryModels:
    models: tuple[NormalizedElementModel, ...]
    allocated_aux_ids: tuple[str, ...]


def normalize_canonical_ir_for_factory(ir: CanonicalIR) -> NormalizedFactoryModels:
    models: list[NormalizedElementModel] = []
    allocated_aux_ids: list[str] = []
    for element in ir.elements:
        canonical_kind = canonicalize_element_kind(element.element_type)
        if canonical_kind is None:
            raise _unknown_kind_error(element)
        model = _normalize_element(element, canonical_kind)
        models.append(model)
        if canonical_kind in _AUX_REQUIRED_KINDS:
            allocated_aux_ids.append(_allocated_aux_id(element.element_id))
    return NormalizedFactoryModels(models=tuple(models), allocated_aux_ids=tuple(allocated_aux_ids))


def ordered_aux_ids_from_models(models: Sequence[NormalizedElementModel]) -> tuple[str, ...]:
    aux_ids: list[str] = []
    for model in models:
        if isinstance(model, InductorModel | VoltageSourceModel | VCVSModel):
            aux_ids.append(model.aux_id)
    return tuple(aux_ids)


def _normalize_element(element: IRElement, canonical_kind: str) -> NormalizedElementModel:
    if canonical_kind == "R":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model: NormalizedElementModel = ResistorModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            resistance_ohm=_require_param(element, "resistance_ohm", canonical_kind),
        )
    elif canonical_kind == "C":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model = CapacitorModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            capacitance_f=_require_param(element, "capacitance_f", canonical_kind),
        )
    elif canonical_kind == "G":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model = ConductanceModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            conductance_s=_require_param(element, "conductance_s", canonical_kind),
        )
    elif canonical_kind == "L":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model = InductorModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            aux_id=_allocated_aux_id(element.element_id),
            inductance_h=_require_param(element, "inductance_h", canonical_kind),
        )
    elif canonical_kind == "I":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model = CurrentSourceModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            current_a=_require_param(element, "current_a", canonical_kind),
        )
    elif canonical_kind == "V":
        node_a, node_b = _require_nodes(element, 2, canonical_kind)
        model = VoltageSourceModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            aux_id=_allocated_aux_id(element.element_id),
            voltage_v=_require_param(element, "voltage_v", canonical_kind),
        )
    elif canonical_kind == "VCCS":
        node_a, node_b, node_c, node_d = _require_nodes(element, 4, canonical_kind)
        model = VCCSModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            node_c=node_c,
            node_d=node_d,
            transconductance_s=_require_param(element, "transconductance_s", canonical_kind),
        )
    else:
        node_a, node_b, node_c, node_d = _require_nodes(element, 4, canonical_kind)
        model = VCVSModel(
            element_id=element.element_id,
            node_a=node_a,
            node_b=node_b,
            node_c=node_c,
            node_d=node_d,
            aux_id=_allocated_aux_id(element.element_id),
            gain_mu=_require_param(element, "gain_mu", canonical_kind),
        )
    return model


def _require_nodes(element: IRElement, expected_count: int, canonical_kind: str) -> tuple[str, ...]:
    nodes = element.nodes
    if len(nodes) != expected_count:
        raise ValueError(
            f"element '{element.element_id}' kind '{canonical_kind}' expects {expected_count} nodes, got {len(nodes)}"
        )
    return nodes


def _require_param(element: IRElement, required_name: str, canonical_kind: str) -> float:
    for name, value in element.params:
        if name == required_name:
            return float(value)
    raise ValueError(
        f"element '{element.element_id}' kind '{canonical_kind}' missing required param '{required_name}'"
    )


def _unknown_kind_error(element: IRElement) -> FactoryModelNormalizationError:
    normalized_candidate = element.element_type.strip().upper().replace("-", "_").replace(" ", "_")
    detail = FactoryModelNormalizationErrorDetail(
        code=UNKNOWN_KIND_CODE,
        message=f"unsupported element kind '{element.element_type}' for element '{element.element_id}'",
        element_id=element.element_id,
        witness={
            "element_id": element.element_id,
            "raw_kind": element.element_type,
            "normalized_candidate": normalized_candidate,
            "supported_kinds": CANONICAL_ELEMENT_KINDS,
        },
    )
    return FactoryModelNormalizationError(detail)


def _allocated_aux_id(element_id: str) -> str:
    return f"{element_id}:i"

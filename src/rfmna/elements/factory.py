from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

from rfmna.ir.models import CANONICAL_ELEMENT_KINDS, CanonicalIR

from .base import ElementStamp
from .controlled import VCCSStamp, VCVSStamp
from .factory_models import (
    CapacitorModel,
    ConductanceModel,
    CurrentSourceModel,
    InductorModel,
    NormalizedElementModel,
    ResistorModel,
    VCCSModel,
    VCVSModel,
    VoltageSourceModel,
    normalize_canonical_ir_for_factory,
)
from .inductor import InductorStamp
from .passive import CapacitorStamp, ConductanceStamp, ResistorStamp
from .sources import CurrentSourceStamp, VoltageSourceStamp

_CONSTRUCTOR_FAILURE_CODE = "FACTORY_CONSTRUCTOR_FAILED"


class FactoryIndexer(Protocol):
    def node_index(self, node_id: str) -> int | None: ...

    def aux_index(self, aux_id: str) -> int: ...


@dataclass(frozen=True, slots=True)
class FactoryConstructionErrorDetail:
    code: str
    message: str
    element_id: str
    kind: str
    witness: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("factory error code must be non-empty")
        if not self.message:
            raise ValueError("factory error message must be non-empty")
        if not self.element_id:
            raise ValueError("factory error element_id must be non-empty")
        if not self.kind:
            raise ValueError("factory error kind must be non-empty")
        canonical_witness = {key: self.witness[key] for key in sorted(self.witness)}
        object.__setattr__(self, "witness", MappingProxyType(canonical_witness))


class FactoryConstructionError(ValueError):
    def __init__(self, detail: FactoryConstructionErrorDetail) -> None:
        super().__init__(f"{detail.code}: {detail.message}")
        self.detail = detail


def _build_resistor(model: ResistorModel, indexing: FactoryIndexer) -> ElementStamp:
    return ResistorStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        resistance_ohm=model.resistance_ohm,
        node_indexer=indexing,
    )


def _build_capacitor(model: CapacitorModel, indexing: FactoryIndexer) -> ElementStamp:
    return CapacitorStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        capacitance_f=model.capacitance_f,
        node_indexer=indexing,
    )


def _build_conductance(model: ConductanceModel, indexing: FactoryIndexer) -> ElementStamp:
    return ConductanceStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        conductance_s=model.conductance_s,
        node_indexer=indexing,
    )


def _build_inductor(model: InductorModel, indexing: FactoryIndexer) -> ElementStamp:
    return InductorStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        aux_id=model.aux_id,
        inductance_h=model.inductance_h,
        indexer=indexing,
    )


def _build_current_source(model: CurrentSourceModel, indexing: FactoryIndexer) -> ElementStamp:
    return CurrentSourceStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        current_a=model.current_a,
        node_indexer=indexing,
    )


def _build_voltage_source(model: VoltageSourceModel, indexing: FactoryIndexer) -> ElementStamp:
    return VoltageSourceStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        aux_id=model.aux_id,
        voltage_v=model.voltage_v,
        indexer=indexing,
    )


def _build_vccs(model: VCCSModel, indexing: FactoryIndexer) -> ElementStamp:
    return VCCSStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        node_c=model.node_c,
        node_d=model.node_d,
        transconductance_s=model.transconductance_s,
        node_indexer=indexing,
    )


def _build_vcvs(model: VCVSModel, indexing: FactoryIndexer) -> ElementStamp:
    return VCVSStamp(
        element_id=model.element_id,
        node_a=model.node_a,
        node_b=model.node_b,
        node_c=model.node_c,
        node_d=model.node_d,
        aux_id=model.aux_id,
        gain_mu=model.gain_mu,
        indexer=indexing,
    )


FACTORY_REGISTRY_BY_KIND: Mapping[str, Callable[[object, FactoryIndexer], ElementStamp]] = (
    MappingProxyType(
        {
            "R": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_resistor),
            "C": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_capacitor),
            "G": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_conductance),
            "L": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_inductor),
            "I": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_current_source),
            "V": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_voltage_source),
            "VCCS": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_vccs),
            "VCVS": cast(Callable[[object, FactoryIndexer], ElementStamp], _build_vcvs),
        }
    )
)


def build_stamps_from_canonical_ir(
    ir: CanonicalIR, indexing: FactoryIndexer
) -> tuple[ElementStamp, ...]:
    normalized = normalize_canonical_ir_for_factory(ir)
    return build_stamps_from_normalized_models(normalized.models, indexing=indexing)


def build_stamps_from_normalized_models(
    models: Sequence[NormalizedElementModel],
    *,
    indexing: FactoryIndexer,
) -> tuple[ElementStamp, ...]:
    stamps: list[ElementStamp] = []
    for model in models:
        kind = _model_kind(model)
        element_id = _model_element_id(model)
        constructor = FACTORY_REGISTRY_BY_KIND.get(kind)
        if constructor is None:
            raise _unknown_kind_error(element_id=element_id, kind=kind)
        try:
            built = constructor(model, indexing)
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised via unit tests with concrete exception classes
            raise _constructor_error(element_id=element_id, kind=kind, exc=exc) from exc
        stamps.append(built)
    return tuple(stamps)


def _model_kind(model: object) -> str:
    value = getattr(model, "kind", None)
    if isinstance(value, str) and value:
        return value
    return "<missing-kind>"


def _model_element_id(model: object) -> str:
    value = getattr(model, "element_id", None)
    if isinstance(value, str) and value:
        return value
    return "<missing-element-id>"


def _unknown_kind_error(*, element_id: str, kind: str) -> FactoryConstructionError:
    detail = FactoryConstructionErrorDetail(
        code="E_IR_KIND_UNKNOWN",
        message=f"unsupported normalized model kind '{kind}' for element '{element_id}'",
        element_id=element_id,
        kind=kind,
        witness={
            "element_id": element_id,
            "supported_kinds": CANONICAL_ELEMENT_KINDS,
            "kind": kind,
        },
    )
    return FactoryConstructionError(detail)


def _constructor_error(*, element_id: str, kind: str, exc: Exception) -> FactoryConstructionError:
    raw_code = getattr(exc, "code", None)
    raw_message = getattr(exc, "message", str(exc))
    code = raw_code if isinstance(raw_code, str) and raw_code else _CONSTRUCTOR_FAILURE_CODE
    message = raw_message if isinstance(raw_message, str) and raw_message else str(exc)
    detail = FactoryConstructionErrorDetail(
        code=code,
        message=message,
        element_id=element_id,
        kind=kind,
        witness={
            "cause_type": type(exc).__name__,
            "element_id": element_id,
            "kind": kind,
        },
    )
    return FactoryConstructionError(detail)

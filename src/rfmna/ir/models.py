from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

CANONICAL_ELEMENT_KINDS: tuple[str, ...] = ("R", "C", "G", "L", "I", "V", "VCCS", "VCVS")

_ELEMENT_KIND_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "R": "R",
        "RES": "R",
        "RESISTOR": "R",
        "C": "C",
        "CAP": "C",
        "CAPACITOR": "C",
        "G": "G",
        "COND": "G",
        "CONDUCTANCE": "G",
        "L": "L",
        "IND": "L",
        "INDUCTOR": "L",
        "I": "I",
        "ISRC": "I",
        "CURRENT_SOURCE": "I",
        "V": "V",
        "VSRC": "V",
        "VOLTAGE_SOURCE": "V",
        "VCCS": "VCCS",
        "E": "VCVS",
        "VCVS": "VCVS",
    }
)


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def canonicalize_element_kind(raw_kind: str) -> str | None:
    token = raw_kind.strip()
    if not token:
        return None
    normalized = token.upper().replace("-", "_").replace(" ", "_")
    return _ELEMENT_KIND_ALIASES.get(normalized)


def _first_duplicate_id(values: list[str]) -> str | None:
    if not values:
        return None
    previous: str | None = None
    for value in sorted(values):
        if value == previous:
            return value
        previous = value
    return None


def _canonical_param_items(
    items: tuple[tuple[str, float], ...], owner: str
) -> tuple[tuple[str, float], ...]:
    canonical: list[tuple[str, float]] = []
    keys: list[str] = []
    for key, value in items:
        _validate_identifier("param name", key)
        if not isfinite(value):
            raise ValueError(f"{owner} parameter '{key}' must be finite")
        canonical.append((key, float(value)))
        keys.append(key)

    duplicate_key = _first_duplicate_id(keys)
    if duplicate_key is not None:
        raise ValueError(f"{owner} has duplicate parameter key: {duplicate_key}")

    return tuple(sorted(canonical, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class IRNode:
    node_id: str
    is_reference: bool = False

    def __post_init__(self) -> None:
        _validate_identifier("node_id", self.node_id)


@dataclass(frozen=True, slots=True)
class IRAuxUnknown:
    aux_id: str
    kind: str = "generic"
    owner_element_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier("aux_id", self.aux_id)
        _validate_identifier("kind", self.kind)
        if self.owner_element_id is not None:
            _validate_identifier("owner_element_id", self.owner_element_id)


@dataclass(frozen=True, slots=True)
class IRElement:
    element_id: str
    element_type: str
    nodes: tuple[str, ...]
    params: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _validate_identifier("element_id", self.element_id)
        _validate_identifier("element_type", self.element_type)
        if not self.nodes:
            raise ValueError("element nodes must be non-empty")
        for node_id in self.nodes:
            _validate_identifier("element node_id", node_id)
        object.__setattr__(
            self,
            "params",
            _canonical_param_items(self.params, f"element '{self.element_id}'"),
        )


@dataclass(frozen=True, slots=True)
class IRPort:
    port_id: str
    p_plus: str
    p_minus: str
    z0_ohm: float = 50.0

    def __post_init__(self) -> None:
        _validate_identifier("port_id", self.port_id)
        _validate_identifier("p_plus", self.p_plus)
        _validate_identifier("p_minus", self.p_minus)
        if isinstance(self.z0_ohm, complex):
            raise ValueError(
                f"E_MODEL_PORT_Z0_COMPLEX: port '{self.port_id}' z0_ohm must be real-valued"
            )
        try:
            z0_ohm = float(self.z0_ohm)
        except TypeError, ValueError:
            raise ValueError(
                f"E_MODEL_PORT_Z0_NONPOSITIVE: port '{self.port_id}' z0_ohm must be finite and > 0"
            ) from None
        if not isfinite(z0_ohm) or z0_ohm <= 0.0:
            raise ValueError(
                f"E_MODEL_PORT_Z0_NONPOSITIVE: port '{self.port_id}' z0_ohm must be finite and > 0"
            )
        object.__setattr__(self, "z0_ohm", z0_ohm)


@dataclass(frozen=True, slots=True)
class CanonicalIR:
    nodes: tuple[IRNode, ...]
    aux_unknowns: tuple[IRAuxUnknown, ...]
    elements: tuple[IRElement, ...]
    ports: tuple[IRPort, ...]
    resolved_params: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.nodes, key=lambda node: node.node_id))
        aux_unknowns = tuple(sorted(self.aux_unknowns, key=lambda aux: aux.aux_id))
        elements = tuple(sorted(self.elements, key=lambda element: element.element_id))
        ports = tuple(sorted(self.ports, key=lambda port: port.port_id))
        resolved_params = _canonical_param_items(self.resolved_params, "resolved_params")

        duplicate_node = _first_duplicate_id([node.node_id for node in nodes])
        if duplicate_node is not None:
            raise ValueError(f"duplicate node_id: {duplicate_node}")
        duplicate_aux = _first_duplicate_id([aux.aux_id for aux in aux_unknowns])
        if duplicate_aux is not None:
            raise ValueError(f"duplicate aux_id: {duplicate_aux}")
        duplicate_element = _first_duplicate_id([element.element_id for element in elements])
        if duplicate_element is not None:
            raise ValueError(f"duplicate element_id: {duplicate_element}")
        duplicate_port = _first_duplicate_id([port.port_id for port in ports])
        if duplicate_port is not None:
            raise ValueError(f"duplicate port_id: {duplicate_port}")

        reference_ids = [node.node_id for node in nodes if node.is_reference]
        if len(reference_ids) != 1:
            raise ValueError("exactly one reference node is required")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "aux_unknowns", aux_unknowns)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "resolved_params", resolved_params)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from rfmna.diagnostics.models import (
    DiagnosticEvent,
    NodeContext,
    PortContext,
    Severity,
    SolverStage,
)
from rfmna.diagnostics.sort import sort_diagnostics

VSRC_LOOP_RESIDUAL_ABS_TOL = 1e-12
_HARD_CONFLICT_ABS_TOL = 1e-12


def _validate_identifier(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _sorted_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _first_duplicate(values: tuple[str, ...]) -> str | None:
    previous: str | None = None
    for value in sorted(values):
        if value == previous:
            return value
        previous = value
    return None


@dataclass(frozen=True, slots=True)
class PortDecl:
    port_id: str
    p_plus: str
    p_minus: str

    def __post_init__(self) -> None:
        _validate_identifier("port_id", self.port_id)
        _validate_identifier("p_plus", self.p_plus)
        _validate_identifier("p_minus", self.p_minus)


@dataclass(frozen=True, slots=True)
class IdealVSource:
    source_id: str
    p_plus: str
    p_minus: str
    voltage_v: float

    def __post_init__(self) -> None:
        _validate_identifier("source_id", self.source_id)
        _validate_identifier("p_plus", self.p_plus)
        _validate_identifier("p_minus", self.p_minus)
        if not isfinite(self.voltage_v):
            raise ValueError(f"source '{self.source_id}' voltage_v must be finite")


@dataclass(frozen=True, slots=True)
class HardConstraint:
    constraint_id: str
    p_plus: str
    p_minus: str
    delta_v: float

    def __post_init__(self) -> None:
        _validate_identifier("constraint_id", self.constraint_id)
        _validate_identifier("p_plus", self.p_plus)
        _validate_identifier("p_minus", self.p_minus)
        if not isfinite(self.delta_v):
            raise ValueError(f"constraint '{self.constraint_id}' delta_v must be finite")


@dataclass(frozen=True, slots=True)
class PreflightInput:
    nodes: tuple[str, ...]
    reference_node: str | None
    ports: tuple[PortDecl, ...] = ()
    voltage_sources: tuple[IdealVSource, ...] = ()
    hard_constraints: tuple[HardConstraint, ...] = ()
    edges_for_connectivity: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for node_id in self.nodes:
            _validate_identifier("node_id", node_id)

        canonical_ports = tuple(
            sorted(self.ports, key=lambda port: (port.port_id, port.p_plus, port.p_minus))
        )
        canonical_sources = tuple(
            sorted(
                self.voltage_sources,
                key=lambda source: (source.source_id, source.p_plus, source.p_minus),
            )
        )
        canonical_constraints = tuple(
            sorted(
                self.hard_constraints,
                key=lambda constraint: (
                    constraint.constraint_id,
                    constraint.p_plus,
                    constraint.p_minus,
                ),
            )
        )
        canonical_edges = tuple(
            sorted(
                (
                    (left, right) if left <= right else (right, left)
                    for left, right in self.edges_for_connectivity
                )
            )
        )

        object.__setattr__(self, "ports", canonical_ports)
        object.__setattr__(self, "voltage_sources", canonical_sources)
        object.__setattr__(self, "hard_constraints", canonical_constraints)
        object.__setattr__(self, "edges_for_connectivity", canonical_edges)


@dataclass(frozen=True, slots=True)
class _ErrorSpec:
    code: str
    message: str
    suggested_action: str
    element_id: str | None = None
    node_id: str | None = None
    port_id: str | None = None
    witness: object | None = None


def _make_error(
    spec: _ErrorSpec,
) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=spec.code,
        severity=Severity.ERROR,
        message=spec.message,
        suggested_action=spec.suggested_action,
        solver_stage=SolverStage.PREFLIGHT,
        element_id=spec.element_id,
        node_context=NodeContext(node_id=spec.node_id) if spec.node_id is not None else None,
        port_context=PortContext(port_id=spec.port_id) if spec.port_id is not None else None,
        witness=spec.witness,
    )


def _collect_connectivity_edges(
    input_data: PreflightInput, known_nodes: set[str]
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set(input_data.edges_for_connectivity)
    for source in input_data.voltage_sources:
        if (
            source.p_plus in known_nodes
            and source.p_minus in known_nodes
            and source.p_plus != source.p_minus
        ):
            edge = (
                (source.p_plus, source.p_minus)
                if source.p_plus <= source.p_minus
                else (source.p_minus, source.p_plus)
            )
            edges.add(edge)
    for constraint in input_data.hard_constraints:
        if (
            constraint.p_plus in known_nodes
            and constraint.p_minus in known_nodes
            and constraint.p_plus != constraint.p_minus
        ):
            edge = (
                (constraint.p_plus, constraint.p_minus)
                if constraint.p_plus <= constraint.p_minus
                else (constraint.p_minus, constraint.p_plus)
            )
            edges.add(edge)
    for port in input_data.ports:
        if (
            port.p_plus in known_nodes
            and port.p_minus in known_nodes
            and port.p_plus != port.p_minus
        ):
            edge = (
                (port.p_plus, port.p_minus)
                if port.p_plus <= port.p_minus
                else (port.p_minus, port.p_plus)
            )
            edges.add(edge)
    return edges


def _reachable_nodes(
    unique_nodes: tuple[str, ...], reference_node: str, edges: set[tuple[str, str]]
) -> set[str]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in unique_nodes}
    for left, right in sorted(edges):
        adjacency[left].add(right)
        adjacency[right].add(left)

    reached: set[str] = set()
    stack: list[str] = [reference_node]
    while stack:
        node_id = stack.pop()
        if node_id in reached:
            continue
        reached.add(node_id)
        for neighbor in sorted(adjacency[node_id], reverse=True):
            if neighbor not in reached:
                stack.append(neighbor)
    return reached


def _reference_and_floating_diagnostics(input_data: PreflightInput) -> list[DiagnosticEvent]:
    diagnostics: list[DiagnosticEvent] = []
    duplicate_node = _first_duplicate(input_data.nodes)
    unique_nodes = _sorted_unique(input_data.nodes)
    known_nodes = set(unique_nodes)
    reference_node = input_data.reference_node

    if duplicate_node is not None:
        diagnostics.append(
            _make_error(
                _ErrorSpec(
                    code="E_TOPO_REFERENCE_INVALID",
                    message="duplicate node declarations are not allowed",
                    suggested_action="deduplicate node declarations before preflight",
                    element_id="reference",
                    witness={
                        "duplicate_node_id": duplicate_node,
                        "nodes": list(unique_nodes),
                    },
                )
            )
        )
        return diagnostics

    if reference_node is None or reference_node not in known_nodes:
        diagnostics.append(
            _make_error(
                _ErrorSpec(
                    code="E_TOPO_REFERENCE_INVALID",
                    message="reference node is missing or not declared",
                    suggested_action="declare exactly one valid reference node",
                    element_id="reference",
                    witness={
                        "reference_node": reference_node,
                        "nodes": list(unique_nodes),
                    },
                )
            )
        )
        return diagnostics

    edges = _collect_connectivity_edges(input_data, known_nodes)
    reached = _reachable_nodes(unique_nodes, reference_node, edges)

    floating = [node_id for node_id in unique_nodes if node_id not in reached]
    for node_id in floating:
        diagnostics.append(
            _make_error(
                _ErrorSpec(
                    code="E_TOPO_FLOATING_NODE",
                    message=f"node '{node_id}' is not reference-reachable",
                    suggested_action="connect node/component to the reference-reachable graph",
                    node_id=node_id,
                    witness={
                        "node_id": node_id,
                        "reference_node": reference_node,
                    },
                )
            )
        )
    return diagnostics


def _port_diagnostics(input_data: PreflightInput) -> list[DiagnosticEvent]:
    diagnostics: list[DiagnosticEvent] = []
    known_nodes = set(_sorted_unique(input_data.nodes))
    ports_by_id: dict[str, list[PortDecl]] = defaultdict(list)
    ports_by_orientation: dict[tuple[str, str], list[str]] = defaultdict(list)

    for port in input_data.ports:
        ports_by_id[port.port_id].append(port)
        ports_by_orientation[(port.p_plus, port.p_minus)].append(port.port_id)

    duplicate_ids = {port_id for port_id, ports in ports_by_id.items() if len(ports) > 1}
    duplicate_orientations = {
        orientation for orientation, port_ids in ports_by_orientation.items() if len(port_ids) > 1
    }

    for port in input_data.ports:
        issues: list[str] = []
        witness: dict[str, object] = {
            "p_minus": port.p_minus,
            "p_plus": port.p_plus,
        }

        unknown_nodes: list[str] = []
        if port.p_plus not in known_nodes:
            unknown_nodes.append(port.p_plus)
        if port.p_minus not in known_nodes:
            unknown_nodes.append(port.p_minus)
        if unknown_nodes:
            issues.append("unknown_node")
            witness["unknown_nodes"] = sorted(set(unknown_nodes))
        if port.p_plus == port.p_minus:
            issues.append("degenerate")
        if port.port_id in duplicate_ids:
            issues.append("duplicate_port_id")
            witness["duplicate_port_id_declarations"] = [
                {"p_minus": decl.p_minus, "p_plus": decl.p_plus}
                for decl in sorted(
                    ports_by_id[port.port_id], key=lambda entry: (entry.p_plus, entry.p_minus)
                )
            ]
        orientation = (port.p_plus, port.p_minus)
        if orientation in duplicate_orientations:
            issues.append("duplicate_orientation")
            witness["duplicate_orientation_port_ids"] = sorted(ports_by_orientation[orientation])
        if not issues:
            continue
        witness["issues"] = sorted(issues)
        diagnostics.append(
            _make_error(
                _ErrorSpec(
                    code="E_TOPO_PORT_INVALID",
                    message=f"port '{port.port_id}' has invalid declaration",
                    suggested_action=(
                        "ensure ports have unique ids, unique orientations, and distinct declared nodes"
                    ),
                    port_id=port.port_id,
                    witness=witness,
                )
            )
        )
    return diagnostics


def _vsource_loop_diagnostics(input_data: PreflightInput) -> list[DiagnosticEvent]:
    diagnostics: list[DiagnosticEvent] = []
    unique_nodes = _sorted_unique(input_data.nodes)
    known_nodes = set(unique_nodes)
    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)

    for source in input_data.voltage_sources:
        if source.p_plus not in known_nodes or source.p_minus not in known_nodes:
            continue
        adjacency[source.p_plus].append((source.p_minus, -source.voltage_v, source.source_id))
        adjacency[source.p_minus].append((source.p_plus, source.voltage_v, source.source_id))

    visited: set[str] = set()
    for root in unique_nodes:
        if root in visited or root not in adjacency:
            continue
        component_nodes: set[str] = set()
        component_source_ids: set[str] = set()
        max_abs_residual = 0.0
        inconsistent = False
        potentials: dict[str, float] = {root: 0.0}
        stack: list[str] = [root]

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            component_nodes.add(node_id)
            for neighbor, offset, source_id in sorted(
                adjacency[node_id], key=lambda item: (item[0], item[2])
            ):
                component_source_ids.add(source_id)
                component_nodes.add(neighbor)
                expected = potentials[node_id] + offset
                if neighbor not in potentials:
                    potentials[neighbor] = expected
                    if neighbor not in visited:
                        stack.append(neighbor)
                    continue
                residual = potentials[neighbor] - expected
                abs_residual = abs(residual)
                if abs_residual > VSRC_LOOP_RESIDUAL_ABS_TOL:
                    inconsistent = True
                    max_abs_residual = max(max_abs_residual, abs_residual)

        if inconsistent:
            source_ids = sorted(component_source_ids)
            diagnostics.append(
                _make_error(
                    _ErrorSpec(
                        code="E_TOPO_VSRC_LOOP_INCONSISTENT",
                        message="inconsistent ideal voltage-source loop constraints detected",
                        suggested_action="adjust source values or topology to remove contradictory loops",
                        element_id=source_ids[0] if source_ids else "vsource",
                        witness={
                            "max_abs_residual_v": max_abs_residual,
                            "nodes": sorted(component_nodes),
                            "source_ids": source_ids,
                            "tolerance_abs_v": VSRC_LOOP_RESIDUAL_ABS_TOL,
                        },
                    )
                )
            )

    return diagnostics


def _hard_constraint_diagnostics(input_data: PreflightInput) -> list[DiagnosticEvent]:
    diagnostics: list[DiagnosticEvent] = []
    known_nodes = set(_sorted_unique(input_data.nodes))
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)

    for constraint in input_data.hard_constraints:
        if constraint.p_plus not in known_nodes or constraint.p_minus not in known_nodes:
            continue
        if constraint.p_plus <= constraint.p_minus:
            key = (constraint.p_plus, constraint.p_minus)
            value = constraint.delta_v
        else:
            key = (constraint.p_minus, constraint.p_plus)
            value = -constraint.delta_v
        grouped[key].append((constraint.constraint_id, value))

    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda item: item[0])
        values = [value for _, value in entries]
        max_delta = max(values) - min(values) if values else 0.0
        if max_delta <= _HARD_CONFLICT_ABS_TOL:
            continue
        constraint_ids = [constraint_id for constraint_id, _ in entries]
        diagnostics.append(
            _make_error(
                _ErrorSpec(
                    code="E_TOPO_HARD_CONSTRAINT_CONFLICT",
                    message="conflicting hard constraints detected",
                    suggested_action="remove or reconcile contradictory hard constraints",
                    element_id=constraint_ids[0],
                    witness={
                        "constraint_ids": constraint_ids,
                        "max_delta_v": max_delta,
                        "nodes": [key[0], key[1]],
                        "values_v": sorted(values),
                    },
                )
            )
        )

    return diagnostics


def preflight_check(input_data: PreflightInput) -> tuple[DiagnosticEvent, ...]:
    diagnostics: list[DiagnosticEvent] = []
    diagnostics.extend(_reference_and_floating_diagnostics(input_data))
    diagnostics.extend(_port_diagnostics(input_data))
    diagnostics.extend(_vsource_loop_diagnostics(input_data))
    diagnostics.extend(_hard_constraint_diagnostics(input_data))
    return tuple(sort_diagnostics(diagnostics))

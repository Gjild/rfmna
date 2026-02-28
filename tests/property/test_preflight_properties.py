from __future__ import annotations

import json
from collections.abc import Mapping
from string import ascii_lowercase
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfmna.diagnostics import DiagnosticEvent
from rfmna.parser import (
    HardConstraint,
    IdealVSource,
    PortDecl,
    PreflightInput,
    preflight_check,
)

pytestmark = pytest.mark.property

_POSITIVE_PERTURBATION = st.floats(
    min_value=1e-9,
    max_value=1e-3,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _preflight_inputs(draw: st.DrawFn) -> PreflightInput:
    n_nodes = draw(st.integers(min_value=3, max_value=6))
    canonical_nodes = tuple(f"n{index}" for index in range(n_nodes))
    shuffled_nodes = tuple(draw(st.permutations(canonical_nodes)))

    connectivity_flags = draw(st.lists(st.booleans(), min_size=n_nodes - 1, max_size=n_nodes - 1))
    edges = tuple(
        (canonical_nodes[0], canonical_nodes[index])
        for index, is_connected in enumerate(connectivity_flags, start=1)
        if is_connected
    )

    include_invalid_ports = draw(st.booleans())
    ports = (
        (
            PortDecl("P1", canonical_nodes[1], canonical_nodes[2]),
            PortDecl("P2", canonical_nodes[1], canonical_nodes[2]),
            PortDecl("P3", canonical_nodes[2], canonical_nodes[2]),
        )
        if include_invalid_ports
        else (
            PortDecl("P1", canonical_nodes[1], canonical_nodes[0]),
            PortDecl("P2", canonical_nodes[2], canonical_nodes[0]),
            PortDecl("P3", canonical_nodes[1], canonical_nodes[2]),
        )
    )

    include_inconsistent_loop = draw(st.booleans())
    voltage_sources = (
        (
            IdealVSource("V1", canonical_nodes[0], canonical_nodes[1], 1.0),
            IdealVSource("V2", canonical_nodes[1], canonical_nodes[2], 1.0),
            IdealVSource(
                "V3",
                canonical_nodes[2],
                canonical_nodes[0],
                -2.0 + draw(_POSITIVE_PERTURBATION),
            ),
        )
        if include_inconsistent_loop
        else ()
    )

    include_hard_conflict = draw(st.booleans())
    hard_constraints = (
        (
            HardConstraint("H1", canonical_nodes[1], canonical_nodes[2], 1.0),
            HardConstraint(
                "H2",
                canonical_nodes[2],
                canonical_nodes[1],
                -(1.0 + draw(_POSITIVE_PERTURBATION)),
            ),
        )
        if include_hard_conflict
        else ()
    )

    return PreflightInput(
        nodes=shuffled_nodes,
        reference_node=canonical_nodes[0],
        ports=ports,
        voltage_sources=voltage_sources,
        hard_constraints=hard_constraints,
        edges_for_connectivity=tuple(reversed(edges)),
    )


@st.composite
def _valid_preflight_inputs(draw: st.DrawFn) -> PreflightInput:
    n_nodes = draw(st.integers(min_value=3, max_value=6))
    canonical_nodes = tuple(f"n{index}" for index in range(n_nodes))
    shuffled_nodes = tuple(draw(st.permutations(canonical_nodes)))
    edges = tuple((canonical_nodes[0], canonical_nodes[index]) for index in range(1, n_nodes))
    ports = (
        PortDecl("P1", canonical_nodes[1], canonical_nodes[0]),
        PortDecl("P2", canonical_nodes[2], canonical_nodes[0]),
    )

    return PreflightInput(
        nodes=shuffled_nodes,
        reference_node=canonical_nodes[0],
        ports=ports,
        edges_for_connectivity=tuple(reversed(edges)),
    )


def _remap_node_ids(value: object, node_aliases: Mapping[str, str]) -> object:
    if isinstance(value, str):
        remapped = value
        for source, target in sorted(
            node_aliases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            remapped = remapped.replace(source, target)
        return remapped
    if isinstance(value, list):
        return [_remap_node_ids(item, node_aliases) for item in value]
    if isinstance(value, tuple):
        return [_remap_node_ids(item, node_aliases) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _remap_node_ids(cast(dict[object, object], value)[key], node_aliases)
            for key in sorted(cast(dict[object, object], value), key=str)
        }
    return value


def _diag_signature(diag: DiagnosticEvent, *, node_aliases: Mapping[str, str] | None = None) -> str:
    payload = cast(dict[str, object], diag.model_dump(mode="json"))
    if node_aliases is not None:
        payload = cast(dict[str, object], _remap_node_ids(payload, node_aliases))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _rename_preflight_input(
    input_data: PreflightInput, node_aliases: Mapping[str, str]
) -> PreflightInput:
    return PreflightInput(
        nodes=tuple(node_aliases[node_id] for node_id in input_data.nodes),
        reference_node=(
            None if input_data.reference_node is None else node_aliases[input_data.reference_node]
        ),
        ports=tuple(
            PortDecl(
                port_id=port.port_id,
                p_plus=node_aliases[port.p_plus],
                p_minus=node_aliases[port.p_minus],
            )
            for port in input_data.ports
        ),
        voltage_sources=tuple(
            IdealVSource(
                source_id=source.source_id,
                p_plus=node_aliases[source.p_plus],
                p_minus=node_aliases[source.p_minus],
                voltage_v=source.voltage_v,
            )
            for source in input_data.voltage_sources
        ),
        hard_constraints=tuple(
            HardConstraint(
                constraint_id=constraint.constraint_id,
                p_plus=node_aliases[constraint.p_plus],
                p_minus=node_aliases[constraint.p_minus],
                delta_v=constraint.delta_v,
            )
            for constraint in input_data.hard_constraints
        ),
        edges_for_connectivity=tuple(
            (node_aliases[left], node_aliases[right])
            for left, right in input_data.edges_for_connectivity
        ),
    )


@given(
    input_data=_preflight_inputs(), prefix=st.text(alphabet=ascii_lowercase, min_size=1, max_size=4)
)
def test_preflight_node_relabeling_invariance(input_data: PreflightInput, prefix: str) -> None:
    node_aliases = {node_id: f"{prefix}_{node_id}" for node_id in sorted(set(input_data.nodes))}
    inverse_aliases = {value: key for key, value in node_aliases.items()}

    baseline = tuple(_diag_signature(diag) for diag in preflight_check(input_data))
    relabeled = tuple(
        _diag_signature(diag, node_aliases=inverse_aliases)
        for diag in preflight_check(_rename_preflight_input(input_data, node_aliases))
    )

    assert relabeled == baseline


@given(input_data=_preflight_inputs())
def test_preflight_input_order_permutation_invariance(input_data: PreflightInput) -> None:
    permuted = PreflightInput(
        nodes=tuple(reversed(input_data.nodes)),
        reference_node=input_data.reference_node,
        ports=tuple(reversed(input_data.ports)),
        voltage_sources=tuple(reversed(input_data.voltage_sources)),
        hard_constraints=tuple(reversed(input_data.hard_constraints)),
        edges_for_connectivity=tuple(
            (right, left) for left, right in reversed(input_data.edges_for_connectivity)
        ),
    )

    baseline = tuple(_diag_signature(diag) for diag in preflight_check(input_data))
    current = tuple(_diag_signature(diag) for diag in preflight_check(permuted))

    assert current == baseline


@given(
    input_data=_valid_preflight_inputs(),
    prefix=st.text(alphabet=ascii_lowercase, min_size=1, max_size=4),
)
def test_preflight_valid_inputs_remain_diagnostic_free_under_relabel_and_permutation(
    input_data: PreflightInput, prefix: str
) -> None:
    node_aliases = {node_id: f"{prefix}_{node_id}" for node_id in sorted(set(input_data.nodes))}

    permuted = PreflightInput(
        nodes=tuple(reversed(input_data.nodes)),
        reference_node=input_data.reference_node,
        ports=tuple(reversed(input_data.ports)),
        voltage_sources=tuple(reversed(input_data.voltage_sources)),
        hard_constraints=tuple(reversed(input_data.hard_constraints)),
        edges_for_connectivity=tuple(
            (right, left) for left, right in reversed(input_data.edges_for_connectivity)
        ),
    )

    assert preflight_check(input_data) == ()
    assert preflight_check(_rename_preflight_input(input_data, node_aliases)) == ()
    assert preflight_check(permuted) == ()

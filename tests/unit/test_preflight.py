from __future__ import annotations

import json
from random import Random

import pytest

from rfmna.diagnostics import canonical_witness_json
from rfmna.parser import (
    HardConstraint,
    IdealVSource,
    PortDecl,
    PreflightInput,
    preflight_check,
)
from rfmna.parser.preflight import VSRC_LOOP_RESIDUAL_ABS_TOL

pytestmark = pytest.mark.unit

_DUPLICATE_PORT_DECL_COUNT = 2


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_reference_invalid_missing_or_ambiguous_and_floating_deterministic() -> None:
    missing = preflight_check(
        PreflightInput(
            nodes=("0", "n1"),
            reference_node=None,
        )
    )
    assert [diag.code for diag in missing] == ["E_TOPO_REFERENCE_INVALID"]

    undeclared = preflight_check(
        PreflightInput(
            nodes=("0", "n1"),
            reference_node="gnd",
        )
    )
    assert [diag.code for diag in undeclared] == ["E_TOPO_REFERENCE_INVALID"]

    ambiguous = preflight_check(
        PreflightInput(
            nodes=("0", "0", "n1"),
            reference_node="0",
        )
    )
    assert [diag.code for diag in ambiguous] == ["E_TOPO_REFERENCE_INVALID"]

    floating_a = preflight_check(
        PreflightInput(
            nodes=("n2", "0", "n1"),
            reference_node="0",
            edges_for_connectivity=(("0", "n1"),),
        )
    )
    floating_b = preflight_check(
        PreflightInput(
            nodes=("n1", "0", "n2"),
            reference_node="0",
            edges_for_connectivity=(("n1", "0"),),
        )
    )
    assert [diag.code for diag in floating_a] == ["E_TOPO_FLOATING_NODE"]
    assert floating_a[0].node_context is not None
    assert floating_a[0].node_context.node_id == "n2"
    assert _diag_json(floating_a) == _diag_json(floating_b)


def test_port_validity_and_deterministic_ordering() -> None:
    input_data = PreflightInput(
        nodes=("0", "n1"),
        reference_node="0",
        edges_for_connectivity=(("0", "n1"),),
        ports=(
            PortDecl(port_id="P2", p_plus="n1", p_minus="n1"),
            PortDecl(port_id="P1", p_plus="n1", p_minus="n9"),
        ),
    )
    diags = preflight_check(input_data)
    assert [diag.code for diag in diags] == ["E_TOPO_PORT_INVALID", "E_TOPO_PORT_INVALID"]
    assert [diag.port_context.port_id for diag in diags if diag.port_context is not None] == [
        "P1",
        "P2",
    ]

    permuted = preflight_check(
        PreflightInput(
            nodes=("n1", "0"),
            reference_node="0",
            edges_for_connectivity=(("n1", "0"),),
            ports=(
                PortDecl(port_id="P1", p_plus="n1", p_minus="n9"),
                PortDecl(port_id="P2", p_plus="n1", p_minus="n1"),
            ),
        )
    )
    assert _diag_json(diags) == _diag_json(permuted)


def test_port_duplicate_id_and_orientation_are_rejected_as_errors() -> None:
    diagnostics = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            edges_for_connectivity=(("0", "n1"), ("0", "n2")),
            ports=(
                PortDecl(port_id="P1", p_plus="n1", p_minus="0"),
                PortDecl(port_id="P1", p_plus="n2", p_minus="0"),
                PortDecl(port_id="P2", p_plus="n1", p_minus="0"),
            ),
        )
    )

    assert [diag.code for diag in diagnostics] == [
        "E_TOPO_PORT_INVALID",
        "E_TOPO_PORT_INVALID",
        "E_TOPO_PORT_INVALID",
    ]
    assert [diag.severity.value for diag in diagnostics] == ["error", "error", "error"]

    issues_by_port_decl: dict[tuple[str, str, str], list[str]] = {}
    witness_json_by_port_decl: dict[tuple[str, str, str], str] = {}
    duplicate_id_witnesses: list[dict[str, object]] = []
    for diag in diagnostics:
        assert diag.port_context is not None
        assert diag.witness is not None
        witness = diag.witness
        assert isinstance(witness, dict)
        key = (diag.port_context.port_id, str(witness["p_plus"]), str(witness["p_minus"]))
        issues_by_port_decl[key] = list(witness["issues"])
        witness_json_by_port_decl[key] = canonical_witness_json(witness)
        if diag.port_context.port_id == "P1":
            duplicate_id_witnesses.append(witness)

    assert issues_by_port_decl[("P1", "n1", "0")] == ["duplicate_orientation", "duplicate_port_id"]
    assert issues_by_port_decl[("P1", "n2", "0")] == ["duplicate_port_id"]
    assert issues_by_port_decl[("P2", "n1", "0")] == ["duplicate_orientation"]
    assert witness_json_by_port_decl[("P2", "n1", "0")] == (
        '{"duplicate_orientation_port_ids":["P1","P2"],"issues":["duplicate_orientation"],"p_minus":"0","p_plus":"n1"}'
    )

    assert len(duplicate_id_witnesses) == _DUPLICATE_PORT_DECL_COUNT
    for witness in duplicate_id_witnesses:
        assert witness["duplicate_port_id_declarations"] == [
            {"p_minus": "0", "p_plus": "n1"},
            {"p_minus": "0", "p_plus": "n2"},
        ]
        if "duplicate_orientation" in witness["issues"]:
            assert witness["duplicate_orientation_port_ids"] == ["P1", "P2"]


def test_port_invalid_diagnostics_are_permutation_stable_with_witness_ordering() -> None:
    base = PreflightInput(
        nodes=("0", "n1", "n2"),
        reference_node="0",
        edges_for_connectivity=(("0", "n1"), ("0", "n2")),
        ports=(
            PortDecl(port_id="P3", p_plus="n1", p_minus="n1"),
            PortDecl(port_id="P2", p_plus="n1", p_minus="n9"),
            PortDecl(port_id="P1", p_plus="n2", p_minus="0"),
            PortDecl(port_id="P1", p_plus="n1", p_minus="0"),
            PortDecl(port_id="P4", p_plus="n1", p_minus="0"),
        ),
    )
    baseline = preflight_check(base)
    baseline_json = _diag_json(baseline)
    baseline_witness = [canonical_witness_json(diag.witness) for diag in baseline]

    rng = Random(0)
    for _ in range(50):
        nodes = list(base.nodes)
        ports = list(base.ports)
        rng.shuffle(nodes)
        rng.shuffle(ports)
        permuted = PreflightInput(
            nodes=tuple(nodes),
            reference_node=base.reference_node,
            edges_for_connectivity=base.edges_for_connectivity,
            ports=tuple(ports),
        )
        current = preflight_check(permuted)
        assert _diag_json(current) == baseline_json
        assert [canonical_witness_json(diag.witness) for diag in current] == baseline_witness


def test_vsource_loop_consistency_and_tolerance_boundaries() -> None:
    consistent = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            voltage_sources=(
                IdealVSource("V1", "0", "n1", 1.0),
                IdealVSource("V2", "n1", "n2", 1.0),
                IdealVSource("V3", "0", "n2", 2.0),
            ),
        )
    )
    assert "E_TOPO_VSRC_LOOP_INCONSISTENT" not in {diag.code for diag in consistent}

    below_tol = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            voltage_sources=(
                IdealVSource("V1", "0", "n1", 1.0),
                IdealVSource("V2", "n1", "n2", 1.0),
                IdealVSource("V3", "n2", "0", -2.0 + VSRC_LOOP_RESIDUAL_ABS_TOL * 0.5),
            ),
        )
    )
    assert "E_TOPO_VSRC_LOOP_INCONSISTENT" not in {diag.code for diag in below_tol}

    above_tol = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            voltage_sources=(
                IdealVSource("V1", "0", "n1", 1.0),
                IdealVSource("V2", "n1", "n2", 1.0),
                IdealVSource("V3", "n2", "0", -2.0 + VSRC_LOOP_RESIDUAL_ABS_TOL * 2.0),
            ),
        )
    )
    loop_diags = [diag for diag in above_tol if diag.code == "E_TOPO_VSRC_LOOP_INCONSISTENT"]
    assert len(loop_diags) == 1
    witness = loop_diags[0].witness
    assert isinstance(witness, dict)
    assert witness["source_ids"] == ["V1", "V2", "V3"]
    assert witness["nodes"] == ["0", "n1", "n2"]
    assert witness["max_abs_residual_v"] > VSRC_LOOP_RESIDUAL_ABS_TOL


def test_vsource_and_hard_conflict_diagnostics_are_permutation_stable() -> None:
    base = PreflightInput(
        nodes=("0", "n1", "n2"),
        reference_node="0",
        voltage_sources=(
            IdealVSource("V2", "n1", "n2", 1.0),
            IdealVSource("V1", "0", "n1", 1.0),
            IdealVSource("V3", "n2", "0", -1.5),
        ),
        hard_constraints=(
            HardConstraint("H2", "n2", "n1", -3.0),
            HardConstraint("H1", "n1", "n2", 1.0),
        ),
    )
    baseline = _diag_json(preflight_check(base))
    rng = Random(0)

    for _ in range(50):
        nodes = list(base.nodes)
        ports = list(base.ports)
        sources = list(base.voltage_sources)
        constraints = list(base.hard_constraints)
        rng.shuffle(nodes)
        rng.shuffle(ports)
        rng.shuffle(sources)
        rng.shuffle(constraints)
        permuted = PreflightInput(
            nodes=tuple(nodes),
            reference_node=base.reference_node,
            ports=tuple(ports),
            voltage_sources=tuple(sources),
            hard_constraints=tuple(constraints),
        )
        assert _diag_json(preflight_check(permuted)) == baseline


def test_hard_constraint_conflict_and_non_conflict() -> None:
    conflict = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            edges_for_connectivity=(("0", "n1"), ("n1", "n2")),
            hard_constraints=(
                HardConstraint("H1", "n1", "n2", 1.0),
                HardConstraint("H2", "n2", "n1", -3.0),
            ),
        )
    )
    codes = [diag.code for diag in conflict]
    assert codes == ["E_TOPO_HARD_CONSTRAINT_CONFLICT"]
    hard = conflict[0]
    assert hard.witness is not None

    non_conflict = preflight_check(
        PreflightInput(
            nodes=("0", "n1", "n2"),
            reference_node="0",
            edges_for_connectivity=(("0", "n1"), ("n1", "n2")),
            hard_constraints=(
                HardConstraint("H1", "n1", "n2", 1.0),
                HardConstraint("H2", "n2", "n1", -1.0),
            ),
        )
    )
    assert "E_TOPO_HARD_CONSTRAINT_CONFLICT" not in {diag.code for diag in non_conflict}


def test_preflight_repeated_runs_are_identical() -> None:
    input_data = PreflightInput(
        nodes=("0", "n1", "n2", "n3"),
        reference_node="0",
        edges_for_connectivity=(("0", "n1"), ("n1", "n2")),
        ports=(PortDecl("P1", "n1", "n1"),),
        voltage_sources=(
            IdealVSource("V1", "0", "n1", 1.0),
            IdealVSource("V2", "n1", "n2", 2.0),
            IdealVSource("V3", "n2", "0", -4.0),
        ),
        hard_constraints=(
            HardConstraint("H1", "n1", "n2", 1.0),
            HardConstraint("H2", "n2", "n1", -3.0),
        ),
    )
    baseline = preflight_check(input_data)
    baseline_json = _diag_json(baseline)

    for _ in range(30):
        current = preflight_check(input_data)
        assert current == baseline
        assert _diag_json(current) == baseline_json

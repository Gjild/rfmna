from __future__ import annotations

import pytest

from rfmna.ir import (
    CanonicalIR,
    IRAuxUnknown,
    IRElement,
    IRNode,
    IRPort,
    canonical_ir_json,
    hash_canonical_ir,
    serialize_canonical_ir,
)

pytestmark = pytest.mark.conformance


def _build_ir_variant_one() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n2"),
            IRNode("0", is_reference=True),
            IRNode("n1"),
        ),
        aux_unknowns=(
            IRAuxUnknown(aux_id="ix2", kind="branch", owner_element_id="V2"),
            IRAuxUnknown(aux_id="ix1", kind="branch", owner_element_id="V1"),
        ),
        elements=(
            IRElement(
                element_id="R2",
                element_type="R",
                nodes=("n2", "0"),
                params=(("r", 200.0), ("tc", 1.0)),
            ),
            IRElement(
                element_id="R1",
                element_type="R",
                nodes=("n1", "0"),
                params=(("tc", 1.0), ("r", 100.0)),
            ),
        ),
        ports=(
            IRPort(port_id="P2", p_plus="n2", p_minus="0", z0_ohm=50.0),
            IRPort(port_id="P1", p_plus="n1", p_minus="0"),
        ),
        resolved_params=(("beta", 2.0), ("alpha", 1.0)),
    )


def _build_ir_variant_two() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n1"),
            IRNode("n2"),
            IRNode("0", is_reference=True),
        ),
        aux_unknowns=(
            IRAuxUnknown(aux_id="ix1", kind="branch", owner_element_id="V1"),
            IRAuxUnknown(aux_id="ix2", kind="branch", owner_element_id="V2"),
        ),
        elements=(
            IRElement(
                element_id="R1",
                element_type="R",
                nodes=("n1", "0"),
                params=(("r", 100.0), ("tc", 1.0)),
            ),
            IRElement(
                element_id="R2",
                element_type="R",
                nodes=("n2", "0"),
                params=(("tc", 1.0), ("r", 200.0)),
            ),
        ),
        ports=(
            IRPort(port_id="P1", p_plus="n1", p_minus="0", z0_ohm=50.0),
            IRPort(port_id="P2", p_plus="n2", p_minus="0"),
        ),
        resolved_params=(("alpha", 1.0), ("beta", 2.0)),
    )


def test_ir_canonical_serialization_and_hash_are_permutation_invariant() -> None:
    left = _build_ir_variant_one()
    right = _build_ir_variant_two()

    assert canonical_ir_json(left) == canonical_ir_json(right)
    assert serialize_canonical_ir(left) == serialize_canonical_ir(right)
    assert hash_canonical_ir(left) == hash_canonical_ir(right)


def test_ir_hash_changes_when_canonical_semantics_change() -> None:
    baseline = _build_ir_variant_one()
    changed = CanonicalIR(
        nodes=baseline.nodes,
        aux_unknowns=baseline.aux_unknowns,
        elements=(
            IRElement(
                element_id="R1",
                element_type="R",
                nodes=("n1", "0"),
                params=(("r", 150.0), ("tc", 1.0)),
            ),
            IRElement(
                element_id="R2",
                element_type="R",
                nodes=("n2", "0"),
                params=(("r", 200.0), ("tc", 1.0)),
            ),
        ),
        ports=baseline.ports,
        resolved_params=baseline.resolved_params,
    )

    assert hash_canonical_ir(changed) != hash_canonical_ir(baseline)

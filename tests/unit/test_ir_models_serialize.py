from __future__ import annotations

from dataclasses import FrozenInstanceError

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

pytestmark = pytest.mark.unit

_Z0_DEFAULT = 50.0
_REPEAT_COUNT = 20


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
            IRPort(port_id="P2", p_plus="n2", p_minus="0", z0_ohm=_Z0_DEFAULT),
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
            IRPort(port_id="P1", p_plus="n1", p_minus="0", z0_ohm=_Z0_DEFAULT),
            IRPort(port_id="P2", p_plus="n2", p_minus="0"),
        ),
        resolved_params=(("alpha", 1.0), ("beta", 2.0)),
    )


def test_models_are_immutable() -> None:
    ir = _build_ir_variant_one()

    with pytest.raises(FrozenInstanceError):
        ir.nodes = ()

    with pytest.raises(FrozenInstanceError):
        ir.nodes[0].node_id = "x"


def test_semantically_equivalent_inputs_produce_identical_bytes_and_hash() -> None:
    left = _build_ir_variant_one()
    right = _build_ir_variant_two()

    assert serialize_canonical_ir(left) == serialize_canonical_ir(right)
    assert hash_canonical_ir(left) == hash_canonical_ir(right)


def test_duplicate_ids_are_rejected_deterministically() -> None:
    with pytest.raises(ValueError, match="duplicate node_id: n1"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("n1"), IRNode("n1")),
            aux_unknowns=(),
            elements=(),
            ports=(),
            resolved_params=(),
        )

    with pytest.raises(ValueError, match="duplicate element_id: R1"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("n1")),
            aux_unknowns=(),
            elements=(
                IRElement("R1", "R", ("n1", "0"), (("r", 1.0),)),
                IRElement("R1", "R", ("n1", "0"), (("r", 2.0),)),
            ),
            ports=(),
            resolved_params=(),
        )

    with pytest.raises(ValueError, match="duplicate port_id: P1"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("n1")),
            aux_unknowns=(),
            elements=(),
            ports=(
                IRPort("P1", "n1", "0"),
                IRPort("P1", "n1", "0"),
            ),
            resolved_params=(),
        )

    with pytest.raises(ValueError, match="duplicate aux_id: ix"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("n1")),
            aux_unknowns=(
                IRAuxUnknown("ix"),
                IRAuxUnknown("ix"),
            ),
            elements=(),
            ports=(),
            resolved_params=(),
        )


def test_reference_node_policy_is_explicit_and_deterministic() -> None:
    with pytest.raises(ValueError, match="exactly one reference node is required"):
        CanonicalIR(
            nodes=(IRNode("n1"), IRNode("n2")),
            aux_unknowns=(),
            elements=(),
            ports=(),
            resolved_params=(),
        )

    with pytest.raises(ValueError, match="exactly one reference node is required"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("gnd", is_reference=True)),
            aux_unknowns=(),
            elements=(),
            ports=(),
            resolved_params=(),
        )


def test_port_payload_stability_and_default_z0() -> None:
    left = CanonicalIR(
        nodes=(IRNode("0", is_reference=True), IRNode("n1")),
        aux_unknowns=(),
        elements=(),
        ports=(IRPort("P1", "n1", "0"),),
        resolved_params=(),
    )
    right = CanonicalIR(
        nodes=(IRNode("n1"), IRNode("0", is_reference=True)),
        aux_unknowns=(),
        elements=(),
        ports=(IRPort("P1", "n1", "0", z0_ohm=_Z0_DEFAULT),),
        resolved_params=(),
    )

    assert canonical_ir_json(left) == canonical_ir_json(right)


def test_parameter_payload_is_canonicalized() -> None:
    ir = CanonicalIR(
        nodes=(IRNode("0", is_reference=True), IRNode("n1")),
        aux_unknowns=(),
        elements=(
            IRElement(
                element_id="R1",
                element_type="R",
                nodes=("n1", "0"),
                params=(("z", 2.0), ("a", 1.0)),
            ),
        ),
        ports=(),
        resolved_params=(("zeta", 2.0), ("alpha", 1.0)),
    )

    assert ir.elements[0].params == (("a", 1.0), ("z", 2.0))
    assert ir.resolved_params == (("alpha", 1.0), ("zeta", 2.0))


def test_nonfinite_numeric_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="E_MODEL_PORT_Z0_NONPOSITIVE"):
        IRPort("P1", "n1", "0", z0_ohm=float("inf"))
    with pytest.raises(ValueError, match="E_MODEL_PORT_Z0_NONPOSITIVE"):
        IRPort("P1", "n1", "0", z0_ohm=0.0)
    with pytest.raises(ValueError, match="E_MODEL_PORT_Z0_NONPOSITIVE"):
        IRPort("P1", "n1", "0", z0_ohm=-10.0)
    with pytest.raises(ValueError, match="E_MODEL_PORT_Z0_COMPLEX"):
        IRPort("P1", "n1", "0", z0_ohm=complex(50.0, 0.0))

    with pytest.raises(ValueError, match="parameter 'r' must be finite"):
        IRElement("R1", "R", ("n1", "0"), (("r", float("nan")),))

    with pytest.raises(ValueError, match="resolved_params parameter 'alpha' must be finite"):
        CanonicalIR(
            nodes=(IRNode("0", is_reference=True), IRNode("n1")),
            aux_unknowns=(),
            elements=(),
            ports=(),
            resolved_params=(("alpha", float("inf")),),
        )


def test_serialization_reproducibility_loop() -> None:
    ir = _build_ir_variant_one()
    baseline = serialize_canonical_ir(ir)

    for _ in range(_REPEAT_COUNT):
        assert serialize_canonical_ir(ir) == baseline

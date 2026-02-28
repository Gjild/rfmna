from __future__ import annotations

import pytest

from rfmna.elements.factory_models import (
    UNKNOWN_KIND_CODE,
    FactoryModelNormalizationError,
    NormalizedFactoryModels,
    normalize_canonical_ir_for_factory,
    ordered_aux_ids_from_models,
)
from rfmna.ir import (
    CanonicalIR,
    IRAuxUnknown,
    IRElement,
    IRNode,
    canonical_ir_json,
    hash_canonical_ir,
)

pytestmark = pytest.mark.unit

EXPECTED_PHASE0_HASH = "82b6b316fc5a37a8b5fc434370804c8303bb466b19331d4d524b394b90db5448"
EXPECTED_PHASE0_JSON = (
    '{"aux_unknowns":[{"aux_id":"L1:i","kind":"branch_current","owner_element_id":"L1"},'
    '{"aux_id":"V1:i","kind":"branch_current","owner_element_id":"V1"}],"elements":[{"element_id":"C1",'
    '"element_type":"C","nodes":["n1","n2"],"params":[["capacitance_f",1e-12]]},{"element_id":"G1",'
    '"element_type":"G","nodes":["n2","0"],"params":[["conductance_s",0.02]]},{"element_id":"I1",'
    '"element_type":"I","nodes":["n1","0"],"params":[["current_a",0.01]]},{"element_id":"L1",'
    '"element_type":"L","nodes":["n1","0"],"params":[["inductance_h",2e-09]]},{"element_id":"R1",'
    '"element_type":"R","nodes":["n1","0"],"params":[["resistance_ohm",100.0]]},{"element_id":"V1",'
    '"element_type":"V","nodes":["n2","0"],"params":[["voltage_v",1.25]]}],"nodes":[{"is_reference":true,'
    '"node_id":"0"},{"is_reference":false,"node_id":"n1"},{"is_reference":false,"node_id":"n2"}],'
    '"ports":[],"resolved_params":[["alpha",1.0],["beta",2.0]]}'
)
EXPECTED_AUX_IDS = ("E1:i", "L1:i", "V1:i")
EXPECTED_AUX_IDS_MIXED = ("E2:i", "L1:i", "V2:i")
EXPECTED_UNKNOWN_KIND_WITNESS_KEYS = (
    "element_id",
    "normalized_candidate",
    "raw_kind",
    "supported_kinds",
)


def _phase1_ir_variant_one() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n2"),
            IRNode("0", is_reference=True),
            IRNode("n1"),
        ),
        aux_unknowns=(
            IRAuxUnknown("any:a"),
            IRAuxUnknown("zzz:z"),
        ),
        elements=(
            IRElement("V1", "vsrc", ("n1", "0"), (("voltage_v", 5.0),)),
            IRElement("R1", "resistor", ("n1", "0"), (("resistance_ohm", 100.0),)),
            IRElement("GM1", "vccs", ("n1", "0", "n2", "0"), (("transconductance_s", 0.5),)),
            IRElement("E1", "e", ("n2", "0", "n1", "0"), (("gain_mu", 2.0),)),
            IRElement("L1", "inductor", ("n1", "0"), (("inductance_h", 2e-9),)),
            IRElement("I1", "isrc", ("n2", "0"), (("current_a", 0.01),)),
            IRElement("C1", "cap", ("n1", "n2"), (("capacitance_f", 1e-12),)),
            IRElement("G1", "conductance", ("n2", "0"), (("conductance_s", 0.02),)),
        ),
        ports=(),
        resolved_params=(("beta", 2.0), ("alpha", 1.0)),
    )


def _phase1_ir_variant_two() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n1"),
            IRNode("n2"),
            IRNode("0", is_reference=True),
        ),
        aux_unknowns=(),
        elements=(
            IRElement("C1", "C", ("n1", "n2"), (("capacitance_f", 1e-12),)),
            IRElement("E1", "VCVS", ("n2", "0", "n1", "0"), (("gain_mu", 2.0),)),
            IRElement("G1", "G", ("n2", "0"), (("conductance_s", 0.02),)),
            IRElement("GM1", "VCCS", ("n1", "0", "n2", "0"), (("transconductance_s", 0.5),)),
            IRElement("I1", "I", ("n2", "0"), (("current_a", 0.01),)),
            IRElement("L1", "L", ("n1", "0"), (("inductance_h", 2e-9),)),
            IRElement("R1", "R", ("n1", "0"), (("resistance_ohm", 100.0),)),
            IRElement("V1", "V", ("n1", "0"), (("voltage_v", 5.0),)),
        ),
        ports=(),
        resolved_params=(("alpha", 1.0), ("beta", 2.0)),
    )


def _phase0_ir_regression_fixture() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n2"),
            IRNode("0", is_reference=True),
            IRNode("n1"),
        ),
        aux_unknowns=(
            IRAuxUnknown(aux_id="L1:i", kind="branch_current", owner_element_id="L1"),
            IRAuxUnknown(aux_id="V1:i", kind="branch_current", owner_element_id="V1"),
        ),
        elements=(
            IRElement("V1", "V", ("n2", "0"), (("voltage_v", 1.25),)),
            IRElement("R1", "R", ("n1", "0"), (("resistance_ohm", 100.0),)),
            IRElement("I1", "I", ("n1", "0"), (("current_a", 0.01),)),
            IRElement("C1", "C", ("n1", "n2"), (("capacitance_f", 1e-12),)),
            IRElement("G1", "G", ("n2", "0"), (("conductance_s", 0.02),)),
            IRElement("L1", "L", ("n1", "0"), (("inductance_h", 2e-9),)),
        ),
        ports=(),
        resolved_params=(("beta", 2.0), ("alpha", 1.0)),
    )


def test_equivalent_ir_permutations_produce_identical_normalized_models() -> None:
    left = normalize_canonical_ir_for_factory(_phase1_ir_variant_one())
    right = normalize_canonical_ir_for_factory(_phase1_ir_variant_two())

    assert left.models == right.models
    assert left.allocated_aux_ids == right.allocated_aux_ids == EXPECTED_AUX_IDS


def test_aux_allocation_is_deterministic_for_l_v_vcvs() -> None:
    ir = CanonicalIR(
        nodes=(IRNode("0", is_reference=True), IRNode("n1"), IRNode("n2")),
        aux_unknowns=(),
        elements=(
            IRElement("V2", "V", ("n1", "0"), (("voltage_v", 2.0),)),
            IRElement("L1", "L", ("n2", "0"), (("inductance_h", 3.0),)),
            IRElement("E2", "VCVS", ("n2", "0", "n1", "0"), (("gain_mu", 4.0),)),
            IRElement("R1", "R", ("n1", "0"), (("resistance_ohm", 50.0),)),
        ),
        ports=(),
        resolved_params=(),
    )

    normalized = normalize_canonical_ir_for_factory(ir)
    assert normalized.allocated_aux_ids == EXPECTED_AUX_IDS_MIXED
    assert ordered_aux_ids_from_models(normalized.models) == EXPECTED_AUX_IDS_MIXED


def test_unknown_kind_error_has_stable_code_message_and_witness() -> None:
    ir = CanonicalIR(
        nodes=(IRNode("0", is_reference=True), IRNode("n1"), IRNode("n2")),
        aux_unknowns=(),
        elements=(IRElement("X1", "mystery-device", ("n1", "n2"), (("value", 1.0),)),),
        ports=(),
        resolved_params=(),
    )

    with pytest.raises(FactoryModelNormalizationError) as exc_info:
        normalize_canonical_ir_for_factory(ir)
    detail = exc_info.value.detail

    assert detail.code == UNKNOWN_KIND_CODE
    assert detail.message == "unsupported element kind 'mystery-device' for element 'X1'"
    assert tuple(detail.witness) == EXPECTED_UNKNOWN_KIND_WITNESS_KEYS
    assert detail.witness == {
        "element_id": "X1",
        "normalized_candidate": "MYSTERY_DEVICE",
        "raw_kind": "mystery-device",
        "supported_kinds": ("R", "C", "G", "L", "I", "V", "VCCS", "VCVS"),
    }


def test_phase0_canonical_ir_json_and_hash_are_unchanged() -> None:
    ir = _phase0_ir_regression_fixture()
    assert canonical_ir_json(ir) == EXPECTED_PHASE0_JSON
    assert hash_canonical_ir(ir) == EXPECTED_PHASE0_HASH


def test_normalization_owns_aux_policy_not_ir_aux_declarations() -> None:
    base = _phase1_ir_variant_two()
    with_declared_aux = CanonicalIR(
        nodes=base.nodes,
        aux_unknowns=(
            IRAuxUnknown("z_aux"),
            IRAuxUnknown("a_aux"),
        ),
        elements=base.elements,
        ports=base.ports,
        resolved_params=base.resolved_params,
    )

    normalized_without = normalize_canonical_ir_for_factory(base)
    normalized_with = normalize_canonical_ir_for_factory(with_declared_aux)
    assert normalized_without == normalized_with


def test_normalized_model_output_is_immutable_tuple_order() -> None:
    normalized = normalize_canonical_ir_for_factory(_phase1_ir_variant_two())
    assert isinstance(normalized, NormalizedFactoryModels)
    assert isinstance(normalized.models, tuple)
    assert isinstance(normalized.allocated_aux_ids, tuple)

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.elements import ResistorStamp, StampContext
from rfmna.elements.controlled import VCVSStamp
from rfmna.elements.factory import (
    FACTORY_REGISTRY_BY_KIND,
    FactoryConstructionError,
    build_stamps_from_canonical_ir,
    build_stamps_from_normalized_models,
)
from rfmna.elements.factory_models import (
    InductorModel,
    NormalizedElementModel,
    ResistorModel,
    VCVSModel,
    VoltageSourceModel,
)
from rfmna.elements.inductor import InductorStamp
from rfmna.elements.sources import VoltageSourceStamp
from rfmna.ir import CanonicalIR, IRElement, IRNode

pytestmark = pytest.mark.unit


def _ctx() -> StampContext:
    return StampContext(omega_rad_s=1.0, resolved_params={})


def _canonical_ir_variant_one() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("n2"),
            IRNode("0", is_reference=True),
            IRNode("n1"),
            IRNode("nc"),
            IRNode("nd"),
        ),
        aux_unknowns=(),
        elements=(
            IRElement("V1", "V", ("n1", "0"), (("voltage_v", 1.0),)),
            IRElement("R1", "R", ("n1", "0"), (("resistance_ohm", 10.0),)),
            IRElement("GM1", "VCCS", ("n1", "0", "nc", "nd"), (("transconductance_s", 0.25),)),
            IRElement("E1", "VCVS", ("n2", "0", "n1", "0"), (("gain_mu", 2.0),)),
            IRElement("L1", "L", ("n1", "0"), (("inductance_h", 1e-9),)),
            IRElement("I1", "I", ("n2", "0"), (("current_a", 0.01),)),
            IRElement("C1", "C", ("n1", "n2"), (("capacitance_f", 1e-12),)),
            IRElement("G1", "G", ("n2", "0"), (("conductance_s", 0.02),)),
        ),
        ports=(),
        resolved_params=(),
    )


def _canonical_ir_variant_two() -> CanonicalIR:
    return CanonicalIR(
        nodes=(
            IRNode("nd"),
            IRNode("n1"),
            IRNode("nc"),
            IRNode("0", is_reference=True),
            IRNode("n2"),
        ),
        aux_unknowns=(),
        elements=(
            IRElement("C1", "C", ("n1", "n2"), (("capacitance_f", 1e-12),)),
            IRElement("E1", "VCVS", ("n2", "0", "n1", "0"), (("gain_mu", 2.0),)),
            IRElement("G1", "G", ("n2", "0"), (("conductance_s", 0.02),)),
            IRElement("GM1", "VCCS", ("n1", "0", "nc", "nd"), (("transconductance_s", 0.25),)),
            IRElement("I1", "I", ("n2", "0"), (("current_a", 0.01),)),
            IRElement("L1", "L", ("n1", "0"), (("inductance_h", 1e-9),)),
            IRElement("R1", "R", ("n1", "0"), (("resistance_ohm", 10.0),)),
            IRElement("V1", "V", ("n1", "0"), (("voltage_v", 1.0),)),
        ),
        ports=(),
        resolved_params=(),
    )


def test_registry_supports_all_phase1_kinds() -> None:
    assert tuple(FACTORY_REGISTRY_BY_KIND) == ("R", "C", "G", "L", "I", "V", "VCCS", "VCVS")


def test_stamp_list_order_is_stable_canonical_and_permutation_invariant() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i", "L1:i", "V1:i"))
    left = build_stamps_from_canonical_ir(_canonical_ir_variant_one(), indexing)
    right = build_stamps_from_canonical_ir(_canonical_ir_variant_two(), indexing)

    expected_ids = ("C1", "E1", "G1", "GM1", "I1", "L1", "R1", "V1")
    assert tuple(stamp.element_id for stamp in left) == expected_ids
    assert tuple(stamp.element_id for stamp in right) == expected_ids
    assert tuple(type(stamp).__name__ for stamp in left) == tuple(
        type(stamp).__name__ for stamp in right
    )


def test_unknown_kind_failure_has_deterministic_code_and_witness() -> None:
    @dataclass(frozen=True, slots=True)
    class UnsupportedModel:
        element_id: str
        kind: str

    indexing = build_unknown_indexing(("0", "n1"), "0")
    unknown = cast(NormalizedElementModel, UnsupportedModel(element_id="X1", kind="X_UNKNOWN"))

    with pytest.raises(FactoryConstructionError) as exc_info:
        build_stamps_from_normalized_models((unknown,), indexing=indexing)
    detail = exc_info.value.detail

    assert detail.code == "E_IR_KIND_UNKNOWN"
    assert detail.message == "unsupported normalized model kind 'X_UNKNOWN' for element 'X1'"
    assert detail.witness == {
        "element_id": "X1",
        "kind": "X_UNKNOWN",
        "supported_kinds": ("R", "C", "G", "L", "I", "V", "VCCS", "VCVS"),
    }


def test_constructor_failure_payload_is_machine_mappable() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    invalid_model = ResistorModel(element_id="", node_a="n1", node_b="0", resistance_ohm=10.0)

    with pytest.raises(FactoryConstructionError) as exc_info:
        build_stamps_from_normalized_models((invalid_model,), indexing=indexing)
    detail = exc_info.value.detail

    assert detail.code == "E_MODEL_STAMP_ELEMENT_INVALID"
    assert detail.element_id == "<missing-element-id>"
    assert detail.kind == "R"
    assert detail.witness == {
        "cause_type": "StampContractError",
        "element_id": "<missing-element-id>",
        "kind": "R",
    }


def test_validation_payload_propagates_without_string_only_conversion() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    model = ResistorModel(element_id="Rbad", node_a="n1", node_b="0", resistance_ohm=-1.0)
    stamps = build_stamps_from_normalized_models((model,), indexing=indexing)
    assert len(stamps) == 1
    assert isinstance(stamps[0], ResistorStamp)

    issues = stamps[0].validate(_ctx())
    assert len(issues) == 1
    assert issues[0].code == "E_MODEL_R_NONPOSITIVE"
    assert issues[0].message == "resistance_ohm must be > 0"
    assert issues[0].context == {"element_id": "Rbad", "resistance_ohm": -1.0}


def test_factory_consumes_aux_ids_from_models_without_allocation_or_reordering() -> None:
    indexing = build_unknown_indexing(
        ("0", "n1", "n2", "nc", "nd"),
        "0",
        ("custom_v_aux", "custom_l_aux", "custom_e_aux"),
    )
    models: tuple[NormalizedElementModel, ...] = (
        VoltageSourceModel("V1", "n1", "0", "custom_v_aux", 1.0),
        InductorModel("L1", "n2", "0", "custom_l_aux", 2.0),
        VCVSModel("E1", "n2", "0", "n1", "0", "custom_e_aux", 3.0),
    )

    stamps = build_stamps_from_normalized_models(models, indexing=indexing)
    assert isinstance(stamps[0], VoltageSourceStamp)
    assert isinstance(stamps[1], InductorStamp)
    assert isinstance(stamps[2], VCVSStamp)
    assert (stamps[0].aux_id, stamps[1].aux_id, stamps[2].aux_id) == (
        "custom_v_aux",
        "custom_l_aux",
        "custom_e_aux",
    )

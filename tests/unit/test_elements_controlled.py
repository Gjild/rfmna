from __future__ import annotations

from random import Random

import pytest

from rfmna.assembler import UnknownAuxIdError, build_unknown_indexing
from rfmna.elements import (
    MatrixCoord,
    MatrixEntry,
    StampContext,
    VCCSStamp,
    VCVSStamp,
)

pytestmark = pytest.mark.unit

_REPEATS = 20


def _ctx() -> StampContext:
    return StampContext(omega_rad_s=10.0, resolved_params={"alpha": 1.0})


def test_vccs_nominal_equations_signs_and_footprint() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0")
    stamp = VCCSStamp("G1", "n1", "n2", "nc", "nd", 2.5, indexing)

    assert stamp.touched_indices(_ctx()) == (0, 1, 2, 3)
    assert stamp.stamp_A(_ctx()) == (
        MatrixEntry(0, 2, 2.5 + 0.0j),
        MatrixEntry(0, 3, -2.5 + 0.0j),
        MatrixEntry(1, 2, -2.5 + 0.0j),
        MatrixEntry(1, 3, 2.5 + 0.0j),
    )
    assert stamp.footprint(_ctx()) == (
        MatrixCoord(0, 2),
        MatrixCoord(0, 3),
        MatrixCoord(1, 2),
        MatrixCoord(1, 3),
    )
    assert stamp.stamp_b(_ctx()) == ()


def test_vccs_reference_and_validation_behavior() -> None:
    indexing = build_unknown_indexing(("0", "n1", "nc", "nd"), "0")
    one_ref = VCCSStamp("G1", "n1", "0", "nc", "nd", 1.5, indexing)
    both_output_ref = VCCSStamp("G0", "0", "0", "nc", "nd", 1.5, indexing)

    assert one_ref.touched_indices(_ctx()) == (0, 1, 2)
    assert one_ref.stamp_A(_ctx()) == (
        MatrixEntry(0, 1, 1.5 + 0.0j),
        MatrixEntry(0, 2, -1.5 + 0.0j),
    )
    assert both_output_ref.touched_indices(_ctx()) == ()
    assert both_output_ref.stamp_A(_ctx()) == ()
    assert VCCSStamp("Gbad", "n1", "0", "nc", "nd", float("nan"), indexing).validate(_ctx())[
        0
    ].code == ("E_MODEL_VCCS_INVALID")


def test_vcvs_nominal_equations_signs_aux_and_footprint() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))
    stamp = VCVSStamp("E1", "n1", "n2", "nc", "nd", "E1:i", 4.0, indexing)

    assert stamp.touched_indices(_ctx()) == (0, 1, 2, 3, 4)
    assert stamp.stamp_A(_ctx()) == (
        MatrixEntry(0, 4, 1.0 + 0.0j),
        MatrixEntry(1, 4, -1.0 + 0.0j),
        MatrixEntry(4, 0, 1.0 + 0.0j),
        MatrixEntry(4, 1, -1.0 + 0.0j),
        MatrixEntry(4, 2, -4.0 + 0.0j),
        MatrixEntry(4, 3, 4.0 + 0.0j),
    )
    assert stamp.footprint(_ctx()) == (
        MatrixCoord(0, 4),
        MatrixCoord(1, 4),
        MatrixCoord(4, 0),
        MatrixCoord(4, 1),
        MatrixCoord(4, 2),
        MatrixCoord(4, 3),
    )
    assert stamp.stamp_b(_ctx()) == ()


def test_vcvs_reference_variants_and_validation_behavior() -> None:
    indexing = build_unknown_indexing(("0", "n1", "nc", "nd"), "0", ("E1:i", "E2:i", "E3:i"))

    one_ref_a = VCVSStamp("E1", "n1", "0", "nc", "nd", "E1:i", 2.0, indexing)
    assert one_ref_a.stamp_A(_ctx()) == (
        MatrixEntry(0, 3, 1.0 + 0.0j),
        MatrixEntry(3, 0, 1.0 + 0.0j),
        MatrixEntry(3, 1, -2.0 + 0.0j),
        MatrixEntry(3, 2, 2.0 + 0.0j),
    )

    one_ref_b = VCVSStamp("E2", "0", "n1", "nc", "nd", "E2:i", 2.0, indexing)
    assert one_ref_b.stamp_A(_ctx()) == (
        MatrixEntry(0, 4, -1.0 + 0.0j),
        MatrixEntry(4, 0, -1.0 + 0.0j),
        MatrixEntry(4, 1, -2.0 + 0.0j),
        MatrixEntry(4, 2, 2.0 + 0.0j),
    )

    both_ref = VCVSStamp("E3", "0", "0", "nc", "nd", "E3:i", 2.0, indexing)
    assert both_ref.touched_indices(_ctx()) == (1, 2, 5)
    assert both_ref.stamp_A(_ctx()) == (
        MatrixEntry(5, 1, -2.0 + 0.0j),
        MatrixEntry(5, 2, 2.0 + 0.0j),
    )

    assert VCVSStamp("Ebad", "n1", "0", "nc", "nd", "E1:i", float("inf"), indexing).validate(
        _ctx()
    )[0].code == ("E_MODEL_VCVS_INVALID")


def test_vcvs_unknown_aux_lookup_raises_typed_error() -> None:
    indexing = build_unknown_indexing(("0", "n1", "nc", "nd"), "0", ("E1:i",))
    stamp = VCVSStamp("E1", "n1", "0", "nc", "nd", "missing_aux", 1.0, indexing)
    with pytest.raises(UnknownAuxIdError) as exc_info:
        stamp.touched_indices(_ctx())
    assert exc_info.value.code == "E_INDEX_AUX_UNKNOWN"


def test_controlled_sources_are_deterministic_and_pure() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))
    params = {"b": 2.0, "a": 1.0}
    ctx = StampContext(omega_rad_s=5.0, resolved_params=params)
    vccs = VCCSStamp("G1", "n1", "n2", "nc", "nd", 1.5, indexing)
    vcvs = VCVSStamp("E1", "n1", "n2", "nc", "nd", "E1:i", 3.0, indexing)
    baseline = (
        vccs.touched_indices(ctx),
        vccs.footprint(ctx),
        vccs.stamp_A(ctx),
        vcvs.touched_indices(ctx),
        vcvs.footprint(ctx),
        vcvs.stamp_A(ctx),
    )

    for _ in range(_REPEATS):
        assert (
            vccs.touched_indices(ctx),
            vccs.footprint(ctx),
            vccs.stamp_A(ctx),
            vcvs.touched_indices(ctx),
            vcvs.footprint(ctx),
            vcvs.stamp_A(ctx),
        ) == baseline
    assert params == {"b": 2.0, "a": 1.0}


def test_orientation_permutations_are_stable() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))
    rng = Random(0)
    output_variants = [("n1", "n2"), ("n2", "n1")]
    control_variants = [("nc", "nd"), ("nd", "nc")]

    expected_vccs = {
        VCCSStamp("G1", a, b, c, d, 1.0, indexing).stamp_A(_ctx())
        for a, b in output_variants
        for c, d in control_variants
    }
    expected_vcvs = {
        VCVSStamp("E1", a, b, c, d, "E1:i", 1.0, indexing).stamp_A(_ctx())
        for a, b in output_variants
        for c, d in control_variants
    }

    for _ in range(_REPEATS):
        rng.shuffle(output_variants)
        rng.shuffle(control_variants)
        a, b = output_variants[0]
        c, d = control_variants[0]
        assert VCCSStamp("G1", a, b, c, d, 1.0, indexing).stamp_A(_ctx()) in expected_vccs
        assert VCVSStamp("E1", a, b, c, d, "E1:i", 1.0, indexing).stamp_A(_ctx()) in expected_vcvs

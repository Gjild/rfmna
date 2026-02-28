from __future__ import annotations

from random import Random

import pytest

from rfmna.assembler import UnknownAuxIdError, build_unknown_indexing
from rfmna.elements import (
    CurrentSourceStamp,
    MatrixCoord,
    MatrixEntry,
    RhsEntry,
    StampContext,
    VoltageSourceStamp,
)

pytestmark = pytest.mark.unit

_REPEATS = 20


def _ctx() -> StampContext:
    return StampContext(omega_rad_s=123.0, resolved_params={"alpha": 1.0})


def test_current_source_rhs_sign_conventions_match_canonical_equations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    src_ab = CurrentSourceStamp("Iab", "n1", "n2", 2.5, indexing)
    src_ba = CurrentSourceStamp("Iba", "n2", "n1", 2.5, indexing)

    assert src_ab.stamp_A(_ctx()) == ()
    assert src_ab.footprint(_ctx()) == ()
    assert src_ab.touched_indices(_ctx()) == (0, 1)
    assert src_ab.stamp_b(_ctx()) == (
        RhsEntry(0, -2.5 + 0.0j),
        RhsEntry(1, 2.5 + 0.0j),
    )

    assert src_ba.stamp_b(_ctx()) == (
        RhsEntry(0, 2.5 + 0.0j),
        RhsEntry(1, -2.5 + 0.0j),
    )


def test_current_source_reference_omission_and_validation() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    src_one_ref = CurrentSourceStamp("I1", "n1", "0", 1.25, indexing)
    src_both_ref = CurrentSourceStamp("I0", "0", "0", 1.25, indexing)

    assert src_one_ref.touched_indices(_ctx()) == (0,)
    assert src_one_ref.stamp_b(_ctx()) == (RhsEntry(0, -1.25 + 0.0j),)
    assert src_both_ref.touched_indices(_ctx()) == ()
    assert src_both_ref.stamp_b(_ctx()) == ()

    assert CurrentSourceStamp("Ibad", "n1", "0", float("nan"), indexing).validate(_ctx())[
        0
    ].code == ("E_MODEL_ISRC_INVALID")


def test_voltage_source_aux_formulation_and_orientation() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("V1:i", "V2:i"))
    src_ab = VoltageSourceStamp("V1", "n1", "n2", "V1:i", 4.0, indexing)
    src_ba = VoltageSourceStamp("V2", "n2", "n1", "V2:i", 4.0, indexing)

    assert src_ab.touched_indices(_ctx()) == (0, 1, 2)
    assert src_ab.stamp_A(_ctx()) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
    )
    assert src_ab.footprint(_ctx()) == (
        MatrixCoord(0, 2),
        MatrixCoord(1, 2),
        MatrixCoord(2, 0),
        MatrixCoord(2, 1),
    )
    assert src_ab.stamp_b(_ctx()) == (RhsEntry(2, 4.0 + 0.0j),)

    assert src_ba.stamp_A(_ctx()) == (
        MatrixEntry(0, 3, -1.0 + 0.0j),
        MatrixEntry(1, 3, 1.0 + 0.0j),
        MatrixEntry(3, 0, -1.0 + 0.0j),
        MatrixEntry(3, 1, 1.0 + 0.0j),
    )
    assert src_ba.stamp_b(_ctx()) == (RhsEntry(3, 4.0 + 0.0j),)


def test_voltage_source_reference_omission_and_validation() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("V1:i", "V2:i", "V3:i"))
    src_one_ref_a = VoltageSourceStamp("V1", "n1", "0", "V1:i", 2.0, indexing)
    src_one_ref_b = VoltageSourceStamp("V2", "0", "n1", "V2:i", 2.0, indexing)
    src_both_ref = VoltageSourceStamp("V3", "0", "0", "V3:i", 2.0, indexing)

    assert src_one_ref_a.stamp_A(_ctx()) == (
        MatrixEntry(0, 1, 1.0 + 0.0j),
        MatrixEntry(1, 0, 1.0 + 0.0j),
    )
    assert src_one_ref_a.stamp_b(_ctx()) == (RhsEntry(1, 2.0 + 0.0j),)

    assert src_one_ref_b.stamp_A(_ctx()) == (
        MatrixEntry(0, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, -1.0 + 0.0j),
    )
    assert src_one_ref_b.stamp_b(_ctx()) == (RhsEntry(2, 2.0 + 0.0j),)

    assert src_both_ref.touched_indices(_ctx()) == (3,)
    assert src_both_ref.stamp_A(_ctx()) == ()
    assert src_both_ref.footprint(_ctx()) == ()
    assert src_both_ref.stamp_b(_ctx()) == (RhsEntry(3, 2.0 + 0.0j),)

    assert VoltageSourceStamp("Vbad", "n1", "0", "V1:i", float("inf"), indexing).validate(_ctx())[
        0
    ].code == ("E_MODEL_VSRC_INVALID")


def test_voltage_source_unknown_aux_lookup_raises_typed_error() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("V1:i",))
    src = VoltageSourceStamp("Vx", "n1", "0", "missing_aux", 1.0, indexing)
    with pytest.raises(UnknownAuxIdError) as exc_info:
        src.touched_indices(_ctx())
    assert exc_info.value.code == "E_INDEX_AUX_UNKNOWN"


def test_sources_are_deterministic_and_pure() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("V1:i",))
    params = {"b": 2.0, "a": 1.0}
    ctx = StampContext(omega_rad_s=5.0, resolved_params=params)
    isrc = CurrentSourceStamp("I1", "n1", "n2", 1.0, indexing)
    vsrc = VoltageSourceStamp("V1", "n1", "n2", "V1:i", 3.0, indexing)
    baseline = (
        isrc.touched_indices(ctx),
        isrc.stamp_b(ctx),
        vsrc.touched_indices(ctx),
        vsrc.stamp_A(ctx),
        vsrc.stamp_b(ctx),
    )

    for _ in range(_REPEATS):
        assert (
            isrc.touched_indices(ctx),
            isrc.stamp_b(ctx),
            vsrc.touched_indices(ctx),
            vsrc.stamp_A(ctx),
            vsrc.stamp_b(ctx),
        ) == baseline
    assert params == {"b": 2.0, "a": 1.0}


def test_orientation_permutations_are_stable() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("V1:i", "V2:i"))
    rng = Random(0)
    variants = [("n1", "n2"), ("n2", "n1")]
    baseline_i = CurrentSourceStamp("I1", "n1", "n2", 1.0, indexing).stamp_b(_ctx())
    baseline_v = VoltageSourceStamp("V1", "n1", "n2", "V1:i", 1.0, indexing).stamp_A(_ctx())

    for _ in range(_REPEATS):
        rng.shuffle(variants)
        i_first = CurrentSourceStamp("I1", variants[0][0], variants[0][1], 1.0, indexing).stamp_b(
            _ctx()
        )
        i_second = CurrentSourceStamp("I1", variants[1][0], variants[1][1], 1.0, indexing).stamp_b(
            _ctx()
        )
        v_first = VoltageSourceStamp(
            "V1", variants[0][0], variants[0][1], "V1:i", 1.0, indexing
        ).stamp_A(_ctx())
        v_second = VoltageSourceStamp(
            "V1", variants[1][0], variants[1][1], "V1:i", 1.0, indexing
        ).stamp_A(_ctx())
        assert baseline_i in (i_first, i_second)
        assert baseline_v in (v_first, v_second)

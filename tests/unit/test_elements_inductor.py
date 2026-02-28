from __future__ import annotations

from random import Random

import pytest

from rfmna.assembler import UnknownAuxIdError, build_unknown_indexing
from rfmna.elements import InductorStamp, MatrixCoord, MatrixEntry, StampContext

pytestmark = pytest.mark.unit

_REPEATS = 20


def _ctx(omega: float = 10.0) -> StampContext:
    return StampContext(omega_rad_s=omega, resolved_params={"alpha": 1.0})


def test_nominal_non_reference_case_equations_and_aux_touched() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    stamp = InductorStamp("L1", "n1", "n2", "L1:i", 2.0, indexing)
    y_kk = -1j * (10.0 * 2.0)

    assert stamp.touched_indices(_ctx()) == (0, 1, 2)
    assert stamp.stamp_A(_ctx()) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
        MatrixEntry(2, 2, y_kk),
    )
    assert stamp.stamp_b(_ctx()) == ()


def test_reference_variants_and_both_reference_behavior() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("L1:i", "L2:i", "L3:i"))

    one_ref_a = InductorStamp("L1", "n1", "0", "L1:i", 1.0, indexing)
    assert one_ref_a.stamp_A(_ctx(omega=5.0)) == (
        MatrixEntry(0, 1, 1.0 + 0.0j),
        MatrixEntry(1, 0, 1.0 + 0.0j),
        MatrixEntry(1, 1, -5j),
    )

    one_ref_b = InductorStamp("L2", "0", "n1", "L2:i", 1.0, indexing)
    assert one_ref_b.stamp_A(_ctx(omega=5.0)) == (
        MatrixEntry(0, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, -1.0 + 0.0j),
        MatrixEntry(2, 2, -5j),
    )

    both_ref_nonzero = InductorStamp("L3", "0", "0", "L3:i", 2.0, indexing)
    assert both_ref_nonzero.touched_indices(_ctx(omega=7.0)) == (3,)
    assert both_ref_nonzero.stamp_A(_ctx(omega=7.0)) == (MatrixEntry(3, 3, -14j),)
    assert both_ref_nonzero.footprint(_ctx(omega=7.0)) == (MatrixCoord(3, 3),)

    assert both_ref_nonzero.stamp_A(_ctx(omega=0.0)) == ()
    assert both_ref_nonzero.footprint(_ctx(omega=0.0)) == ()


def test_orientation_sign_checks() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("Lab:i", "Lba:i"))
    lab = InductorStamp("Lab", "n1", "n2", "Lab:i", 1.5, indexing)
    lba = InductorStamp("Lba", "n2", "n1", "Lba:i", 1.5, indexing)

    assert lab.stamp_A(_ctx(omega=4.0)) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
        MatrixEntry(2, 2, -6j),
    )
    assert lba.stamp_A(_ctx(omega=4.0)) == (
        MatrixEntry(0, 3, -1.0 + 0.0j),
        MatrixEntry(1, 3, 1.0 + 0.0j),
        MatrixEntry(3, 0, -1.0 + 0.0j),
        MatrixEntry(3, 1, 1.0 + 0.0j),
        MatrixEntry(3, 3, -6j),
    )


def test_validation_for_nonpositive_and_nonfinite_L() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("L1:i",))
    assert InductorStamp("L0", "n1", "0", "L1:i", 0.0, indexing).validate(_ctx())[0].code == (
        "E_MODEL_L_NONPOSITIVE"
    )
    assert InductorStamp("LN", "n1", "0", "L1:i", -1.0, indexing).validate(_ctx())[0].code == (
        "E_MODEL_L_NONPOSITIVE"
    )
    assert InductorStamp("Linf", "n1", "0", "L1:i", float("inf"), indexing).validate(_ctx())[
        0
    ].code == ("E_MODEL_L_NONPOSITIVE")
    assert InductorStamp("Lok", "n1", "0", "L1:i", 1e-12, indexing).validate(_ctx()) == ()


def test_omega_zero_behavior_has_no_extra_terms() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    stamp = InductorStamp("L1", "n1", "n2", "L1:i", 3.0, indexing)
    assert stamp.stamp_A(_ctx(omega=0.0)) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
    )


def test_aux_lookup_failure_raises_typed_exception() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("L1:i",))
    stamp = InductorStamp("Lbad", "n1", "0", "missing_aux", 1.0, indexing)
    with pytest.raises(UnknownAuxIdError) as exc_info:
        stamp.touched_indices(_ctx())
    assert exc_info.value.code == "E_INDEX_AUX_UNKNOWN"
    assert str(exc_info.value) == "'unknown auxiliary id: missing_aux'"


def test_determinism_purity_and_no_context_mutation() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    params = {"b": 2.0, "a": 1.0}
    ctx = StampContext(omega_rad_s=8.0, resolved_params=params)
    stamp = InductorStamp("L1", "n1", "n2", "L1:i", 2.0, indexing)
    baseline = (
        stamp.touched_indices(ctx),
        stamp.footprint(ctx),
        stamp.stamp_A(ctx),
        stamp.stamp_b(ctx),
    )

    for _ in range(_REPEATS):
        assert (
            stamp.touched_indices(ctx),
            stamp.footprint(ctx),
            stamp.stamp_A(ctx),
            stamp.stamp_b(ctx),
        ) == (baseline)
    assert params == {"b": 2.0, "a": 1.0}


def test_stable_projection_across_equivalent_terminal_permutations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    rng = Random(0)
    baseline = InductorStamp("L1", "n1", "n2", "L1:i", 1.0, indexing).stamp_A(_ctx(omega=3.0))
    variants = [("n1", "n2"), ("n2", "n1")]

    for _ in range(_REPEATS):
        rng.shuffle(variants)
        first = InductorStamp("L1", variants[0][0], variants[0][1], "L1:i", 1.0, indexing).stamp_A(
            _ctx(omega=3.0)
        )
        second = InductorStamp("L1", variants[1][0], variants[1][1], "L1:i", 1.0, indexing).stamp_A(
            _ctx(omega=3.0)
        )
        assert baseline in (first, second)


def test_aux_always_in_touched_indices_even_both_reference() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("L1:i",))
    stamp = InductorStamp("L1", "0", "0", "L1:i", 1.0, indexing)
    assert stamp.touched_indices(_ctx()) == (1,)

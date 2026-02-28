from __future__ import annotations

from random import Random

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.elements import (
    CapacitorStamp,
    ConductanceStamp,
    MatrixCoord,
    MatrixEntry,
    ResistorStamp,
    StampContext,
)

pytestmark = pytest.mark.unit

_REPEATS = 20


def _ctx(omega: float = 1000.0) -> StampContext:
    return StampContext(omega_rad_s=omega, resolved_params={"alpha": 1.0})


def test_nominal_two_node_resistor_capacitor_conductance_stamps() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = _ctx(omega=1000.0)

    r = ResistorStamp("R1", "n1", "n2", 2.0, indexing)
    c = CapacitorStamp("C1", "n1", "n2", 1.0e-6, indexing)
    g = ConductanceStamp("G1", "n1", "n2", 0.25, indexing)

    y_r = 0.5 + 0j
    y_c = 1j * 0.001
    y_g = 0.25 + 0j

    assert r.stamp_A(ctx) == (
        MatrixEntry(0, 0, y_r),
        MatrixEntry(0, 1, -y_r),
        MatrixEntry(1, 0, -y_r),
        MatrixEntry(1, 1, y_r),
    )
    assert c.stamp_A(ctx) == (
        MatrixEntry(0, 0, y_c),
        MatrixEntry(0, 1, -y_c),
        MatrixEntry(1, 0, -y_c),
        MatrixEntry(1, 1, y_c),
    )
    assert g.stamp_A(ctx) == (
        MatrixEntry(0, 0, y_g),
        MatrixEntry(0, 1, -y_g),
        MatrixEntry(1, 0, -y_g),
        MatrixEntry(1, 1, y_g),
    )
    assert r.stamp_b(ctx) == ()
    assert c.stamp_b(ctx) == ()
    assert g.stamp_b(ctx) == ()


def test_reference_variants_one_reference_and_both_reference() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    ctx = _ctx()

    r_one = ResistorStamp("R1", "n1", "0", 5.0, indexing)
    assert r_one.stamp_A(ctx) == (MatrixEntry(0, 0, 0.2 + 0j),)
    assert r_one.footprint(ctx) == (MatrixCoord(0, 0),)

    c_one = CapacitorStamp("C1", "0", "n1", 2.0e-6, indexing)
    y_c = 1j * (ctx.omega_rad_s * 2.0e-6)
    assert c_one.stamp_A(ctx) == (MatrixEntry(0, 0, y_c),)

    g_both_ref = ConductanceStamp("G0", "0", "0", 1.0, indexing)
    assert g_both_ref.touched_indices(ctx) == ()
    assert g_both_ref.footprint(ctx) == ()
    assert g_both_ref.stamp_A(ctx) == ()


def test_validation_domains_and_boundaries() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = _ctx()

    assert (
        ResistorStamp("R0", "n1", "n2", 0.0, indexing).validate(ctx)[0].code
        == "E_MODEL_R_NONPOSITIVE"
    )
    assert (
        ResistorStamp("RN", "n1", "n2", -1.0, indexing).validate(ctx)[0].code
        == "E_MODEL_R_NONPOSITIVE"
    )
    assert (
        CapacitorStamp("CN", "n1", "n2", -1.0, indexing).validate(ctx)[0].code
        == "E_MODEL_C_NEGATIVE"
    )
    assert (
        ConductanceStamp("GN", "n1", "n2", -0.1, indexing).validate(ctx)[0].code
        == "E_MODEL_G_NEGATIVE"
    )

    c_zero = CapacitorStamp("C0", "n1", "n2", 0.0, indexing)
    g_zero = ConductanceStamp("G0", "n1", "n2", 0.0, indexing)
    r_small = ResistorStamp("RS", "n1", "n2", 1e-12, indexing)
    assert c_zero.validate(ctx) == ()
    assert g_zero.validate(ctx) == ()
    assert r_small.validate(ctx) == ()
    assert c_zero.stamp_A(ctx) == ()
    assert g_zero.stamp_A(ctx) == ()
    assert c_zero.footprint(ctx) == ()
    assert g_zero.footprint(ctx) == ()


def test_touched_indices_and_footprint_deterministic_order() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = _ctx()
    stamp = ResistorStamp("R1", "n2", "n1", 10.0, indexing)

    assert stamp.touched_indices(ctx) == (0, 1)
    assert stamp.footprint(ctx) == (
        MatrixCoord(0, 0),
        MatrixCoord(0, 1),
        MatrixCoord(1, 0),
        MatrixCoord(1, 1),
    )
    assert tuple(
        MatrixCoord(entry.row, entry.col) for entry in stamp.stamp_A(ctx)
    ) == stamp.footprint(ctx)


def test_repeatability_and_purity_no_context_mutation() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    params = {"beta": 2.0, "alpha": 1.0}
    ctx = StampContext(omega_rad_s=10.0, resolved_params=params)
    stamp = ConductanceStamp("G1", "n1", "n2", 0.5, indexing)

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
        ) == baseline
    assert params == {"beta": 2.0, "alpha": 1.0}


def test_capacitor_complex_numeric_path() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = _ctx(omega=123.0)
    stamp = CapacitorStamp("C1", "n1", "n2", 1e-3, indexing)
    entries = stamp.stamp_A(ctx)
    assert all(isinstance(entry.value, complex) for entry in entries)
    assert any(entry.value.imag != 0.0 for entry in entries)


def test_permuted_equivalent_inputs_stay_deterministic() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = _ctx(omega=5.0)
    baseline = ResistorStamp("R1", "n1", "n2", 3.0, indexing).stamp_A(ctx)
    rng = Random(0)
    variants = [("n1", "n2"), ("n2", "n1")]
    for _ in range(_REPEATS):
        rng.shuffle(variants)
        first = ResistorStamp("R1", variants[0][0], variants[0][1], 3.0, indexing).stamp_A(ctx)
        second = ResistorStamp("R1", variants[1][0], variants[1][1], 3.0, indexing).stamp_A(ctx)
        assert first == baseline
        assert second == baseline

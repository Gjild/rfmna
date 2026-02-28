from __future__ import annotations

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.elements import (
    CapacitorStamp,
    ConductanceStamp,
    MatrixEntry,
    ResistorStamp,
    StampContext,
)

pytestmark = pytest.mark.conformance


def test_two_node_sign_convention_matches_canonical_equations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = StampContext(omega_rad_s=200.0, resolved_params={})

    resistor = ResistorStamp("R1", "n1", "n2", 4.0, indexing)
    capacitor = CapacitorStamp("C1", "n1", "n2", 5e-6, indexing)
    conductance = ConductanceStamp("G1", "n1", "n2", 0.125, indexing)

    assert resistor.stamp_A(ctx) == (
        MatrixEntry(0, 0, 0.25 + 0j),
        MatrixEntry(0, 1, -0.25 + 0j),
        MatrixEntry(1, 0, -0.25 + 0j),
        MatrixEntry(1, 1, 0.25 + 0j),
    )
    y_c = 1j * (ctx.omega_rad_s * 5e-6)
    assert capacitor.stamp_A(ctx) == (
        MatrixEntry(0, 0, y_c),
        MatrixEntry(0, 1, -y_c),
        MatrixEntry(1, 0, -y_c),
        MatrixEntry(1, 1, y_c),
    )
    assert conductance.stamp_A(ctx) == (
        MatrixEntry(0, 0, 0.125 + 0j),
        MatrixEntry(0, 1, -0.125 + 0j),
        MatrixEntry(1, 0, -0.125 + 0j),
        MatrixEntry(1, 1, 0.125 + 0j),
    )


def test_reference_node_omission_and_no_aux_indices() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("aux_1", "aux_2"))
    ctx = StampContext(omega_rad_s=50.0, resolved_params={})
    resistor = ResistorStamp("R1", "n1", "0", 10.0, indexing)
    capacitor = CapacitorStamp("C1", "0", "0", 1e-6, indexing)
    conductance = ConductanceStamp("G1", "0", "n1", 0.5, indexing)

    assert resistor.stamp_A(ctx) == (MatrixEntry(0, 0, 0.1 + 0j),)
    assert capacitor.stamp_A(ctx) == ()
    assert conductance.stamp_A(ctx) == (MatrixEntry(0, 0, 0.5 + 0j),)

    assert all(index < indexing.n_nodes for index in resistor.touched_indices(ctx))
    assert all(index < indexing.n_nodes for index in capacitor.touched_indices(ctx))
    assert all(index < indexing.n_nodes for index in conductance.touched_indices(ctx))

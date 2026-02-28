from __future__ import annotations

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.elements import InductorStamp, MatrixEntry, StampContext

pytestmark = pytest.mark.conformance


def test_inductor_aux_current_formulation_matches_canonical_equations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    stamp = InductorStamp("L1", "n1", "n2", "L1:i", 3.0, indexing)
    ctx = StampContext(omega_rad_s=2.0, resolved_params={})

    assert stamp.touched_indices(ctx) == (0, 1, 2)
    assert stamp.stamp_A(ctx) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
        MatrixEntry(2, 2, -6j),
    )
    assert stamp.stamp_b(ctx) == ()


def test_inductor_reference_omission_is_deterministic_and_aux_is_always_touched() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("L1:i",))
    ctx = StampContext(omega_rad_s=5.0, resolved_params={})

    one_ref = InductorStamp("L1", "n1", "0", "L1:i", 1.0, indexing)
    assert one_ref.stamp_A(ctx) == (
        MatrixEntry(0, 1, 1.0 + 0.0j),
        MatrixEntry(1, 0, 1.0 + 0.0j),
        MatrixEntry(1, 1, -5j),
    )

    both_ref = InductorStamp("L2", "0", "0", "L1:i", 1.0, indexing)
    assert both_ref.touched_indices(ctx) == (1,)
    assert both_ref.stamp_A(ctx) == (MatrixEntry(1, 1, -5j),)

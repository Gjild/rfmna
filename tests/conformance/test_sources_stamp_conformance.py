from __future__ import annotations

import pytest

from rfmna.assembler import build_unknown_indexing
from rfmna.elements import (
    CurrentSourceStamp,
    MatrixEntry,
    RhsEntry,
    StampContext,
    VoltageSourceStamp,
)

pytestmark = pytest.mark.conformance


def test_independent_current_source_rhs_sign_conventions() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    ctx = StampContext(omega_rad_s=1.0, resolved_params={})
    src = CurrentSourceStamp("I1", "n1", "n2", 3.0, indexing)
    src_rev = CurrentSourceStamp("I2", "n2", "n1", 3.0, indexing)

    assert src.stamp_A(ctx) == ()
    assert src.stamp_b(ctx) == (
        RhsEntry(0, -3.0 + 0.0j),
        RhsEntry(1, 3.0 + 0.0j),
    )
    assert src_rev.stamp_b(ctx) == (
        RhsEntry(0, 3.0 + 0.0j),
        RhsEntry(1, -3.0 + 0.0j),
    )


def test_independent_voltage_source_aux_current_formulation_and_reference_omission() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("V1:i", "V2:i", "V3:i"))
    ctx = StampContext(omega_rad_s=1.0, resolved_params={})

    two_node = VoltageSourceStamp("V1", "n1", "n2", "V1:i", 5.0, indexing)
    assert two_node.stamp_A(ctx) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
    )
    assert two_node.stamp_b(ctx) == (RhsEntry(2, 5.0 + 0.0j),)

    one_ref = VoltageSourceStamp("V2", "n1", "0", "V2:i", 2.0, indexing)
    assert one_ref.stamp_A(ctx) == (
        MatrixEntry(0, 3, 1.0 + 0.0j),
        MatrixEntry(3, 0, 1.0 + 0.0j),
    )
    assert one_ref.stamp_b(ctx) == (RhsEntry(3, 2.0 + 0.0j),)

    both_ref = VoltageSourceStamp("V3", "0", "0", "V3:i", 7.0, indexing)
    assert both_ref.stamp_A(ctx) == ()
    assert both_ref.stamp_b(ctx) == (RhsEntry(4, 7.0 + 0.0j),)


def test_independent_voltage_source_orientation_sign_conventions() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("V1:i", "V2:i"))
    ctx = StampContext(omega_rad_s=1.0, resolved_params={})

    src_ab = VoltageSourceStamp("V1", "n1", "n2", "V1:i", 4.0, indexing)
    src_ba = VoltageSourceStamp("V2", "n2", "n1", "V2:i", 4.0, indexing)

    assert src_ab.stamp_A(ctx) == (
        MatrixEntry(0, 2, 1.0 + 0.0j),
        MatrixEntry(1, 2, -1.0 + 0.0j),
        MatrixEntry(2, 0, 1.0 + 0.0j),
        MatrixEntry(2, 1, -1.0 + 0.0j),
    )
    assert src_ab.stamp_b(ctx) == (RhsEntry(2, 4.0 + 0.0j),)

    assert src_ba.stamp_A(ctx) == (
        MatrixEntry(0, 3, -1.0 + 0.0j),
        MatrixEntry(1, 3, 1.0 + 0.0j),
        MatrixEntry(3, 0, -1.0 + 0.0j),
        MatrixEntry(3, 1, 1.0 + 0.0j),
    )
    assert src_ba.stamp_b(ctx) == (RhsEntry(3, 4.0 + 0.0j),)

from __future__ import annotations

import pytest

from rfmna.assembler import (
    build_unknown_indexing,
    compile_pattern,
    fill_numeric,
    pattern_projection,
)
from rfmna.elements import InductorStamp, ResistorStamp, StampContext

pytestmark = pytest.mark.conformance


def _ctx(omega: float) -> StampContext:
    return StampContext(omega_rad_s=omega, resolved_params={})


def test_two_stage_separation_and_deterministic_slot_mapping() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    elements = (
        ResistorStamp("R1", "n1", "n2", 2.0, indexing),
        InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing),
    )
    pattern_low = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=0.0))
    pattern_high = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=1.0e6))
    assert pattern_projection(pattern_low) == pattern_projection(pattern_high)
    assert tuple(sorted(pattern_low.coord_to_slot.items())) == tuple(
        sorted(pattern_high.coord_to_slot.items())
    )


def test_compiled_pattern_reuse_has_no_structural_drift_and_full_numeric_payload() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    elements = (
        ResistorStamp("R1", "n1", "n2", 2.0, indexing),
        InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing),
    )
    pattern = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=0.0))
    first = fill_numeric(pattern, elements, _ctx(omega=1.0))
    second = fill_numeric(pattern, elements, _ctx(omega=10.0))

    assert pattern_projection(first.pattern) == pattern_projection(second.pattern)
    assert len(first.slot_values) == len(pattern.coords)
    assert len(second.slot_values) == len(pattern.coords)
    assert first.b.shape == (indexing.total_unknowns,)
    assert second.b.shape == (indexing.total_unknowns,)

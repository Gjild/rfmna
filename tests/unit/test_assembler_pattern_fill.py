from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random

import pytest
from scipy.sparse import issparse

from rfmna.assembler import (
    AssemblerError,
    build_unknown_indexing,
    compile_pattern,
    fill_numeric,
    filled_projection,
    pattern_projection,
)
from rfmna.elements import (
    ElementStamp,
    InductorStamp,
    MatrixCoord,
    MatrixEntry,
    ResistorStamp,
    RhsEntry,
    StampContext,
    ValidationIssue,
)

pytestmark = pytest.mark.unit

_REPEATS = 20


@dataclass(frozen=True, slots=True)
class MockElement(ElementStamp):
    element_id: str
    coords: tuple[MatrixCoord, ...]
    matrix_entries: tuple[MatrixEntry, ...]
    rhs_entries: tuple[RhsEntry, ...]

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        touched = {entry.row for entry in self.matrix_entries}
        touched.update(entry.col for entry in self.matrix_entries)
        touched.update(entry.row for entry in self.rhs_entries)
        return tuple(sorted(touched))

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
        del ctx
        return self.coords

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        del ctx
        return self.matrix_entries

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        del ctx
        return self.rhs_entries

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        del ctx
        return ()


def _ctx(omega: float) -> StampContext:
    return StampContext(omega_rad_s=omega, resolved_params={"alpha": 1.0})


def _pattern_bytes() -> bytes:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    elements: tuple[ElementStamp, ...] = (
        ResistorStamp("R1", "n1", "n2", 2.0, indexing),
        InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing),
    )
    pattern = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=0.0))
    payload = {
        "coords": pattern_projection(pattern)[0],
        "rhs_rows": pattern_projection(pattern)[1],
        "coord_to_slot": tuple(sorted(pattern.coord_to_slot.items())),
        "rhs_row_to_slot": tuple(sorted(pattern.rhs_row_to_slot.items())),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def test_pattern_repeatability_is_byte_stable() -> None:
    baseline = _pattern_bytes()
    for _ in range(_REPEATS):
        assert _pattern_bytes() == baseline


def test_pattern_invariance_under_equivalent_canonicalized_input_permutations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    e1: ElementStamp = ResistorStamp("R1", "n1", "n2", 2.0, indexing)
    e2: ElementStamp = InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing)
    left = compile_pattern(indexing.total_unknowns, (e1, e2), _ctx(omega=11.0))
    right = compile_pattern(indexing.total_unknowns, (e2, e1), _ctx(omega=3.0))
    assert pattern_projection(left) == pattern_projection(right)


def test_pattern_independence_from_omega() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    elements: tuple[ElementStamp, ...] = (InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing),)
    low = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=0.0))
    high = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=1.0e9))
    assert pattern_projection(low) == pattern_projection(high)


def test_numeric_correctness_fixture() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    elements: tuple[ElementStamp, ...] = (
        ResistorStamp("R1", "n1", "n2", 2.0, indexing),
        InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing),
        MockElement("Iinj", (), (), (RhsEntry(0, 1.0 + 0.0j),)),
    )
    pattern = compile_pattern(indexing.total_unknowns, elements, _ctx(omega=3.0))
    filled = fill_numeric(pattern, elements, _ctx(omega=4.0))
    coords_values, rhs = filled_projection(filled)

    assert coords_values == (
        (0, 0, 0.5 + 0.0j),
        (0, 1, -0.5 + 0.0j),
        (1, 0, -0.5 + 0.0j),
        (1, 1, 0.5 + 0.0j),
        (1, 2, 1.0 + 0.0j),
        (2, 1, 1.0 + 0.0j),
        (2, 2, -4j),
    )
    assert rhs == (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)


def test_pattern_reuse_across_points_preserves_structure() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0")
    capacitor = MockElement(
        "C1",
        (
            MatrixCoord(0, 0),
            MatrixCoord(0, 1),
            MatrixCoord(1, 0),
            MatrixCoord(1, 1),
        ),
        (),
        (),
    )
    pattern = compile_pattern(indexing.total_unknowns, (capacitor,), _ctx(omega=0.0))

    @dataclass(frozen=True, slots=True)
    class OmegaCap(ElementStamp):
        element_id: str

        def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
            del ctx
            return (0, 1)

        def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
            del ctx
            return pattern.coords

        def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
            y = 1j * ctx.omega_rad_s * 1.0e-6
            return (
                MatrixEntry(0, 0, y),
                MatrixEntry(0, 1, -y),
                MatrixEntry(1, 0, -y),
                MatrixEntry(1, 1, y),
            )

        def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
            del ctx
            return ()

        def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
            del ctx
            return ()

    element = OmegaCap("Cdyn")
    first = fill_numeric(pattern, (element,), _ctx(omega=1.0))
    second = fill_numeric(pattern, (element,), _ctx(omega=10.0))
    assert pattern_projection(first.pattern) == pattern_projection(second.pattern)
    assert first.slot_values != second.slot_values


def test_deterministic_accumulation_duplicate_contributions() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    element = MockElement(
        "DUP",
        (MatrixCoord(0, 0),),
        (MatrixEntry(0, 0, 1.0 + 0j), MatrixEntry(0, 0, 2.0 + 0j)),
        (RhsEntry(0, 1.0 + 0j), RhsEntry(0, 2.0 + 0j)),
    )
    pattern = compile_pattern(indexing.total_unknowns, (element,), _ctx(omega=0.0))
    filled = fill_numeric(pattern, (element,), _ctx(omega=0.0))
    coords_values, rhs = filled_projection(filled)
    assert coords_values == ((0, 0, 3.0 + 0.0j),)
    assert rhs == (3.0 + 0.0j,)


def test_reference_omission_consistency() -> None:
    indexing = build_unknown_indexing(("0",), "0")
    resistor = ResistorStamp("R0", "0", "0", 10.0, indexing)
    pattern = compile_pattern(indexing.total_unknowns, (resistor,), _ctx(omega=1.0))
    assert pattern.coords == ()
    filled = fill_numeric(pattern, (resistor,), _ctx(omega=1.0))
    assert filled.slot_values == ()
    assert tuple(complex(value) for value in filled.b) == ()


def test_sparse_only_and_no_dense_shortcut() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    resistor = ResistorStamp("R1", "n1", "0", 2.0, indexing)
    pattern = compile_pattern(indexing.total_unknowns, (resistor,), _ctx(omega=1.0))
    filled = fill_numeric(pattern, (resistor,), _ctx(omega=1.0))
    assert issparse(filled.A)


def test_compile_and_fill_do_not_mutate_inputs() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    resistor = ResistorStamp("R1", "n1", "0", 2.0, indexing)
    elements: tuple[ElementStamp, ...] = (resistor,)
    params = {"b": 2.0, "a": 1.0}
    ctx = StampContext(omega_rad_s=3.0, resolved_params=params)
    _ = compile_pattern(indexing.total_unknowns, elements, ctx)
    _ = fill_numeric(compile_pattern(indexing.total_unknowns, elements, ctx), elements, ctx)
    assert params == {"b": 2.0, "a": 1.0}


def test_structure_violation_paths_raise_deterministic_exceptions() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0")
    base = MockElement("BASE", (MatrixCoord(0, 0),), (), (RhsEntry(0, 0.0 + 0.0j),))
    pattern = compile_pattern(indexing.total_unknowns, (base,), _ctx(omega=1.0))

    bad_coord = MockElement("BADC", (MatrixCoord(0, 0),), (MatrixEntry(0, 1, 1.0 + 0j),), ())
    with pytest.raises(AssemblerError) as coord_error:
        fill_numeric(pattern, (bad_coord,), _ctx(omega=1.0))
    assert coord_error.value.code == "E_ASSEMBLER_FILL_COORD_NOT_COMPILED"

    bad_rhs = MockElement("BADR", (MatrixCoord(0, 0),), (), (RhsEntry(1, 1.0 + 0j),))
    with pytest.raises(AssemblerError) as rhs_error:
        fill_numeric(pattern, (bad_rhs,), _ctx(omega=1.0))
    assert rhs_error.value.code == "E_ASSEMBLER_FILL_RHS_NOT_COMPILED"


def test_pattern_and_fill_are_deterministic_across_element_order_permutations() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2"), "0", ("L1:i",))
    e1: ElementStamp = ResistorStamp("R1", "n1", "n2", 2.0, indexing)
    e2: ElementStamp = InductorStamp("L1", "n2", "0", "L1:i", 1.0, indexing)
    elements = [e1, e2]
    baseline_pattern = compile_pattern(indexing.total_unknowns, tuple(elements), _ctx(omega=8.0))
    baseline_fill = filled_projection(
        fill_numeric(baseline_pattern, tuple(elements), _ctx(omega=8.0))
    )

    rng = Random(0)
    for _ in range(_REPEATS):
        rng.shuffle(elements)
        current_pattern = compile_pattern(indexing.total_unknowns, tuple(elements), _ctx(omega=2.0))
        current_fill = filled_projection(
            fill_numeric(current_pattern, tuple(elements), _ctx(omega=8.0))
        )
        assert pattern_projection(current_pattern) == pattern_projection(baseline_pattern)
        assert current_fill == baseline_fill

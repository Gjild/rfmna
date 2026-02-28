from __future__ import annotations

from dataclasses import FrozenInstanceError
from random import Random

import pytest

from rfmna.elements import (
    ElementStamp,
    MatrixCoord,
    MatrixEntry,
    RhsEntry,
    StampContext,
    StampContractError,
    ValidationIssue,
    canonicalize_coords,
    canonicalize_indices,
    canonicalize_matrix_entries,
    canonicalize_rhs_entries,
)

pytestmark = pytest.mark.unit

_REPEATS = 25


class MockElement:
    element_id = "X1"

    def touched_indices(self, ctx: StampContext) -> tuple[int, ...]:
        del ctx
        return canonicalize_indices((3, 1, 3, 0))

    def footprint(self, ctx: StampContext) -> tuple[MatrixCoord, ...]:
        del ctx
        return canonicalize_coords(
            (
                MatrixCoord(1, 1),
                MatrixCoord(0, 3),
                MatrixCoord(1, 1),
                MatrixCoord(0, 0),
            )
        )

    def stamp_A(self, ctx: StampContext) -> tuple[MatrixEntry, ...]:
        gain = complex(ctx.resolved_params["gain"])
        return canonicalize_matrix_entries(
            (
                MatrixEntry(1, 1, gain),
                MatrixEntry(0, 0, 1.0 + 0j),
                MatrixEntry(1, 1, 2.0 + 0j),
            )
        )

    def stamp_b(self, ctx: StampContext) -> tuple[RhsEntry, ...]:
        return canonicalize_rhs_entries(
            (
                RhsEntry(1, complex(ctx.omega_rad_s, 0.0)),
                RhsEntry(1, 1.0 + 0j),
                RhsEntry(0, -1.0 + 0j),
            )
        )

    def validate(self, ctx: StampContext) -> tuple[ValidationIssue, ...]:
        if ctx.omega_rad_s < 0.0:
            return (
                ValidationIssue(
                    code="E_MODEL_PARAM_DOMAIN",
                    message="omega_rad_s must be non-negative for this mock element",
                    context={"element_id": self.element_id},
                ),
            )
        return ()


def test_interface_conformance_with_mock_element() -> None:
    element = MockElement()
    assert isinstance(element, ElementStamp)

    ctx = StampContext(omega_rad_s=1.0, resolved_params={"gain": 3.0})
    assert element.touched_indices(ctx) == (0, 1, 3)
    assert element.footprint(ctx) == (MatrixCoord(0, 0), MatrixCoord(0, 3), MatrixCoord(1, 1))
    assert element.stamp_A(ctx) == (
        MatrixEntry(0, 0, 1.0 + 0j),
        MatrixEntry(1, 1, 5.0 + 0j),
    )
    assert element.stamp_b(ctx) == (
        RhsEntry(0, -1.0 + 0j),
        RhsEntry(1, 2.0 + 0j),
    )
    assert element.validate(ctx) == ()


def test_deterministic_ordering_helpers_from_shuffled_inputs() -> None:
    rng = Random(0)
    indices = [5, 1, 3, 1, 5, 2]
    coords = [MatrixCoord(2, 1), MatrixCoord(0, 2), MatrixCoord(2, 1), MatrixCoord(0, 1)]
    matrix_entries = [
        MatrixEntry(2, 0, 1.0 + 0j),
        MatrixEntry(1, 1, 1.0 + 0j),
        MatrixEntry(2, 0, 3.0 + 0j),
    ]
    rhs_entries = [RhsEntry(1, 2.0 + 0j), RhsEntry(0, -1.0 + 0j), RhsEntry(1, 1.0 + 0j)]

    baseline = (
        canonicalize_indices(indices),
        canonicalize_coords(coords),
        canonicalize_matrix_entries(matrix_entries),
        canonicalize_rhs_entries(rhs_entries),
    )
    for _ in range(_REPEATS):
        rng.shuffle(indices)
        rng.shuffle(coords)
        rng.shuffle(matrix_entries)
        rng.shuffle(rhs_entries)
        current = (
            canonicalize_indices(indices),
            canonicalize_coords(coords),
            canonicalize_matrix_entries(matrix_entries),
            canonicalize_rhs_entries(rhs_entries),
        )
        assert current == baseline


def test_purity_and_no_mutation_of_caller_mappings() -> None:
    params = {"z": 2.0, "a": 1.0, "gain": 3.0}
    context = StampContext(omega_rad_s=2.5, resolved_params=params)
    element = MockElement()

    baseline_a = element.stamp_A(context)
    baseline_b = element.stamp_b(context)
    for _ in range(_REPEATS):
        assert element.stamp_A(context) == baseline_a
        assert element.stamp_b(context) == baseline_b

    assert params == {"z": 2.0, "a": 1.0, "gain": 3.0}
    assert tuple(context.resolved_params.keys()) == ("a", "gain", "z")


def test_validation_issue_structure_is_deterministic() -> None:
    issue = ValidationIssue(
        code="E_MODEL_PARAM_DOMAIN",
        message="bad domain",
        context={"b": 2, "a": 1},
    )
    assert issue.code == "E_MODEL_PARAM_DOMAIN"
    assert issue.message == "bad domain"
    assert issue.context is not None
    assert tuple(issue.context.items()) == (("a", 1), ("b", 2))

    element = MockElement()
    issues = element.validate(StampContext(omega_rad_s=-1.0, resolved_params={"gain": 1.0}))
    assert issues == (
        ValidationIssue(
            code="E_MODEL_PARAM_DOMAIN",
            message="omega_rad_s must be non-negative for this mock element",
            context={"element_id": "X1"},
        ),
    )


def test_contract_violations_raise_typed_exceptions() -> None:
    with pytest.raises(StampContractError) as neg_index:
        canonicalize_indices((0, -1, 2))
    assert neg_index.value.code == "E_MODEL_STAMP_INDEX_INVALID"
    assert neg_index.value.message == "index must be a non-negative integer"

    with pytest.raises(StampContractError) as invalid_context:
        StampContext(omega_rad_s=float("nan"), resolved_params={})
    assert invalid_context.value.code == "E_MODEL_STAMP_CONTEXT_INVALID"

    with pytest.raises(StampContractError) as invalid_value:
        MatrixEntry(0, 0, complex(float("inf"), 0.0))
    assert invalid_value.value.code == "E_MODEL_STAMP_VALUE_INVALID"


def test_immutable_records() -> None:
    coord = MatrixCoord(0, 1)
    entry = MatrixEntry(0, 1, 1.0 + 0j)
    rhs = RhsEntry(0, 1.0 + 0j)
    issue = ValidationIssue("E_MODEL_PARAM_DOMAIN", "bad")
    context = StampContext(omega_rad_s=1.0, resolved_params={"a": 1.0})

    with pytest.raises(FrozenInstanceError):
        coord.row = 1
    with pytest.raises(FrozenInstanceError):
        entry.value = 0.0 + 0j
    with pytest.raises(FrozenInstanceError):
        rhs.row = 2
    with pytest.raises(FrozenInstanceError):
        issue.code = "X"
    with pytest.raises(FrozenInstanceError):
        context.omega_rad_s = 3.0

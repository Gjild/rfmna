from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from random import Random

import pytest

from rfmna.assembler import UnknownAuxIdError, build_unknown_indexing

pytestmark = pytest.mark.unit

_REPEAT_COUNT = 30


def _canonical_projection_bytes() -> bytes:
    indexing = build_unknown_indexing(
        node_ids=("0", "n1", "n2"),
        reference_node="0",
        aux_ids=("E1:Iaux", "E2:Iaux"),
    )
    payload = {
        "ordered_node_ids": indexing.ordered_node_ids,
        "ordered_aux_ids": indexing.ordered_aux_ids,
        "n_nodes": indexing.n_nodes,
        "n_aux": indexing.n_aux,
        "total_unknowns": indexing.total_unknowns,
        "projection": indexing.projection(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def test_basic_indexing_and_reference_behavior() -> None:
    node_ids = ("0", "n1", "n2")
    aux_ids = ("E1:Iaux", "E2:Iaux")
    indexing = build_unknown_indexing(
        node_ids=node_ids,
        reference_node="0",
        aux_ids=aux_ids,
    )
    expected_n_nodes = len(node_ids) - 1
    expected_n_aux = len(aux_ids)
    expected_total = expected_n_nodes + expected_n_aux

    assert indexing.n_nodes == expected_n_nodes
    assert indexing.n_aux == expected_n_aux
    assert indexing.total_unknowns == expected_total
    assert indexing.ordered_node_ids == ("n1", "n2")
    assert indexing.ordered_aux_ids == aux_ids
    assert indexing.node_index("0") is None
    assert indexing.node_index("n1") == 0
    assert indexing.node_index("n2") == 1
    assert indexing.aux_index("E1:Iaux") == expected_n_nodes
    assert indexing.aux_index("E2:Iaux") == expected_n_nodes + 1
    assert indexing.node_indices(("n2", "0", "n1")) == (1, None, 0)
    assert indexing.aux_indices(("E2:Iaux", "E1:Iaux")) == (expected_n_nodes + 1, expected_n_nodes)
    assert set(range(indexing.total_unknowns)) == set(range(expected_total))


def test_deterministic_aux_ordering_across_runs() -> None:
    baseline = build_unknown_indexing(
        node_ids=("0", "n1", "n2"),
        reference_node="0",
        aux_ids=("A:ix", "B:ix", "C:ix"),
    )
    for _ in range(_REPEAT_COUNT):
        current = build_unknown_indexing(
            node_ids=("0", "n1", "n2"),
            reference_node="0",
            aux_ids=("A:ix", "B:ix", "C:ix"),
        )
        assert current.projection() == baseline.projection()


def test_permutation_stability_with_equivalent_canonicalized_inputs() -> None:
    raw_nodes_one = ("n2", "0", "n1")
    raw_nodes_two = ("n1", "n2", "0")
    raw_aux_one = (("E2", "E2:Iaux"), ("E1", "E1:Iaux"))
    raw_aux_two = (("E1", "E1:Iaux"), ("E2", "E2:Iaux"))

    canonical_nodes_one = tuple(sorted(raw_nodes_one, key=lambda node: (node != "0", node)))
    canonical_nodes_two = tuple(sorted(raw_nodes_two, key=lambda node: (node != "0", node)))
    canonical_aux_one = tuple(aux_id for _, aux_id in sorted(raw_aux_one, key=lambda item: item[0]))
    canonical_aux_two = tuple(aux_id for _, aux_id in sorted(raw_aux_two, key=lambda item: item[0]))

    left = build_unknown_indexing(canonical_nodes_one, "0", canonical_aux_one)
    right = build_unknown_indexing(canonical_nodes_two, "0", canonical_aux_two)
    assert left.projection() == right.projection()


def test_slice_and_total_consistency() -> None:
    indexing = build_unknown_indexing(("ref", "n1", "n2", "n3"), "ref", ("a1",))
    assert indexing.node_voltage_slice == slice(0, 3)
    assert indexing.aux_slice == slice(3, 4)
    assert indexing.node_voltage_slice.stop == indexing.n_nodes
    assert indexing.aux_slice.start == indexing.n_nodes
    assert indexing.aux_slice.stop == indexing.total_unknowns


def test_repeated_run_serialized_projection_is_identical() -> None:
    baseline = _canonical_projection_bytes()
    for _ in range(_REPEAT_COUNT):
        assert _canonical_projection_bytes() == baseline


def test_edge_cases_no_aux_and_reference_placement() -> None:
    no_aux = build_unknown_indexing(("0", "n1"), "0", ())
    assert no_aux.n_aux == 0
    assert no_aux.aux_slice == slice(1, 1)
    assert no_aux.total_unknowns == 1

    shifted_reference = build_unknown_indexing(("n2", "n1", "ref"), "ref", ())
    assert shifted_reference.ordered_node_ids == ("n2", "n1")
    assert shifted_reference.node_index("ref") is None
    assert shifted_reference.node_index("n2") == 0


def test_invalid_aux_lookup_raises_typed_deterministic_exception() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("E1:Iaux",))
    with pytest.raises(UnknownAuxIdError) as exc_info:
        indexing.aux_index("missing")
    assert exc_info.value.code == "E_INDEX_AUX_UNKNOWN"
    assert str(exc_info.value) == "'unknown auxiliary id: missing'"


def test_no_mutation_of_indexing_container() -> None:
    indexing = build_unknown_indexing(("0", "n1"), "0", ("E1:Iaux",))
    with pytest.raises(FrozenInstanceError):
        indexing.n_nodes = 3


def test_randomized_input_permutations_do_not_change_projection_after_canonicalization() -> None:
    rng = Random(0)
    baseline = build_unknown_indexing(("0", "n1", "n2"), "0", ("E1:Iaux", "E2:Iaux")).projection()
    nodes = ["0", "n1", "n2"]
    aux_pairs = [("E1", "E1:Iaux"), ("E2", "E2:Iaux")]

    for _ in range(_REPEAT_COUNT):
        rng.shuffle(nodes)
        rng.shuffle(aux_pairs)
        canonical_nodes = tuple(sorted(nodes, key=lambda node: (node != "0", node)))
        canonical_aux = tuple(aux_id for _, aux_id in sorted(aux_pairs, key=lambda item: item[0]))
        current = build_unknown_indexing(canonical_nodes, "0", canonical_aux).projection()
        assert current == baseline

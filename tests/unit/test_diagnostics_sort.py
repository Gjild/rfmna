from __future__ import annotations

from random import Random

import pytest

from rfmna.diagnostics.models import DiagnosticEvent, Severity, SolverStage
from rfmna.diagnostics.sort import canonical_witness_json, sort_diagnostics

pytestmark = pytest.mark.unit


def _event(**overrides: object) -> DiagnosticEvent:
    payload: dict[str, object] = {
        "code": "E_NUM_SOLVE_FAILED",
        "severity": Severity.ERROR,
        "message": "m",
        "suggested_action": "a",
        "solver_stage": SolverStage.SOLVE,
        "element_id": "E1",
        "frequency_index": 2,
        "witness": {"k": 1},
    }
    payload.update(overrides)
    return DiagnosticEvent(**payload)


def test_canonical_witness_json_is_stable_and_compact() -> None:
    assert canonical_witness_json({"b": 1, "a": {"d": 4, "c": 3}}) == '{"a":{"c":3,"d":4},"b":1}'
    assert canonical_witness_json(None) == ""


def test_sort_diagnostics_follows_canonical_policy() -> None:
    events = [
        _event(message="d", severity=Severity.WARNING),
        _event(message="c", solver_stage=SolverStage.POSTPROCESS),
        _event(message="b", code="E_MODEL_C_NEGATIVE"),
        _event(message="a", element_id=None, node_context={"node_id": "n1"}),
        _event(message="e", frequency_index=None, sweep_index=None),
        _event(message="f", witness={"k": 2}),
    ]

    ordered = sort_diagnostics(events)

    assert [event.message for event in ordered] == ["b", "f", "e", "a", "c", "d"]


def test_sort_diagnostics_point_indices_are_explicitly_lexicographic() -> None:
    events = [
        _event(message="only-sweep", frequency_index=None, sweep_index=0),
        _event(message="both-set", frequency_index=1, sweep_index=5),
        _event(message="only-freq", frequency_index=1, sweep_index=None),
        _event(message="both-none", frequency_index=None, sweep_index=None),
    ]

    ordered = sort_diagnostics(events)

    assert [event.message for event in ordered] == [
        "both-set",
        "only-freq",
        "only-sweep",
        "both-none",
    ]


def test_sort_diagnostics_is_deterministic_across_permutations() -> None:
    seed_events = [
        _event(message="m1", code="E_MODEL_C_NEGATIVE", solver_stage=SolverStage.PARSE),
        _event(message="m2", code="E_MODEL_C_NEGATIVE", solver_stage=SolverStage.ASSEMBLE),
        _event(message="m3", code="E_MODEL_C_NEGATIVE", solver_stage=SolverStage.SOLVE),
        _event(message="m4", code="E_NUM_SOLVE_FAILED", solver_stage=SolverStage.SOLVE),
        _event(message="m5", code="W_NUM_ILL_CONDITIONED", severity=Severity.WARNING),
        _event(message="m6", frequency_index=1, sweep_index=4),
        _event(message="m7", frequency_index=None, sweep_index=4),
    ]
    baseline = sort_diagnostics(seed_events)
    rng = Random(0)

    for _ in range(200):
        permuted = list(seed_events)
        rng.shuffle(permuted)
        assert sort_diagnostics(permuted) == baseline

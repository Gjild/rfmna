from __future__ import annotations

import pytest
from pydantic import ValidationError

from rfmna.diagnostics.models import (
    DiagnosticEvent,
    NodeContext,
    PortContext,
    Severity,
    SolverStage,
)

pytestmark = pytest.mark.unit


def _base_event(**overrides: object) -> DiagnosticEvent:
    payload: dict[str, object] = {
        "code": "E_NUM_SOLVE_FAILED",
        "severity": Severity.ERROR,
        "message": "matrix solve failed",
        "suggested_action": "check topology and parameters",
        "solver_stage": SolverStage.SOLVE,
        "element_id": "R1",
    }
    payload.update(overrides)
    return DiagnosticEvent(**payload)


def test_context_presence_accepts_element_or_node_or_port() -> None:
    by_element = _base_event()
    by_node = _base_event(element_id=None, node_context=NodeContext(node_id="n1"))
    by_port = _base_event(element_id=None, port_context=PortContext(port_id="p1"))

    assert by_element.element_id == "R1"
    assert by_node.node_context == NodeContext(node_id="n1")
    assert by_port.port_context == PortContext(port_id="p1")


def test_context_presence_validation_failure_when_missing() -> None:
    with pytest.raises(ValidationError):
        _base_event(element_id=None, node_context=None, port_context=None)


def test_witness_validation_and_normalization() -> None:
    event = _base_event(witness={"z": 1, "a": {"y": 2, "x": 3}})
    assert event.witness == {"a": {"x": 3, "y": 2}, "z": 1}

    with pytest.raises(ValidationError):
        _base_event(witness={"bad": object()})


def test_models_are_immutable() -> None:
    event = _base_event(node_context=NodeContext(node_id="n1"))

    with pytest.raises(ValidationError):
        event.message = "changed"

    with pytest.raises(ValidationError):
        event.node_context = NodeContext(node_id="n2")


def test_model_dump_json_is_reproducible_for_equal_inputs() -> None:
    left = _base_event(
        witness={"outer": {"b": 2, "a": 1}, "z": [2, 1]},
        frequency_index=3,
        sweep_index=1,
    )
    right = _base_event(
        witness={"z": [2, 1], "outer": {"a": 1, "b": 2}},
        frequency_index=3,
        sweep_index=1,
    )

    assert left.model_dump_json() == right.model_dump_json()

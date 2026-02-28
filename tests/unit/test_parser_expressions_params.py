from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rfmna.parser import (
    ParseError,
    ParseErrorCode,
    extract_dependencies,
    resolve_parameters,
)

pytestmark = pytest.mark.unit


def test_extract_dependencies_sorted_unique() -> None:
    assert extract_dependencies("b + a + b") == ("a", "b")


def test_resolve_parameters_basic_literals_and_chain() -> None:
    resolved = resolve_parameters(
        {
            "c": "-b + +4",
            "b": "(a + 2) * 3",
            "a": "1.0",
        }
    )
    assert resolved.items == (("a", 1.0), ("b", 9.0), ("c", -5.0))
    assert resolved["b"] == resolved.items[1][1]
    assert resolved.as_dict() == {"a": 1.0, "b": 9.0, "c": -5.0}


def test_override_precedence_replaces_file_before_resolution() -> None:
    resolved = resolve_parameters(
        file_params={"a": "1.0", "b": "a + 2"},
        overrides={"a": "10.0"},
    )
    assert resolved.items == (("a", 10.0), ("b", 12.0))


def test_override_can_break_dependency_chain_deterministically() -> None:
    resolved = resolve_parameters(
        file_params={"a": "b + 1", "b": "2"},
        overrides={"a": "5"},
    )
    assert resolved.items == (("a", 5.0), ("b", 2.0))


def test_cycle_detection_self_cycle_has_exact_code_and_witness() -> None:
    with pytest.raises(ParseError) as exc_info:
        resolve_parameters({"a": "a + 1"})
    detail = exc_info.value.detail
    assert detail.code == ParseErrorCode.E_MODEL_PARAM_CYCLE.value
    assert detail.message == "parameter dependency cycle detected: a -> a"
    assert detail.witness == ("a",)


def test_cycle_detection_multi_node_cycle_has_deterministic_witness() -> None:
    with pytest.raises(ParseError) as exc_info:
        resolve_parameters({"a": "b + 1", "b": "c + 1", "c": "a + 1"})
    detail = exc_info.value.detail
    assert detail.code == ParseErrorCode.E_MODEL_PARAM_CYCLE.value
    assert detail.message == "parameter dependency cycle detected: a -> b -> c -> a"
    assert detail.witness == ("a", "b", "c")


def test_unknown_symbol_is_structured_and_deterministic() -> None:
    with pytest.raises(ParseError) as exc_info:
        resolve_parameters({"a": "b + 1"})
    detail = exc_info.value.detail
    assert detail.code == ParseErrorCode.E_PARSE_PARAM_UNDEFINED.value
    assert detail.message == "undefined parameter reference: b"
    assert detail.input_text == "a"
    assert detail.witness == ("b",)


def test_nonfinite_result_is_rejected() -> None:
    with pytest.raises(ParseError) as exc_info:
        resolve_parameters({"a": "1e308 * 1e308"})
    detail = exc_info.value.detail
    assert detail.code == ParseErrorCode.E_PARSE_PARAM_NONFINITE.value
    assert detail.message == "parameter expression must evaluate to a finite value"
    assert detail.input_text == "1e308 * 1e308"


def test_invalid_expression_syntax_or_forbidden_nodes_are_rejected() -> None:
    with pytest.raises(ParseError) as expr_error:
        resolve_parameters({"a": "sin(1)"})
    assert expr_error.value.detail.code == ParseErrorCode.E_PARSE_EXPR_INVALID.value

    with pytest.raises(ParseError) as syntax_error:
        resolve_parameters({"a": "1 + "})
    assert syntax_error.value.detail.code == ParseErrorCode.E_PARSE_EXPR_INVALID.value


def test_resolved_parameters_are_frozen() -> None:
    resolved = resolve_parameters({"a": 1.0})
    with pytest.raises(FrozenInstanceError):
        resolved.items = ()


def test_resolution_is_deterministic_across_insertion_orders_and_repeated_runs() -> None:
    first = resolve_parameters({"b": "a + 3", "a": "2", "c": "b * 2"})
    second = resolve_parameters({"c": "b * 2", "a": "2", "b": "a + 3"})

    assert first.items == second.items
    assert first.to_canonical_json() == second.to_canonical_json()

    baseline = first.to_canonical_json()
    for _ in range(20):
        assert (
            resolve_parameters({"b": "a + 3", "a": "2", "c": "b * 2"}).to_canonical_json()
            == baseline
        )

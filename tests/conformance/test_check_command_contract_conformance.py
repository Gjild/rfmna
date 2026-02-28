from __future__ import annotations

import json
from pathlib import Path
from random import Random

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage
from rfmna.parser import PreflightInput
from rfmna.sweep_engine import SweepLayout

pytestmark = pytest.mark.conformance

runner = CliRunner()
_SCHEMA_ID = "docs/spec/schemas/check_output_v1.json"
_EXIT_FAIL = 2


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / _SCHEMA_ID


def _bundle() -> cli_main.CliDesignBundle:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, np.ndarray]:
        del index, frequency_hz
        raise AssertionError("assemble_point not expected in check conformance tests")

    return cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=(1.0,),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )


def _is_type(value: object, expected: str) -> bool:
    checkers: dict[str, bool] = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if expected not in checkers:
        raise AssertionError(f"unsupported schema type token: {expected}")
    return checkers[expected]


def _resolve_ref(root_schema: dict[str, object], ref: str) -> dict[str, object]:
    if not ref.startswith("#/"):
        raise AssertionError(f"unsupported non-local $ref: {ref}")
    current: object = root_schema
    for token in ref[2:].split("/"):
        assert isinstance(current, dict), ref
        current = current[token]
    assert isinstance(current, dict), ref
    return current


def _resolve_schema(
    schema: dict[str, object], *, root_schema: dict[str, object]
) -> dict[str, object]:
    if "$ref" not in schema:
        return schema
    ref = schema["$ref"]
    assert isinstance(ref, str), "$ref must be a string"
    return _resolve_schema(_resolve_ref(root_schema, ref), root_schema=root_schema)


def _validate_type_constraint(value: object, schema: dict[str, object], *, path: str) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        assert _is_type(value, schema_type), f"{path}: expected {schema_type}"
        return
    if isinstance(schema_type, list):
        assert any(_is_type(value, expected) for expected in schema_type), (
            f"{path}: expected one of {schema_type}"
        )


def _validate_scalar_constraints(value: object, schema: dict[str, object], *, path: str) -> None:
    if "const" in schema:
        assert value == schema["const"], f"{path}: const mismatch"
    if "enum" in schema:
        enum_values = schema["enum"]
        assert isinstance(enum_values, list), path
        assert value in enum_values, f"{path}: value not in enum"
    if isinstance(value, str) and "minLength" in schema:
        min_length = schema["minLength"]
        assert isinstance(min_length, int), path
        assert len(value) >= min_length, f"{path}: minLength violation"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        minimum = schema["minimum"]
        assert isinstance(minimum, (int, float)), path
        assert value >= minimum, f"{path}: minimum violation"


def _validate_array(
    value: object,
    schema: dict[str, object],
    *,
    root_schema: dict[str, object],
    path: str,
) -> None:
    if not isinstance(value, list):
        return
    items_schema = schema.get("items")
    if not isinstance(items_schema, dict):
        return
    for index, item in enumerate(value):
        _validate_json_schema(
            item,
            items_schema,
            root_schema=root_schema,
            path=f"{path}[{index}]",
        )


def _required_keys(schema: dict[str, object]) -> tuple[str, ...]:
    required_raw = schema.get("required", [])
    if not isinstance(required_raw, list):
        return ()
    keys: list[str] = []
    for key in required_raw:
        assert isinstance(key, str), "required key must be a string"
        keys.append(key)
    return tuple(keys)


def _properties(schema: dict[str, object]) -> dict[str, dict[str, object]]:
    properties_raw = schema.get("properties", {})
    if not isinstance(properties_raw, dict):
        return {}
    typed: dict[str, dict[str, object]] = {}
    for key, value in properties_raw.items():
        assert isinstance(key, str), "property key must be a string"
        assert isinstance(value, dict), f"property schema {key} must be an object"
        typed[key] = value
    return typed


def _validate_any_of(
    value: dict[str, object],
    schema: dict[str, object],
    *,
    root_schema: dict[str, object],
    path: str,
) -> None:
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return
    for variant in any_of:
        assert isinstance(variant, dict), path
        try:
            _validate_json_schema(value, variant, root_schema=root_schema, path=path)
            return
        except AssertionError:
            continue
    raise AssertionError(f"{path}: anyOf did not match")


def _validate_object(
    value: object,
    schema: dict[str, object],
    *,
    root_schema: dict[str, object],
    path: str,
) -> None:
    if not isinstance(value, dict):
        return
    properties = _properties(schema)
    for required_key in _required_keys(schema):
        assert required_key in value, f"{path}: missing required key {required_key}"

    additional = schema.get("additionalProperties", True)
    if additional is False:
        allowed = set(properties)
        for key in value:
            assert key in allowed, f"{path}: unexpected key {key}"

    for key, child in value.items():
        if key in properties:
            _validate_json_schema(
                child,
                properties[key],
                root_schema=root_schema,
                path=f"{path}.{key}",
            )
            continue
        if isinstance(additional, dict):
            _validate_json_schema(
                child,
                additional,
                root_schema=root_schema,
                path=f"{path}.{key}",
            )

    _validate_any_of(value, schema, root_schema=root_schema, path=path)


def _validate_json_schema(
    value: object,
    schema: dict[str, object],
    *,
    root_schema: dict[str, object],
    path: str = "$",
) -> None:
    resolved = _resolve_schema(schema, root_schema=root_schema)
    _validate_type_constraint(value, resolved, path=path)
    _validate_scalar_constraints(value, resolved, path=path)
    _validate_array(value, resolved, root_schema=root_schema, path=path)
    _validate_object(value, resolved, root_schema=root_schema, path=path)


def _check_result_payload(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: tuple[DiagnosticEvent, ...],
) -> tuple[int, dict[str, object], str]:
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: _bundle())
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: diagnostics)
    result = runner.invoke(cli_main.app, ["check", "d.net", "--format", "json"])
    return (result.exit_code, json.loads(result.stdout), result.stdout)


def test_check_schema_artifact_location_and_version_are_canonical() -> None:
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    assert schema["$id"] == _SCHEMA_ID
    assert schema["properties"]["schema"]["const"] == _SCHEMA_ID
    assert schema["properties"]["schema_version"]["const"] == 1


def test_check_json_output_validates_against_canonical_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = (
        DiagnosticEvent(
            code="W_NUM_ILL_CONDITIONED",
            severity=Severity.WARNING,
            message="warn",
            suggested_action="act",
            solver_stage=SolverStage.SOLVE,
            element_id="N1",
            witness={"b": 2, "a": 1},
        ),
    )
    exit_code, payload, _ = _check_result_payload(monkeypatch, diagnostics)
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    _validate_json_schema(payload, schema, root_schema=schema)

    assert exit_code == 0
    assert payload["schema"] == _SCHEMA_ID
    assert payload["status"] == "pass"
    assert payload["exit_code"] == 0
    assert payload["diagnostics"][0]["witness"] == {"a": 1, "b": 2}


def test_check_json_deterministic_ordering_and_witness_stability_under_permutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = (
        DiagnosticEvent(
            code="W_NUM_ILL_CONDITIONED",
            severity=Severity.WARNING,
            message="warn-2",
            suggested_action="act",
            solver_stage=SolverStage.SOLVE,
            element_id="E2",
            witness={"y": 2, "x": 1},
        ),
        DiagnosticEvent(
            code="E_TOPO_FLOATING_NODE",
            severity=Severity.ERROR,
            message="err-1",
            suggested_action="act",
            solver_stage=SolverStage.PREFLIGHT,
            element_id="E1",
            witness={"z": {"b": 2, "a": 1}},
        ),
    )
    baseline_payload: dict[str, object] | None = None
    baseline_stdout: str | None = None
    rng = Random(0)

    for _ in range(80):
        current = list(base)
        rng.shuffle(current)
        exit_code, payload, stdout = _check_result_payload(monkeypatch, tuple(current))
        assert exit_code == _EXIT_FAIL
        if baseline_payload is None:
            baseline_payload = payload
            baseline_stdout = stdout
            continue
        assert payload == baseline_payload
        assert stdout == baseline_stdout

    assert baseline_payload is not None
    diagnostics = baseline_payload["diagnostics"]
    assert isinstance(diagnostics, list)
    assert [diag["code"] for diag in diagnostics if isinstance(diag, dict)] == [
        "E_TOPO_FLOATING_NODE",
        "W_NUM_ILL_CONDITIONED",
    ]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rfmna.diagnostics import CANONICAL_DIAGNOSTIC_CATALOG, DiagnosticEvent, Severity, SolverStage
from rfmna.parser import (
    DESIGN_BUNDLE_SCHEMA_ID,
    DesignBundleLoadError,
    load_design_bundle_document,
)
from rfmna.parser.design_bundle import (
    canonical_bundle_parse_product_json,
    hash_bundle_parse_product,
    parse_design_bundle_document,
)

pytestmark = pytest.mark.conformance


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_bundle(tmp_path: Path, payload: dict[str, object], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _hierarchy_payload() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "elements": [
                {
                    "id": "Rtop",
                    "kind": "R",
                    "nodes": ["n1", "0"],
                    "params": {"resistance_ohm": 50.0},
                }
            ],
            "macros": [
                {
                    "id": "bias block",
                    "kind": "R",
                    "node_formals": ["p", "n"],
                    "params": {"resistance_ohm": 100.0},
                },
                {
                    "id": "load cell",
                    "kind": "R",
                    "node_formals": ["p", "n"],
                    "params": {"resistance_ohm": 50.0},
                },
            ],
            "subcircuits": [
                {
                    "id": "stage-b",
                    "ports": ["in", "out", "0"],
                    "elements": [],
                    "instances": [
                        {
                            "id": "Xload",
                            "instance_type": "macro",
                            "of": "LOAD-CELL",
                            "nodes": ["out", "0"],
                        }
                    ],
                },
                {
                    "id": "stage-a",
                    "ports": ["in", "out", "0"],
                    "elements": [
                {
                    "id": "Rbias",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
                {
                    "id": "Rbleed",
                    "kind": "R",
                    "nodes": ["in", "0"],
                    "params": {"resistance_ohm": 125.0},
                }
            ],
                    "instances": [
                        {
                            "id": "XstageB",
                            "instance_type": "subcircuit",
                            "of": "stage-b",
                            "nodes": ["in", "out", "0"],
                        },
                        {
                            "id": "Xbias",
                            "instance_type": "macro",
                            "of": "bias_block",
                            "nodes": ["out", "0"],
                        },
                    ],
                },
            ],
            "instances": [
                {
                    "id": "Xtop-load",
                    "instance_type": "macro",
                    "of": "LOAD-CELL",
                    "nodes": ["n1", "0"],
                },
                {
                    "id": "Xtop-stage",
                    "instance_type": "subcircuit",
                    "of": "stage-a",
                    "nodes": ["n1", "n2", "0"],
                },
            ],
        },
        "analysis": {
            "type": "ac",
            "frequency_sweep": {
                "mode": "linear",
                "start": {"value": 1.0, "unit": "Hz"},
                "stop": {"value": 1.0, "unit": "Hz"},
                "points": 1,
            }
        },
    }


def _schema_preexisting_v1_surface(schema: dict[str, object]) -> dict[str, object]:
    projection = json.loads(json.dumps(schema))
    assert isinstance(projection, dict)
    defs = projection["$defs"]
    assert isinstance(defs, dict)
    for key in ("hierarchy_instance_type", "hierarchy_instance", "macro", "subcircuit"):
        defs.pop(key, None)
    design_def = defs["design"]
    assert isinstance(design_def, dict)
    design_properties = design_def["properties"]
    assert isinstance(design_properties, dict)
    for key in ("macros", "subcircuits", "instances"):
        design_properties.pop(key, None)
    return projection


def _assert_hierarchy_diagnostic(
    event: DiagnosticEvent,
    *,
    code: str,
    element_id: str | None,
    witness: dict[str, object],
) -> None:
    assert event.code == code
    assert event.severity == Severity.ERROR
    assert event.solver_stage == SolverStage.PARSE
    assert event.element_id == element_id
    assert event.message
    assert event.suggested_action == CANONICAL_DIAGNOSTIC_CATALOG[code].suggested_action
    assert event.witness == witness


def test_additive_v1_schema_evolution_keeps_existing_required_fields_and_defaults() -> None:
    schema = json.loads(
        (_repo_root() / "docs/spec/schemas/design_bundle_v1.json").read_text(encoding="utf-8")
    )

    assert schema["$id"] == DESIGN_BUNDLE_SCHEMA_ID
    assert schema["properties"]["schema"]["const"] == DESIGN_BUNDLE_SCHEMA_ID
    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["required"] == ["schema", "schema_version", "design", "analysis"]
    assert schema["$defs"]["design"]["required"] == ["reference_node", "elements"]
    assert schema["$defs"]["analysis"]["required"] == ["type", "frequency_sweep"]
    assert tuple(sorted(schema["$defs"]["design"]["properties"])) == (
        "elements",
        "instances",
        "macros",
        "nodes",
        "parameters",
        "ports",
        "reference_node",
        "subcircuits",
    )
    assert schema["$defs"]["design"]["properties"]["macros"]["items"]["$ref"] == "#/$defs/macro"
    assert schema["$defs"]["design"]["properties"]["subcircuits"]["items"]["$ref"] == (
        "#/$defs/subcircuit"
    )
    assert schema["$defs"]["design"]["properties"]["instances"]["items"]["$ref"] == (
        "#/$defs/hierarchy_instance"
    )
    projection = _schema_preexisting_v1_surface(schema)
    projection_json = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert hashlib.sha256(projection_json.encode("utf-8")).hexdigest() == (
        "74df26da8d1802db3827cf2e14cb4d69bfeae5fb6f23d5dd27a5f5396f0501f6"
    )
    evolution_doc = (_repo_root() / "docs/dev/p3_02_design_bundle_schema_evolution.md").read_text(
        encoding="utf-8"
    )
    assert "P3-02-CID-001" in evolution_doc
    assert "P3-02-CID-001A" in evolution_doc
    assert "P3-02-CID-002" in evolution_doc
    assert "P3-02-CID-003" in evolution_doc
    assert "Frozen artifact `#9`" in evolution_doc
    assert "Frozen artifact `#10`" in evolution_doc
    assert "explicit no semantic change" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_additive_v1_schema_evolution_keeps_existing_required_fields_and_defaults" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_canonical_parse_product_preserves_flat_v1_order_semantics" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_canonical_parse_product_normalizes_supported_kind_aliases" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_duplicate_definition_diagnostic_is_taxonomy_complete_and_deterministic" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_mixed_duplicate_and_conflicting_hierarchy_diagnostics_are_taxonomy_complete_and_deterministic" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_illegal_hierarchy_interface_declarations_are_taxonomy_complete_and_deterministic" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_parse_surface_preserves_flat_validation_contract" in evolution_doc
    assert "illegal local namespace collisions" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_macro_instance_requires_complete_composed_model_before_unsupported" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_instantiated_subcircuit_override_validation_applies_to_target_body" in evolution_doc
    assert "tests/conformance/test_design_bundle_hierarchy_conformance.py::test_subcircuit_instance_override_scope_can_reference_overridden_siblings" in evolution_doc


def test_canonical_hierarchy_parse_product_is_order_and_hash_stable(tmp_path: Path) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()
    design = candidate["design"]
    assert isinstance(design, dict)
    macros = design["macros"]
    subcircuits = design["subcircuits"]
    top_level_instances = design["instances"]
    assert isinstance(macros, list)
    assert isinstance(subcircuits, list)
    assert isinstance(top_level_instances, list)
    design["macros"] = list(reversed(macros))
    stage_a = subcircuits[1]
    assert isinstance(stage_a, dict)
    instances = stage_a["instances"]
    assert isinstance(instances, list)
    stage_a["instances"] = list(reversed(instances))
    design["subcircuits"] = list(reversed(subcircuits))
    design["instances"] = list(reversed(top_level_instances))

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) == canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) == hash_bundle_parse_product(
        baseline_document
    )


def test_canonical_parse_product_preserves_subcircuit_local_element_order_semantics(
    tmp_path: Path,
) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()
    candidate_design = candidate["design"]
    assert isinstance(candidate_design, dict)
    candidate_subcircuits = candidate_design["subcircuits"]
    assert isinstance(candidate_subcircuits, list)
    stage_a = candidate_subcircuits[1]
    assert isinstance(stage_a, dict)
    elements = stage_a["elements"]
    assert isinstance(elements, list)
    stage_a["elements"] = list(reversed(elements))

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="nested_element_order_baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="nested_element_order_candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) != canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) != hash_bundle_parse_product(
        baseline_document
    )


def test_canonical_hierarchy_parse_product_normalizes_instance_identifiers(tmp_path: Path) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()

    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    baseline_instances = baseline_design["instances"]
    candidate_instances = candidate_design["instances"]
    assert isinstance(baseline_instances, list)
    assert isinstance(candidate_instances, list)
    baseline_instances[0]["id"] = "X-load"
    candidate_instances[0]["id"] = "x load"

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="normalized_id_baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="normalized_id_candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) == canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) == hash_bundle_parse_product(
        baseline_document
    )


def test_canonical_parse_product_normalizes_supported_kind_aliases(tmp_path: Path) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()

    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    baseline_elements = baseline_design["elements"]
    candidate_elements = candidate_design["elements"]
    baseline_macros = baseline_design["macros"]
    candidate_macros = candidate_design["macros"]
    assert isinstance(baseline_elements, list)
    assert isinstance(candidate_elements, list)
    assert isinstance(baseline_macros, list)
    assert isinstance(candidate_macros, list)
    baseline_elements[0]["kind"] = "R"
    candidate_elements[0]["kind"] = "RES"
    baseline_macros[0]["kind"] = "R"
    candidate_macros[0]["kind"] = "RESISTOR"

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="canonical_kind_baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="canonical_kind_candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) == canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) == hash_bundle_parse_product(
        baseline_document
    )


def test_canonical_parse_product_preserves_flat_v1_order_semantics(tmp_path: Path) -> None:
    baseline = {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "nodes": ["n2", "n1"],
            "elements": [
                {
                    "id": "R2",
                    "kind": "R",
                    "nodes": ["n2", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
                {
                    "id": "R1",
                    "kind": "R",
                    "nodes": ["n1", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
            ],
            "ports": [
                {"id": "P2", "p_plus": "n2", "p_minus": "0", "z0_ohm": 50.0},
                {"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 50.0},
            ],
        },
        "analysis": {
            "type": "ac",
            "frequency_sweep": {
                "mode": "linear",
                "start": {"value": 1.0, "unit": "Hz"},
                "stop": {"value": 1.0, "unit": "Hz"},
                "points": 1,
            },
        },
    }
    candidate = json.loads(json.dumps(baseline))
    candidate_design = candidate["design"]
    assert isinstance(candidate_design, dict)
    nodes = candidate_design["nodes"]
    elements = candidate_design["elements"]
    ports = candidate_design["ports"]
    assert isinstance(nodes, list)
    assert isinstance(elements, list)
    assert isinstance(ports, list)
    candidate_design["nodes"] = list(reversed(nodes))
    candidate_design["elements"] = list(reversed(elements))
    candidate_design["ports"] = list(reversed(ports))

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="flat_order_baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="flat_order_candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) != canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) != hash_bundle_parse_product(
        baseline_document
    )


def test_parse_design_bundle_document_exposes_deferred_parameter_sweeps_for_parse_surface(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    analysis = payload["analysis"]
    assert isinstance(analysis, dict)
    analysis["parameter_sweeps"] = [{"parameter": "bias", "values": [1.0, 2.0]}]
    path = _write_bundle(tmp_path, payload, name="parse_exclusion_contract.json")

    document = parse_design_bundle_document(path)

    assert tuple(sweep.parameter for sweep in document.parameter_sweeps) == ("bias",)


def test_hierarchy_macro_instance_requires_complete_composed_model_before_unsupported(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "res-load",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {},
        }
    ]
    design["subcircuits"] = []
    design["instances"] = [
        {
            "id": "Xbad",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="macro_composition_required_param.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == ("E_CLI_DESIGN_VALUE_INVALID",)
    assert diagnostics[0].witness == {"path": "design.instances[Xbad]"}


def test_instantiated_subcircuit_override_validation_applies_to_target_body(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "inner",
            "ports": ["in", "out", "0"],
            "parameters": {"r": 50.0},
            "elements": [
                {
                    "id": "Rsub",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": "r"},
                }
            ],
            "instances": [],
        },
        {
            "id": "outer",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xinner",
                    "instance_type": "subcircuit",
                    "of": "inner",
                    "nodes": ["in", "out", "0"],
                    "params": {"r": 0.0},
                }
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="instantiated_subcircuit_override_validation.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == ("E_CLI_DESIGN_VALUE_INVALID",)
    assert diagnostics[0].witness == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
        "path": "design.subcircuits[outer].instances[Xinner].target[inner].elements[Rsub]",
    }


def test_subcircuit_instance_override_scope_can_reference_overridden_siblings(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "child",
            "ports": ["in", "out", "0"],
            "parameters": {"a": 1.0, "b": "a + 1"},
            "elements": [
                {
                    "id": "Rchild",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": "b"},
                }
            ],
            "instances": [],
        },
        {
            "id": "outer",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xchild",
                    "instance_type": "subcircuit",
                    "of": "child",
                    "nodes": ["in", "out", "0"],
                    "params": {"a": 2.0, "b": "a + 1"},
                }
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="override_scope_siblings_conformance.json")

    parsed = load_design_bundle_document(path)

    assert tuple(element.element_id for element in parsed.ir.elements) == ("Rtop",)


def test_macro_template_instance_overrides_are_schema_and_parse_compatible(tmp_path: Path) -> None:
    schema = json.loads(
        (_repo_root() / "docs/spec/schemas/design_bundle_v1.json").read_text(encoding="utf-8")
    )
    assert schema["$defs"]["macro"]["required"] == ["id", "kind", "node_formals"]

    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "res-template",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {},
        }
    ]
    design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xtemplated",
                    "instance_type": "macro",
                    "of": "res-template",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 125.0},
                }
            ],
        }
    ]
    design["instances"] = []

    document = parse_design_bundle_document(_write_bundle(tmp_path, payload, name="macro_template.json"))

    assert document.macros[0].params == {}
    assert dict(document.subcircuits[0].instances[0].params) == {"resistance_ohm": 125.0}

def test_parse_surface_preserves_flat_validation_contract(tmp_path: Path) -> None:
    invalid_element = _hierarchy_payload()
    invalid_element_design = invalid_element["design"]
    assert isinstance(invalid_element_design, dict)
    invalid_element_design["elements"] = [
        {
            "id": "Rtop",
            "kind": "R",
            "nodes": ["n1", "0"],
            "params": {"resistance_ohm": 0.0},
        }
    ]
    invalid_element_design["instances"] = []

    invalid_port = _hierarchy_payload()
    invalid_port_design = invalid_port["design"]
    assert isinstance(invalid_port_design, dict)
    invalid_port_design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 0.0}]

    invalid_sweep = _hierarchy_payload()
    invalid_sweep_analysis = invalid_sweep["analysis"]
    assert isinstance(invalid_sweep_analysis, dict)
    invalid_sweep_frequency = invalid_sweep_analysis["frequency_sweep"]
    assert isinstance(invalid_sweep_frequency, dict)
    invalid_sweep_frequency["mode"] = "log"
    invalid_sweep_start = invalid_sweep_frequency["start"]
    assert isinstance(invalid_sweep_start, dict)
    invalid_sweep_start["value"] = 0.0

    cases = (
        (invalid_element, "parse_invalid_element_conformance.json", "E_CLI_DESIGN_VALUE_INVALID"),
        (invalid_port, "parse_invalid_port_conformance.json", "E_MODEL_PORT_Z0_NONPOSITIVE"),
        (invalid_sweep, "parse_invalid_sweep_conformance.json", "E_CLI_DESIGN_VALUE_INVALID"),
    )
    for payload, name, expected_code in cases:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            parse_design_bundle_document(_write_bundle(tmp_path, payload, name=name))
        assert exc_info.value.diagnostics[0].code == expected_code


def test_canonical_hierarchy_parse_product_is_unicode_normalization_stable(
    tmp_path: Path,
) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()
    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    baseline_design["macros"] = [
        {
            "id": "Å",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 100.0},
        }
    ]
    candidate_design["macros"] = [
        {
            "id": "A\u030A",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 100.0},
        }
    ]
    baseline_design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xunicode",
                    "instance_type": "macro",
                    "of": "Å",
                    "nodes": ["out", "0"],
                }
            ],
        }
    ]
    candidate_design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xunicode",
                    "instance_type": "macro",
                    "of": "A\u030A",
                    "nodes": ["out", "0"],
                }
            ],
        }
    ]
    baseline_design["instances"] = []
    candidate_design["instances"] = []

    baseline_document = parse_design_bundle_document(
        _write_bundle(tmp_path, baseline, name="unicode_baseline.json")
    )
    candidate_document = parse_design_bundle_document(
        _write_bundle(tmp_path, candidate, name="unicode_candidate.json")
    )

    assert canonical_bundle_parse_product_json(candidate_document) == canonical_bundle_parse_product_json(
        baseline_document
    )
    assert hash_bundle_parse_product(candidate_document) == hash_bundle_parse_product(
        baseline_document
    )


def test_hierarchy_diagnostics_are_ordered_and_witness_stable(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xmissing",
            "instance_type": "subcircuit",
            "of": "missing stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    design["subcircuits"] = [
        {
            "id": "stage-b",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XA",
                    "instance_type": "subcircuit",
                    "of": "stage-a",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XB",
                    "instance_type": "subcircuit",
                    "of": "stage-b",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="diagnostics.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
        element_id="hierarchy:STAGE_A",
        witness={"component": ["STAGE_A", "STAGE_B"]},
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
        element_id="Xmissing",
        witness={
            "instance_id": "Xmissing",
            "instance_type": "subcircuit",
            "normalized_target_id": "MISSING_STAGE",
            "scope_id": "design",
            "scope_type": "design",
            "target_id": "missing stage",
        },
    )


def test_hierarchy_value_validation_failures_are_ordered_and_witness_stable(
    tmp_path: Path,
) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()
    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    baseline_design["macros"] = [
        {
            "id": "bad-b",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        },
        {
            "id": "bad-a",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        },
    ]
    candidate_design["macros"] = [
        {
            "id": "bad-a",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        },
        {
            "id": "bad-b",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        },
    ]
    baseline_design["subcircuits"] = []
    candidate_design["subcircuits"] = []
    baseline_design["instances"] = []
    candidate_design["instances"] = []
    baseline_path = _write_bundle(tmp_path, baseline, name="value_validation_order_baseline.json")
    candidate_path = _write_bundle(tmp_path, candidate, name="value_validation_order_candidate.json")

    with pytest.raises(DesignBundleLoadError) as baseline_exc:
        parse_design_bundle_document(baseline_path)
    with pytest.raises(DesignBundleLoadError) as candidate_exc:
        parse_design_bundle_document(candidate_path)

    baseline_diagnostic = baseline_exc.value.diagnostics[0]
    candidate_diagnostic = candidate_exc.value.diagnostics[0]
    assert baseline_diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert baseline_diagnostic.element_id == "bad-a"
    assert baseline_diagnostic.witness == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "bad-a", "resistance_ohm": 0.0},
        "path": "design.macros[bad-a]",
    }
    assert candidate_diagnostic.code == baseline_diagnostic.code
    assert candidate_diagnostic.element_id == baseline_diagnostic.element_id
    assert candidate_diagnostic.witness == baseline_diagnostic.witness


def test_hierarchy_value_validation_reports_each_affected_instance_path_deterministically(
    tmp_path: Path,
) -> None:
    baseline = _hierarchy_payload()
    candidate = _hierarchy_payload()
    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    shared_subcircuits = [
        {
            "id": "inner",
            "ports": ["in", "out", "0"],
            "parameters": {"r": 50.0},
            "elements": [
                {
                    "id": "Rsub",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": "r"},
                }
            ],
            "instances": [],
        },
        {
            "id": "outer",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xb",
                    "instance_type": "subcircuit",
                    "of": "inner",
                    "nodes": ["in", "out", "0"],
                    "params": {"r": 0.0},
                },
                {
                    "id": "Xa",
                    "instance_type": "subcircuit",
                    "of": "inner",
                    "nodes": ["in", "out", "0"],
                    "params": {"r": 0.0},
                },
            ],
        },
    ]
    baseline_design["subcircuits"] = shared_subcircuits
    candidate_design["subcircuits"] = json.loads(json.dumps(shared_subcircuits))
    candidate_subcircuits = candidate_design["subcircuits"]
    assert isinstance(candidate_subcircuits, list)
    outer = candidate_subcircuits[1]
    assert isinstance(outer, dict)
    instances = outer["instances"]
    assert isinstance(instances, list)
    outer["instances"] = list(reversed(instances))
    baseline_design["instances"] = []
    candidate_design["instances"] = []
    baseline_path = _write_bundle(tmp_path, baseline, name="affected_paths_baseline.json")
    candidate_path = _write_bundle(tmp_path, candidate, name="affected_paths_candidate.json")

    with pytest.raises(DesignBundleLoadError) as baseline_exc:
        parse_design_bundle_document(baseline_path)
    with pytest.raises(DesignBundleLoadError) as candidate_exc:
        parse_design_bundle_document(candidate_path)

    baseline_diagnostics = baseline_exc.value.diagnostics
    candidate_diagnostics = candidate_exc.value.diagnostics
    assert tuple(event.code for event in baseline_diagnostics) == (
        "E_CLI_DESIGN_VALUE_INVALID",
        "E_CLI_DESIGN_VALUE_INVALID",
    )
    assert tuple(event.witness for event in baseline_diagnostics) == (
        {
            "issue_code": "E_MODEL_R_NONPOSITIVE",
            "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
            "path": "design.subcircuits[outer].instances[Xa].target[inner].elements[Rsub]",
        },
        {
            "issue_code": "E_MODEL_R_NONPOSITIVE",
            "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
            "path": "design.subcircuits[outer].instances[Xb].target[inner].elements[Rsub]",
        },
    )
    assert tuple(event.code for event in candidate_diagnostics) == tuple(
        event.code for event in baseline_diagnostics
    )
    assert tuple(event.witness for event in candidate_diagnostics) == tuple(
        event.witness for event in baseline_diagnostics
    )


def test_mixed_hierarchy_failures_accumulate_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "R-load",
                    "kind": "R",
                    "nodes": ["mid", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
                {
                    "id": "R load",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
            ],
            "instances": [
                {
                    "id": "Xmissing",
                    "instance_type": "subcircuit",
                    "of": "missing-stage",
                    "nodes": ["in", "out", "0"],
                }
            ],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="mixed_hierarchy_failures.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID",
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
    )


def test_hierarchy_recursion_diagnostics_are_bounded_per_recursive_component(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = []
    design["subcircuits"] = [
        {
            "id": "stage-d",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XB",
                    "instance_type": "subcircuit",
                    "of": "stage-b",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "stage-c",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XA",
                    "instance_type": "subcircuit",
                    "of": "stage-a",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "stage-b",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XC",
                    "instance_type": "subcircuit",
                    "of": "stage-c",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XB",
                    "instance_type": "subcircuit",
                    "of": "stage-b",
                    "nodes": ["in", "out", "0"],
                },
                {
                    "id": "XD",
                    "instance_type": "subcircuit",
                    "of": "stage-d",
                    "nodes": ["in", "out", "0"],
                },
            ],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="branched_cycle_diagnostics.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
        element_id="hierarchy:STAGE_A",
        witness={"component": ["STAGE_A", "STAGE_B", "STAGE_C", "STAGE_D"]},
    )


def test_hierarchy_duplicate_subcircuit_definition_still_emits_recursion_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = []
    design["instances"] = []
    design["subcircuits"] = [
        {
            "id": "stage a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xself",
                    "instance_type": "subcircuit",
                    "of": "stage-a",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "STAGE-A",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="duplicate_subcircuit_recursion_taxonomy.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="hierarchy:STAGE_A",
        witness={
            "definition_type": "subcircuit",
            "normalized_id": "STAGE_A",
            "raw_ids": ["STAGE-A", "stage a"],
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="stage a:Xself",
        witness={
            "definition_type": "subcircuit",
            "instance_id": "Xself",
            "instance_type": "subcircuit",
            "normalized_id": "STAGE_A",
            "normalized_target_id": "STAGE_A",
            "raw_ids": ["STAGE-A", "stage a"],
            "scope_id": "stage a",
            "scope_type": "subcircuit",
            "target_id": "stage-a",
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[2],
        code="E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
        element_id="hierarchy:STAGE_A",
        witness={"component": ["STAGE_A"]},
    )


def test_hierarchy_unsupported_diagnostic_is_taxonomy_complete_and_stable(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    path = _write_bundle(tmp_path, payload, name="hierarchy_unsupported.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == ("E_CLI_DESIGN_HIERARCHY_UNSUPPORTED",)
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_UNSUPPORTED",
        element_id="cli.design_loader",
        witness={
            "instance_count": 2,
            "instance_ids": ["XTOP_LOAD", "XTOP_STAGE"],
        },
    )


def test_hierarchy_unsupported_precedes_flat_model_validation(tmp_path: Path) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 0.0}]
    path = _write_bundle(tmp_path, payload, name="hierarchy_unsupported_precedes_flat_validation.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == ("E_CLI_DESIGN_HIERARCHY_UNSUPPORTED",)
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_UNSUPPORTED",
        element_id="cli.design_loader",
        witness={
            "instance_count": 2,
            "instance_ids": ["XTOP_LOAD", "XTOP_STAGE"],
        },
    )


def test_illegal_hierarchy_interface_declarations_are_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "dup-macro",
            "kind": "R",
            "node_formals": ["p", "P"],
            "params": {"resistance_ohm": 50.0},
        }
    ]
    design["subcircuits"] = [
        {
            "id": "dup-stage",
            "ports": ["in", "in"],
            "elements": [],
            "instances": [],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="illegal_interface_declarations.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
        "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
        element_id="dup-macro",
        witness={
            "declaration_kind": "macro",
            "duplicate_count": 2,
            "field_name": "node_formals",
            "normalized_id": "P",
            "raw_ids": ["P", "p"],
            "scope_id": "dup-macro",
            "scope_type": "macro",
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
        element_id="dup-stage",
        witness={
            "declaration_kind": "subcircuit",
            "duplicate_count": 2,
            "field_name": "ports",
            "normalized_id": "IN",
            "raw_ids": ["in", "in"],
            "scope_id": "dup-stage",
            "scope_type": "subcircuit",
        },
    )


def test_conflicting_subcircuit_local_namespace_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage",
            "ports": ["in", "out"],
            "elements": [
                {
                    "id": "dup",
                    "kind": "R",
                    "nodes": ["out", "in"],
                    "params": {"resistance_ohm": 50.0},
                }
            ],
            "instances": [
                {
                    "id": "DUP",
                    "instance_type": "macro",
                    "of": "bias block",
                    "nodes": ["out", "in"],
                }
            ],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="conflicting_local_namespace_taxonomy.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
        element_id="stage:dup",
        witness={
            "declaration_kind": "subcircuit_local_namespace",
            "declarations": [
                {"declaration_kind": "element", "raw_id": "dup"},
                {"declaration_kind": "instance", "raw_id": "DUP"},
            ],
            "normalized_id": "DUP",
            "scope_id": "stage",
            "scope_type": "subcircuit",
        },
    )


def test_hierarchy_definition_conflict_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "gain cell",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        }
    ]
    design["subcircuits"] = [
        {
            "id": "GAIN-CELL",
            "ports": ["in", "out", "0"],
            "elements": [],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="definition_conflict.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        element_id="hierarchy:GAIN_CELL",
        witness={
            "definitions": [
                {"definition_type": "macro", "raw_id": "gain cell"},
                {"definition_type": "subcircuit", "raw_id": "GAIN-CELL"},
            ],
            "normalized_id": "GAIN_CELL",
        },
    )


def test_hierarchy_reference_to_definition_conflict_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "gain cell",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        }
    ]
    design["subcircuits"] = [
        {
            "id": "GAIN-CELL",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [],
        }
    ]
    design["instances"] = [
        {
            "id": "xgain",
            "instance_type": "subcircuit",
            "of": "gain cell",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="conflict_reference_taxonomy.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        element_id="hierarchy:GAIN_CELL",
        witness={
            "definitions": [
                {"definition_type": "macro", "raw_id": "gain cell"},
                {"definition_type": "subcircuit", "raw_id": "GAIN-CELL"},
            ],
            "normalized_id": "GAIN_CELL",
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        element_id="xgain",
        witness={
            "definitions": [
                {"definition_type": "macro", "raw_id": "gain cell"},
                {"definition_type": "subcircuit", "raw_id": "GAIN-CELL"},
            ],
            "instance_id": "xgain",
            "instance_type": "subcircuit",
            "normalized_id": "GAIN_CELL",
            "normalized_target_id": "GAIN_CELL",
            "scope_id": "design",
            "scope_type": "design",
            "target_id": "gain cell",
        },
    )


def test_hierarchy_duplicate_definition_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "bias block",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        },
        {
            "id": "BIAS-BLOCK",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 75.0},
        },
    ]
    design["subcircuits"] = []
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="duplicate_definition.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="hierarchy:BIAS_BLOCK",
        witness={
            "definition_type": "macro",
            "normalized_id": "BIAS_BLOCK",
            "raw_ids": ["BIAS-BLOCK", "bias block"],
        },
    )


def test_hierarchy_reference_to_duplicate_definition_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "bias block",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        },
        {
            "id": "BIAS-BLOCK",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 75.0},
        },
    ]
    design["subcircuits"] = []
    design["instances"] = [
        {
            "id": "xbias",
            "instance_type": "macro",
            "of": "bias block",
            "nodes": ["n1", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="duplicate_reference_taxonomy.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="hierarchy:BIAS_BLOCK",
        witness={
            "definition_type": "macro",
            "normalized_id": "BIAS_BLOCK",
            "raw_ids": ["BIAS-BLOCK", "bias block"],
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="xbias",
        witness={
            "definition_type": "macro",
            "instance_id": "xbias",
            "instance_type": "macro",
            "normalized_id": "BIAS_BLOCK",
            "normalized_target_id": "BIAS_BLOCK",
            "raw_ids": ["BIAS-BLOCK", "bias block"],
            "scope_id": "design",
            "scope_type": "design",
            "target_id": "bias block",
        },
    )


def test_mixed_duplicate_and_conflicting_hierarchy_diagnostics_are_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "gain cell",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        },
        {
            "id": "GAIN-CELL",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 75.0},
        },
    ]
    design["subcircuits"] = [
        {
            "id": "GAIN_CELL",
            "ports": ["in", "out"],
            "elements": [],
            "instances": [],
        }
    ]
    design["instances"] = [
        {
            "id": "xgain",
            "instance_type": "macro",
            "of": "gain cell",
            "nodes": ["n1", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="mixed_duplicate_conflict_taxonomy.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(event.code for event in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
    )
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        element_id="hierarchy:GAIN_CELL",
        witness={
            "definitions": [
                {"definition_type": "macro", "raw_id": "GAIN-CELL"},
                {"definition_type": "macro", "raw_id": "gain cell"},
                {"definition_type": "subcircuit", "raw_id": "GAIN_CELL"},
            ],
            "normalized_id": "GAIN_CELL",
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[1],
        code="E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        element_id="xgain",
        witness={
            "definitions": [
                {"definition_type": "macro", "raw_id": "GAIN-CELL"},
                {"definition_type": "macro", "raw_id": "gain cell"},
                {"definition_type": "subcircuit", "raw_id": "GAIN_CELL"},
            ],
            "instance_id": "xgain",
            "instance_type": "macro",
            "normalized_id": "GAIN_CELL",
            "normalized_target_id": "GAIN_CELL",
            "scope_id": "design",
            "scope_type": "design",
            "target_id": "gain cell",
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[2],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="hierarchy:GAIN_CELL",
        witness={
            "definition_type": "macro",
            "normalized_id": "GAIN_CELL",
            "raw_ids": ["GAIN-CELL", "gain cell"],
        },
    )
    _assert_hierarchy_diagnostic(
        diagnostics[3],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        element_id="xgain",
        witness={
            "definition_type": "macro",
            "instance_id": "xgain",
            "instance_type": "macro",
            "normalized_id": "GAIN_CELL",
            "normalized_target_id": "GAIN_CELL",
            "raw_ids": ["GAIN-CELL", "gain cell"],
            "scope_id": "design",
            "scope_type": "design",
            "target_id": "gain cell",
        },
    )


def test_hierarchy_invalid_macro_default_param_key_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "res-load",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"bogus": 1.0},
        }
    ]
    design["subcircuits"] = []
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="invalid_macro_default_param_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_VALUE_INVALID",
        element_id="res-load",
        witness={
            "allowed_keys": ["resistance_ohm"],
            "path": "design.macros[res-load].params",
            "unexpected_keys": ["bogus"],
        },
    )


def test_hierarchy_invalid_instance_override_param_key_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xload",
                    "instance_type": "macro",
                    "of": "load cell",
                    "nodes": ["out", "0"],
                    "params": {"bogus": 1.0},
                }
            ],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="invalid_instance_override_param_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_VALUE_INVALID",
        element_id="gain-stage:Xload",
        witness={
            "allowed_keys": ["resistance_ohm"],
            "path": "design.subcircuits[gain-stage].instances[Xload].params",
            "unexpected_keys": ["bogus"],
        },
    )


def test_hierarchy_duplicate_local_element_id_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "Rdup",
                    "kind": "R",
                    "nodes": ["mid", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
                {
                    "id": "Rdup",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
            ],
            "instances": [],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="duplicate_local_element_id.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID",
        element_id="stage-a:Rdup",
        witness={
            "declaration_kind": "element",
            "duplicate_count": 2,
            "normalized_id": "RDUP",
            "raw_ids": ["Rdup", "Rdup"],
            "scope_id": "stage-a",
            "scope_type": "subcircuit",
        },
    )


def test_hierarchy_normalized_duplicate_local_element_id_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "R-load",
                    "kind": "R",
                    "nodes": ["mid", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
                {
                    "id": "R load",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
            ],
            "instances": [],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="normalized_duplicate_local_element_id.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID",
        element_id="stage-a:R load",
        witness={
            "declaration_kind": "element",
            "duplicate_count": 2,
            "normalized_id": "R_LOAD",
            "raw_ids": ["R load", "R-load"],
            "scope_id": "stage-a",
            "scope_type": "subcircuit",
        },
    )


def test_hierarchy_duplicate_instance_id_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xdup",
            "instance_type": "macro",
            "of": "bias block",
            "nodes": ["n1", "0"],
        },
        {
            "id": "Xdup",
            "instance_type": "subcircuit",
            "of": "stage-a",
            "nodes": ["n1", "n2", "0"],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="duplicate_instance_id.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID",
        element_id="Xdup",
        witness={
            "duplicate_count": 2,
            "normalized_id": "XDUP",
            "raw_ids": ["Xdup", "Xdup"],
            "scope_id": "design",
            "scope_type": "design",
        },
    )


def test_hierarchy_instance_arity_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xshort",
                    "instance_type": "subcircuit",
                    "of": "stage-b",
                    "nodes": ["in", "out"],
                }
            ],
        },
        {
            "id": "stage-b",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="instance_arity_invalid.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_INSTANCE_ARITY_INVALID",
        element_id="stage-a:Xshort",
        witness={
            "actual_node_count": 2,
            "expected_node_count": 3,
            "instance_id": "Xshort",
            "instance_type": "subcircuit",
            "normalized_target_id": "STAGE_B",
            "resolved_definition_type": "subcircuit",
            "resolved_raw_id": "stage-b",
            "scope_id": "stage-a",
            "scope_type": "subcircuit",
            "target_id": "stage-b",
        },
    )


def test_hierarchy_reference_type_mismatch_diagnostic_is_taxonomy_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_payload()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "child-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
        },
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "XwrongType",
                    "instance_type": "macro",
                    "of": "child-stage",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="reference_type_mismatch.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert len(diagnostics) == 1
    _assert_hierarchy_diagnostic(
        diagnostics[0],
        code="E_CLI_DESIGN_HIERARCHY_REFERENCE_TYPE_MISMATCH",
        element_id="gain-stage:XwrongType",
        witness={
            "instance_id": "XwrongType",
            "instance_type": "macro",
            "normalized_target_id": "CHILD_STAGE",
            "resolved_definition_type": "subcircuit",
            "resolved_raw_id": "child-stage",
            "scope_id": "gain-stage",
            "scope_type": "subcircuit",
            "target_id": "child-stage",
        },
    )

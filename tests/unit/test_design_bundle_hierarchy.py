from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rfmna.parser import design_bundle as parser_design_bundle_module
from rfmna.parser.design_bundle import (
    DESIGN_BUNDLE_SCHEMA_ID,
    DesignBundleLoadError,
    canonical_bundle_parse_product_json,
    load_design_bundle_document,
    parse_design_bundle_document,
)

pytestmark = pytest.mark.unit


def _hierarchy_bundle() -> dict[str, object]:
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
                    "id": "res-load",
                    "kind": "R",
                    "node_formals": ["p", "n"],
                    "params": {"resistance_ohm": 75.0},
                }
            ],
            "subcircuits": [
                {
                    "id": "gain-stage",
                    "ports": ["in", "out", "0"],
                    "parameters": {"r_bias": 100.0},
                    "elements": [
                        {
                            "id": "Rbias",
                            "kind": "R",
                            "nodes": ["out", "0"],
                            "params": {"resistance_ohm": "r_bias"},
                        }
                    ],
                    "instances": [
                        {
                            "id": "Xload",
                            "instance_type": "macro",
                            "of": "RES LOAD",
                            "nodes": ["out", "0"],
                            "params": {"resistance_ohm": 50.0},
                        }
                    ],
                }
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


def _write_bundle(tmp_path: Path, payload: dict[str, object], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_flat_loader_accepts_unused_hierarchy_definitions(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _hierarchy_bundle(), name="hierarchy_defs_only.json")

    parsed = load_design_bundle_document(path)

    assert tuple(node.node_id for node in parsed.ir.nodes) == ("0", "n1")
    assert tuple(element.element_id for element in parsed.ir.elements) == ("Rtop",)


def test_top_level_hierarchy_instances_are_rejected_until_elaboration_exists(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xstage",
            "instance_type": "subcircuit",
            "of": "gain stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="top_level_instance.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_UNSUPPORTED"
    assert diagnostic.element_id == "cli.design_loader"
    assert diagnostic.witness == {"instance_count": 1, "instance_ids": ["XSTAGE"]}


def test_loader_exclusions_take_precedence_over_top_level_hierarchy_unsupported(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    analysis = payload["analysis"]
    assert isinstance(design, dict)
    assert isinstance(analysis, dict)
    design["instances"] = [
        {
            "id": "Xstage",
            "instance_type": "subcircuit",
            "of": "gain stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    analysis["parameter_sweeps"] = [
        {
            "parameter": "bias",
            "values": [1.0, 2.0],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="top_level_instance_with_exclusion.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
    )


def test_top_level_hierarchy_unsupported_takes_precedence_over_flat_port_validation(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xstage",
            "instance_type": "subcircuit",
            "of": "gain stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 0.0}]
    path = _write_bundle(tmp_path, payload, name="top_level_instance_invalid_port.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_UNSUPPORTED"
    assert diagnostic.element_id == "cli.design_loader"
    assert diagnostic.witness == {"instance_count": 1, "instance_ids": ["XSTAGE"]}


def test_loader_exclusions_take_precedence_over_hierarchy_value_validation(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    analysis = payload["analysis"]
    assert isinstance(design, dict)
    assert isinstance(analysis, dict)
    design["macros"] = [
        {
            "id": "bad-macro",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        }
    ]
    design["subcircuits"] = []
    analysis["parameter_sweeps"] = [
        {
            "parameter": "bias",
            "values": [1.0, 2.0],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="excluded_before_value_validation.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
    )


def test_invalid_top_level_hierarchy_instance_does_not_emit_unsupported_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xbad",
            "instance_type": "subcircuit",
            "of": "missing-stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="invalid_top_level_instance.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
    )


def test_invalid_top_level_hierarchy_instance_override_does_not_emit_unsupported_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xbad",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
            "params": {"resistance_ohm": "missing_param + 1"},
        }
    ]
    path = _write_bundle(tmp_path, payload, name="invalid_top_level_instance_override.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_VALUE_INVALID",
    )
    assert diagnostics[0].witness == {
        "input_text": "missing_param + 1",
        "path": "design.instances[Xbad].params.resistance_ohm",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }


def test_missing_macro_required_param_after_composition_is_rejected_before_unsupported(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    design["instances"] = [
        {
            "id": "Xbad",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="missing_macro_required_param.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("E_CLI_DESIGN_VALUE_INVALID",)
    assert diagnostics[0].witness == {"path": "design.instances[Xbad]"}


def test_instantiated_subcircuit_body_is_revalidated_under_instance_overrides(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="instantiated_subcircuit_revalidation.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == ("E_CLI_DESIGN_VALUE_INVALID",)
    assert diagnostics[0].witness == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
        "path": "design.subcircuits[outer].instances[Xinner].target[inner].elements[Rsub]",
    }


def test_equivalent_invalid_subcircuit_instance_paths_are_all_reported(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
                    "id": "X1",
                    "instance_type": "subcircuit",
                    "of": "inner",
                    "nodes": ["in", "out", "0"],
                    "params": {"r": 0.0},
                },
                {
                    "id": "X2",
                    "instance_type": "subcircuit",
                    "of": "inner",
                    "nodes": ["in", "out", "0"],
                    "params": {"r": 0.0},
                },
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="duplicate_invalid_instance_paths.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_VALUE_INVALID",
        "E_CLI_DESIGN_VALUE_INVALID",
    )
    assert tuple(diagnostic.witness for diagnostic in diagnostics) == (
        {
            "issue_code": "E_MODEL_R_NONPOSITIVE",
            "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
            "path": "design.subcircuits[outer].instances[X1].target[inner].elements[Rsub]",
        },
        {
            "issue_code": "E_MODEL_R_NONPOSITIVE",
            "issue_context": {"element_id": "Rsub", "resistance_ohm": 0.0},
            "path": "design.subcircuits[outer].instances[X2].target[inner].elements[Rsub]",
        },
    )


def test_subcircuit_instance_override_scope_can_reference_overridden_siblings(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="override_scope_siblings.json")

    parsed = load_design_bundle_document(path)

    assert tuple(element.element_id for element in parsed.ir.elements) == ("Rtop",)


def test_parse_design_bundle_document_allows_loader_excluded_parameter_sweeps_for_parse_surface(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    analysis = payload["analysis"]
    assert isinstance(analysis, dict)
    analysis["parameter_sweeps"] = [
        {
            "parameter": "bias",
            "values": [1.0, 2.0],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="parse_only_parameter_sweep.json")

    document = parse_design_bundle_document(path)

    assert tuple(sweep.parameter for sweep in document.parameter_sweeps) == ("bias",)


def test_parse_design_bundle_document_allows_top_level_hierarchy_instances_for_parse_surface(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xstage",
            "instance_type": "subcircuit",
            "of": "gain stage",
            "nodes": ["n1", "n2", "0"],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="parse_only_top_level_instance.json")

    document = parse_design_bundle_document(path)

    assert tuple(instance.instance_id for instance in document.instances) == ("Xstage",)


def test_parse_product_preserves_parameter_sweep_order_semantics(tmp_path: Path) -> None:
    baseline = _hierarchy_bundle()
    candidate = _hierarchy_bundle()
    baseline_analysis = baseline["analysis"]
    candidate_analysis = candidate["analysis"]
    assert isinstance(baseline_analysis, dict)
    assert isinstance(candidate_analysis, dict)
    baseline_analysis["parameter_sweeps"] = [
        {"parameter": "bias_b", "values": [2.0]},
        {"parameter": "bias_a", "values": [1.0]},
    ]
    candidate_analysis["parameter_sweeps"] = [
        {"parameter": "bias_a", "values": [1.0]},
        {"parameter": "bias_b", "values": [2.0]},
    ]
    baseline_path = _write_bundle(tmp_path, baseline, name="internal_flat_sweep_baseline.json")
    candidate_path = _write_bundle(tmp_path, candidate, name="internal_flat_sweep_candidate.json")
    baseline_document = parse_design_bundle_document(baseline_path)
    candidate_document = parse_design_bundle_document(candidate_path)

    assert canonical_bundle_parse_product_json(candidate_document) != canonical_bundle_parse_product_json(
        baseline_document
    )


def test_parse_product_canonicalizes_supported_kind_aliases(tmp_path: Path) -> None:
    baseline = _hierarchy_bundle()
    candidate = _hierarchy_bundle()
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
    assert parser_design_bundle_module.hash_bundle_parse_product(
        candidate_document
    ) == parser_design_bundle_module.hash_bundle_parse_product(baseline_document)


def test_parse_product_surface_is_immutable_after_parse(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path, _hierarchy_bundle(), name="immutable_surface.json")

    document = parse_design_bundle_document(path)

    with pytest.raises(TypeError):
        document.parameters["bias"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        document.macros[0].params["resistance_ohm"] = 100.0  # type: ignore[index]
    with pytest.raises(TypeError):
        document.subcircuits[0].instances[0].params["resistance_ohm"] = 50.0  # type: ignore[index]


def test_loader_rejects_excluded_macro_kind_inside_unused_hierarchy(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "fd-macro",
            "kind": "FD_LINEAR",
            "node_formals": ["p", "n"],
            "params": {"numerator": 1.0},
        }
    ]
    design["subcircuits"] = []
    path = _write_bundle(tmp_path, payload, name="excluded_macro.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
    )
    assert diagnostics[0].element_id == "fd-macro"
    assert diagnostics[0].witness["scope_type"] == "macro"


def test_loader_rejects_excluded_subcircuit_element_kind_inside_unused_hierarchy(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "FD1",
                    "kind": "FD_LINEAR",
                    "nodes": ["out", "0"],
                    "params": {"numerator": 1.0},
                }
            ],
            "instances": [],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="excluded_subcircuit_element.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
    )
    assert diagnostics[0].element_id == "gain-stage:FD1"
    assert diagnostics[0].witness["scope_type"] == "subcircuit"
    assert diagnostics[0].witness["scope_id"] == "gain-stage"


def test_loader_rejects_invalid_unused_subcircuit_element_expression(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "parameters": {"r_bias": 100.0},
            "elements": [
                {
                    "id": "Rbias",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": "missing_param + 1"},
                }
            ],
            "instances": [],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="invalid_unused_subcircuit_element.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.element_id == "gain-stage:Rbias"
    assert diagnostic.witness == {
        "input_text": "missing_param + 1",
        "path": "design.subcircuits[gain-stage].elements[Rbias].params.resistance_ohm",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }


def test_parse_surface_rejects_invalid_flat_element_model(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    elements = design["elements"]
    assert isinstance(elements, list)
    element = elements[0]
    assert isinstance(element, dict)
    params = element["params"]
    assert isinstance(params, dict)
    params["resistance_ohm"] = 0.0
    path = _write_bundle(tmp_path, payload, name="parse_invalid_flat_element.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    assert exc_info.value.diagnostics[0].code == "E_CLI_DESIGN_VALUE_INVALID"


def test_parse_surface_rejects_invalid_flat_port_z0(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 0.0}]
    path = _write_bundle(tmp_path, payload, name="parse_invalid_flat_port_z0.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    assert exc_info.value.diagnostics[0].code == "E_MODEL_PORT_Z0_NONPOSITIVE"


def test_parse_surface_rejects_invalid_flat_frequency_sweep(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    analysis = payload["analysis"]
    assert isinstance(analysis, dict)
    frequency_sweep = analysis["frequency_sweep"]
    assert isinstance(frequency_sweep, dict)
    frequency_sweep["mode"] = "log"
    start = frequency_sweep["start"]
    assert isinstance(start, dict)
    start["value"] = 0.0
    path = _write_bundle(tmp_path, payload, name="parse_invalid_flat_frequency_sweep.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    assert exc_info.value.diagnostics[0].code == "E_CLI_DESIGN_VALUE_INVALID"


@pytest.mark.parametrize(
    ("loader", "expected_type"),
    (
        (parse_design_bundle_document, "parse"),
        (load_design_bundle_document, "load"),
    ),
)
def test_unused_invalid_macro_defaults_fail_value_validation(
    tmp_path: Path,
    loader: Callable[[Path], object],
    expected_type: str,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "res-load",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 0.0},
        }
    ]
    design["subcircuits"] = []
    path = _write_bundle(tmp_path, payload, name=f"invalid_unused_macro_default_{expected_type}.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        loader(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.element_id == "res-load"
    assert diagnostic.witness == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "res-load", "resistance_ohm": 0.0},
        "path": "design.macros[res-load]",
    }


def test_loader_rejects_invalid_macro_default_param_key(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="invalid_macro_param_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.element_id == "res-load"
    assert diagnostic.witness == {
        "allowed_keys": ["resistance_ohm"],
        "path": "design.macros[res-load].params",
        "unexpected_keys": ["bogus"],
    }


def test_loader_rejects_invalid_macro_instance_override_param_key(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
                    "of": "res-load",
                    "nodes": ["out", "0"],
                    "params": {"bogus": 1.0},
                }
            ],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="invalid_macro_instance_override_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.witness == {
        "allowed_keys": ["resistance_ohm"],
        "path": "design.subcircuits[gain-stage].instances[Xload].params",
        "unexpected_keys": ["bogus"],
    }


def test_invalid_macro_defaults_fail_in_stable_order_across_permutations(tmp_path: Path) -> None:
    baseline = _hierarchy_bundle()
    candidate = _hierarchy_bundle()
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
    baseline_path = _write_bundle(tmp_path, baseline, name="invalid_macro_defaults_baseline.json")
    candidate_path = _write_bundle(tmp_path, candidate, name="invalid_macro_defaults_candidate.json")

    with pytest.raises(DesignBundleLoadError) as baseline_exc:
        parse_design_bundle_document(baseline_path)
    with pytest.raises(DesignBundleLoadError) as candidate_exc:
        parse_design_bundle_document(candidate_path)

    baseline_diagnostic = baseline_exc.value.diagnostics[0]
    candidate_diagnostic = candidate_exc.value.diagnostics[0]
    assert baseline_diagnostic.element_id == "bad-a"
    assert baseline_diagnostic.witness == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "bad-a", "resistance_ohm": 0.0},
        "path": "design.macros[bad-a]",
    }
    assert candidate_diagnostic.element_id == baseline_diagnostic.element_id
    assert candidate_diagnostic.witness == baseline_diagnostic.witness


def test_nested_invalid_subcircuit_instance_overrides_fail_in_stable_order_across_permutations(
    tmp_path: Path,
) -> None:
    baseline = _hierarchy_bundle()
    candidate = _hierarchy_bundle()
    baseline_design = baseline["design"]
    candidate_design = candidate["design"]
    assert isinstance(baseline_design, dict)
    assert isinstance(candidate_design, dict)
    shared_subcircuits = [
        {
            "id": "leaf-b",
            "ports": ["in", "out"],
            "parameters": {"r_load": 50.0},
            "elements": [
                {
                    "id": "Rload",
                    "kind": "R",
                    "nodes": ["out", "in"],
                    "params": {"resistance_ohm": "r_load"},
                }
            ],
            "instances": [],
        },
        {
            "id": "leaf-a",
            "ports": ["in", "out"],
            "parameters": {"r_load": 50.0},
            "elements": [
                {
                    "id": "Rload",
                    "kind": "R",
                    "nodes": ["out", "in"],
                    "params": {"resistance_ohm": "r_load"},
                }
            ],
            "instances": [],
        },
    ]
    baseline_design["subcircuits"] = [
        *shared_subcircuits,
        {
            "id": "top",
            "ports": ["in", "out"],
            "elements": [],
            "instances": [
                {
                    "id": "Xb",
                    "instance_type": "subcircuit",
                    "of": "leaf-b",
                    "nodes": ["in", "out"],
                    "params": {"r_load": "missing_b + 1"},
                },
                {
                    "id": "Xa",
                    "instance_type": "subcircuit",
                    "of": "leaf-a",
                    "nodes": ["in", "out"],
                    "params": {"r_load": "missing_a + 1"},
                },
            ],
        },
    ]
    candidate_design["subcircuits"] = [
        *shared_subcircuits,
        {
            "id": "top",
            "ports": ["in", "out"],
            "elements": [],
            "instances": [
                {
                    "id": "Xa",
                    "instance_type": "subcircuit",
                    "of": "leaf-a",
                    "nodes": ["in", "out"],
                    "params": {"r_load": "missing_a + 1"},
                },
                {
                    "id": "Xb",
                    "instance_type": "subcircuit",
                    "of": "leaf-b",
                    "nodes": ["in", "out"],
                    "params": {"r_load": "missing_b + 1"},
                },
            ],
        },
    ]
    baseline_design["instances"] = []
    candidate_design["instances"] = []
    baseline_path = _write_bundle(tmp_path, baseline, name="nested_invalid_overrides_baseline.json")
    candidate_path = _write_bundle(tmp_path, candidate, name="nested_invalid_overrides_candidate.json")

    with pytest.raises(DesignBundleLoadError) as baseline_exc:
        parse_design_bundle_document(baseline_path)
    with pytest.raises(DesignBundleLoadError) as candidate_exc:
        parse_design_bundle_document(candidate_path)

    baseline_diagnostic = baseline_exc.value.diagnostics[0]
    candidate_diagnostic = candidate_exc.value.diagnostics[0]
    assert baseline_diagnostic.element_id == "top:Xa"
    assert baseline_diagnostic.witness == {
        "input_text": "missing_a + 1",
        "path": "design.subcircuits[top].instances[Xa].params.r_load",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_a"],
    }
    assert candidate_diagnostic.element_id == baseline_diagnostic.element_id
    assert candidate_diagnostic.witness == baseline_diagnostic.witness


def test_loader_rejects_invalid_subcircuit_instance_override_param_key(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "child-stage",
            "ports": ["in", "out", "0"],
            "parameters": {"r_bias": 100.0},
            "elements": [],
            "instances": [],
        },
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xchild",
                    "instance_type": "subcircuit",
                    "of": "child-stage",
                    "nodes": ["in", "out", "0"],
                    "params": {"bogus": 1.0},
                }
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="invalid_subcircuit_instance_override_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.witness == {
        "allowed_keys": ["r_bias"],
        "path": "design.subcircuits[gain-stage].instances[Xchild].params",
        "unexpected_keys": ["bogus"],
    }


def test_subcircuit_declaration_override_fallback_path_uses_parameters_field() -> None:
    subcircuit = parser_design_bundle_module.BundleSubcircuitDecl(
        subcircuit_id="gain-stage",
        ports=("in", "out", "0"),
        parameters={"r_bias": 100.0},
        elements=(),
        instances=(),
    )
    targets = parser_design_bundle_module._HierarchyValidationTargets(
        macro_targets={},
        subcircuit_targets={},
        memo=parser_design_bundle_module._HierarchyValidationMemo(),
    )

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parser_design_bundle_module._validate_subcircuit_declaration_values(
            subcircuit=subcircuit,
            resolved_parameters={},
            targets=targets,
            parameter_overrides={"r_bias": "missing_param + 1"},
        )

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.witness == {
        "input_text": "missing_param + 1",
        "path": "design.subcircuits[gain-stage].parameters.r_bias",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }


def test_parse_design_bundle_document_rejects_invalid_hierarchy_expression_for_parse_surface(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "parameters": {"r_bias": 100.0},
            "elements": [
                {
                    "id": "Rbias",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": "missing_param + 1"},
                }
            ],
            "instances": [],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="parse_invalid_hierarchy_expression.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_VALUE_INVALID"
    assert diagnostic.witness == {
        "input_text": "missing_param + 1",
        "path": "design.subcircuits[gain-stage].elements[Rbias].params.resistance_ohm",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }


def test_parse_design_bundle_document_allows_excluded_macro_kind_inside_unused_hierarchy_for_parse_surface(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "fd-macro",
            "kind": "FD_LINEAR",
            "node_formals": ["p", "n"],
            "params": {"numerator": 1.0},
        }
    ]
    design["subcircuits"] = []
    path = _write_bundle(tmp_path, payload, name="parse_excluded_macro.json")

    document = parse_design_bundle_document(path)

    assert tuple(macro.macro_id for macro in document.macros) == ("fd-macro",)


def test_load_design_bundle_document_reads_payload_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_bundle(tmp_path, _hierarchy_bundle(), name="single_read.json")
    calls = {"count": 0}
    original_read_payload = parser_design_bundle_module._read_payload

    def _spy(source_path: Path) -> dict[str, object]:
        calls["count"] += 1
        return original_read_payload(source_path)

    monkeypatch.setattr(parser_design_bundle_module, "_read_payload", _spy)

    _ = load_design_bundle_document(path)

    assert calls["count"] == 1


def test_shared_subcircuit_validation_reuses_successful_instantiation_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "leaf",
            "ports": ["in", "out"],
            "parameters": {"r_load": 50.0},
            "elements": [
                {
                    "id": "Rleaf",
                    "kind": "R",
                    "nodes": ["out", "in"],
                    "params": {"resistance_ohm": "r_load"},
                }
            ],
            "instances": [],
        },
        {
            "id": "mid",
            "ports": ["in", "out"],
            "elements": [],
            "instances": [
                {
                    "id": "XleafA",
                    "instance_type": "subcircuit",
                    "of": "leaf",
                    "nodes": ["in", "out"],
                },
                {
                    "id": "XleafB",
                    "instance_type": "subcircuit",
                    "of": "leaf",
                    "nodes": ["in", "out"],
                },
            ],
        },
    ]
    design["instances"] = [
        {
            "id": "XmidA",
            "instance_type": "subcircuit",
            "of": "mid",
            "nodes": ["n1", "n2"],
        },
        {
            "id": "XmidB",
            "instance_type": "subcircuit",
            "of": "mid",
            "nodes": ["n1", "n2"],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="shared_subcircuit_validation.json")
    original_validate = parser_design_bundle_module._validate_hierarchy_element_values
    counts = {"leaf_calls": 0}

    def _spy_validate_hierarchy_element_values(**kwargs: object) -> None:
        element = kwargs["element"]
        path_prefix = kwargs["path_prefix"]
        assert isinstance(element, parser_design_bundle_module.BundleElementDecl)
        assert isinstance(path_prefix, str)
        if element.element_id == "Rleaf" and ".target[leaf].elements" in path_prefix:
            counts["leaf_calls"] += 1
        original_validate(**kwargs)

    monkeypatch.setattr(
        parser_design_bundle_module,
        "_validate_hierarchy_element_values",
        _spy_validate_hierarchy_element_values,
    )

    document = parse_design_bundle_document(path)

    assert tuple(subcircuit.subcircuit_id for subcircuit in document.subcircuits) == ("leaf", "mid")
    assert counts["leaf_calls"] == 1


@pytest.mark.parametrize(
    ("node_formals", "normalized_id", "raw_ids"),
    (
        (["p", "p"], "P", ["p", "p"]),
        (["p", "P"], "P", ["P", "p"]),
    ),
)
def test_duplicate_macro_formals_fail_with_illegal_declaration_diagnostic(
    tmp_path: Path,
    node_formals: list[str],
    normalized_id: str,
    raw_ids: list[str],
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "dup-macro",
            "kind": "R",
            "node_formals": node_formals,
            "params": {"resistance_ohm": 50.0},
        }
    ]
    design["subcircuits"] = []
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name=f"duplicate_macro_formals_{normalized_id}.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL"
    assert diagnostic.element_id == "dup-macro"
    assert diagnostic.witness == {
        "declaration_kind": "macro",
        "duplicate_count": 2,
        "field_name": "node_formals",
        "normalized_id": normalized_id,
        "raw_ids": raw_ids,
        "scope_id": "dup-macro",
        "scope_type": "macro",
    }


@pytest.mark.parametrize(
    ("ports", "normalized_id", "raw_ids"),
    (
        (["in", "in"], "IN", ["in", "in"]),
        (["in", "IN"], "IN", ["IN", "in"]),
    ),
)
def test_duplicate_subcircuit_ports_fail_with_illegal_declaration_diagnostic(
    tmp_path: Path,
    ports: list[str],
    normalized_id: str,
    raw_ids: list[str],
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "dup-stage",
            "ports": ports,
            "elements": [],
            "instances": [],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name=f"duplicate_subcircuit_ports_{normalized_id}.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL"
    assert diagnostic.element_id == "dup-stage"
    assert diagnostic.witness == {
        "declaration_kind": "subcircuit",
        "duplicate_count": 2,
        "field_name": "ports",
        "normalized_id": normalized_id,
        "raw_ids": raw_ids,
        "scope_id": "dup-stage",
        "scope_type": "subcircuit",
    }


def test_conflicting_subcircuit_local_element_and_instance_ids_fail_deterministically(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
                    "of": "res-load",
                    "nodes": ["out", "in"],
                }
            ],
        }
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="conflicting_local_namespace.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL"
    assert diagnostic.element_id == "stage:dup"
    assert diagnostic.witness == {
        "declaration_kind": "subcircuit_local_namespace",
        "declarations": [
            {"declaration_kind": "element", "raw_id": "dup"},
            {"declaration_kind": "instance", "raw_id": "DUP"},
        ],
        "normalized_id": "DUP",
        "scope_id": "stage",
        "scope_type": "subcircuit",
    }


def test_duplicate_hierarchy_definitions_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="duplicate_defs.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION"
    assert diagnostic.element_id == "hierarchy:BIAS_BLOCK"
    assert diagnostic.witness == {
        "definition_type": "macro",
        "normalized_id": "BIAS_BLOCK",
        "raw_ids": ["BIAS-BLOCK", "bias block"],
    }


def test_unicode_canonical_equivalent_hierarchy_definitions_fail_as_duplicates(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "Å",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        },
        {
            "id": "A\u030A",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 75.0},
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="unicode_duplicate_defs.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION"
    assert diagnostic.element_id == "hierarchy:Å"
    assert diagnostic.witness == {
        "definition_type": "macro",
        "normalized_id": "Å",
        "raw_ids": ["A\u030A", "Å"],
    }


def test_conflicting_hierarchy_definition_kinds_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT"
    assert diagnostic.element_id == "hierarchy:GAIN_CELL"
    assert diagnostic.witness == {
        "definitions": [
            {"definition_type": "macro", "raw_id": "gain cell"},
            {"definition_type": "subcircuit", "raw_id": "GAIN-CELL"},
        ],
        "normalized_id": "GAIN_CELL",
    }


def test_reference_to_duplicate_hierarchy_definition_is_not_misclassified_as_undefined(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="ambiguous_duplicate_reference.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
    )
    assert diagnostics[0].element_id == "hierarchy:BIAS_BLOCK"
    assert diagnostics[0].witness == {
        "definition_type": "macro",
        "normalized_id": "BIAS_BLOCK",
        "raw_ids": ["BIAS-BLOCK", "bias block"],
    }
    assert diagnostics[1].element_id == "xbias"
    assert diagnostics[1].witness == {
        "definition_type": "macro",
        "instance_id": "xbias",
        "instance_type": "macro",
        "normalized_id": "BIAS_BLOCK",
        "normalized_target_id": "BIAS_BLOCK",
        "raw_ids": ["BIAS-BLOCK", "bias block"],
        "scope_id": "design",
        "scope_type": "design",
        "target_id": "bias block",
    }


def test_reference_to_duplicate_wrong_kind_hierarchy_definition_is_not_misclassified_as_undefined(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = []
    design["subcircuits"] = [
        {
            "id": "foo",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [],
        },
        {
            "id": "FOO",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [],
        },
        {
            "id": "top",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xbad",
                    "instance_type": "macro",
                    "of": "foo",
                    "nodes": ["out", "0"],
                }
            ],
        },
    ]
    design["instances"] = []
    path = _write_bundle(tmp_path, payload, name="ambiguous_wrong_kind_duplicate_reference.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
    )
    assert diagnostics[0].element_id == "hierarchy:FOO"
    assert diagnostics[0].witness == {
        "definition_type": "subcircuit",
        "normalized_id": "FOO",
        "raw_ids": ["FOO", "foo"],
    }
    assert diagnostics[1].element_id == "top:Xbad"
    assert diagnostics[1].witness == {
        "definition_type": "subcircuit",
        "instance_id": "Xbad",
        "instance_type": "macro",
        "normalized_id": "FOO",
        "normalized_target_id": "FOO",
        "raw_ids": ["FOO", "foo"],
        "requested_instance_type": "macro",
        "scope_id": "top",
        "scope_type": "subcircuit",
        "target_id": "foo",
    }


def test_reference_to_conflicting_hierarchy_definition_is_not_misclassified_as_undefined(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="ambiguous_conflict_reference.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
    )
    assert diagnostics[0].element_id == "hierarchy:GAIN_CELL"
    assert diagnostics[0].witness == {
        "definitions": [
            {"definition_type": "macro", "raw_id": "gain cell"},
            {"definition_type": "subcircuit", "raw_id": "GAIN-CELL"},
        ],
        "normalized_id": "GAIN_CELL",
    }
    assert diagnostics[1].element_id == "xgain"
    assert diagnostics[1].witness == {
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
    }


def test_mixed_duplicate_and_conflicting_hierarchy_definitions_emit_both_failure_classes(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="mixed_duplicate_conflict_reference.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
    )
    assert diagnostics[0].element_id == "hierarchy:GAIN_CELL"
    assert diagnostics[0].witness == {
        "definitions": [
            {"definition_type": "macro", "raw_id": "GAIN-CELL"},
            {"definition_type": "macro", "raw_id": "gain cell"},
            {"definition_type": "subcircuit", "raw_id": "GAIN_CELL"},
        ],
        "normalized_id": "GAIN_CELL",
    }
    assert diagnostics[1].element_id == "xgain"
    assert diagnostics[1].witness == {
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
    }
    assert diagnostics[2].element_id == "hierarchy:GAIN_CELL"
    assert diagnostics[2].witness == {
        "definition_type": "macro",
        "normalized_id": "GAIN_CELL",
        "raw_ids": ["GAIN-CELL", "gain cell"],
    }
    assert diagnostics[3].element_id == "xgain"
    assert diagnostics[3].witness == {
        "definition_type": "macro",
        "instance_id": "xgain",
        "instance_type": "macro",
        "normalized_id": "GAIN_CELL",
        "normalized_target_id": "GAIN_CELL",
        "raw_ids": ["GAIN-CELL", "gain cell"],
        "scope_id": "design",
        "scope_type": "design",
        "target_id": "gain cell",
    }


def test_duplicate_top_level_hierarchy_instance_ids_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "Xstage",
            "instance_type": "subcircuit",
            "of": "gain-stage",
            "nodes": ["n1", "n2", "0"],
        },
        {
            "id": "Xstage",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="duplicate_top_level_instance_ids.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID"
    assert diagnostic.element_id == "Xstage"
    assert diagnostic.witness == {
        "duplicate_count": 2,
        "normalized_id": "XSTAGE",
        "raw_ids": ["Xstage", "Xstage"],
        "scope_id": "design",
        "scope_type": "design",
    }


def test_normalized_top_level_hierarchy_instance_ids_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["instances"] = [
        {
            "id": "X-load",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        },
        {
            "id": "x load",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="normalized_duplicate_top_level_instance_ids.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID"
    assert diagnostic.element_id == "X-load"
    assert diagnostic.witness == {
        "duplicate_count": 2,
        "normalized_id": "X_LOAD",
        "raw_ids": ["X-load", "x load"],
        "scope_id": "design",
        "scope_type": "design",
    }


def test_duplicate_subcircuit_local_element_ids_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
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
    path = _write_bundle(tmp_path, payload, name="duplicate_local_element_ids.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID"
    assert diagnostic.element_id == "gain-stage:Rdup"
    assert diagnostic.witness == {
        "declaration_kind": "element",
        "duplicate_count": 2,
        "normalized_id": "RDUP",
        "raw_ids": ["Rdup", "Rdup"],
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
    }


def test_normalized_subcircuit_local_element_ids_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
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
    path = _write_bundle(tmp_path, payload, name="normalized_duplicate_local_element_ids.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID"
    assert diagnostic.element_id == "gain-stage:R load"
    assert diagnostic.witness == {
        "declaration_kind": "element",
        "duplicate_count": 2,
        "normalized_id": "R_LOAD",
        "raw_ids": ["R load", "R-load"],
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
    }


def test_duplicate_subcircuit_instance_ids_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xdup",
                    "instance_type": "macro",
                    "of": "res-load",
                    "nodes": ["mid", "0"],
                },
                {
                    "id": "Xdup",
                    "instance_type": "macro",
                    "of": "res-load",
                    "nodes": ["out", "0"],
                },
            ],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="duplicate_subcircuit_instance_ids.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID"
    assert diagnostic.element_id == "gain-stage:Xdup"
    assert diagnostic.witness == {
        "duplicate_count": 2,
        "normalized_id": "XDUP",
        "raw_ids": ["Xdup", "Xdup"],
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
    }


def test_macro_templates_allow_required_values_from_instantiation(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
            "id": "gain-stage",
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
    path = _write_bundle(tmp_path, payload, name="macro_template_instance_override.json")

    document = parse_design_bundle_document(path)

    assert document.macros[0].params == {}
    assert dict(document.subcircuits[0].instances[0].params) == {"resistance_ohm": 125.0}


def test_subcircuit_reference_diagnostics_qualify_scope_in_element_id(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "A",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xbad",
                    "instance_type": "subcircuit",
                    "of": "missing",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
        {
            "id": "B",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xbad",
                    "instance_type": "subcircuit",
                    "of": "missing",
                    "nodes": ["in", "out", "0"],
                }
            ],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="scoped_reference_context.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics

    assert tuple(diagnostic.element_id for diagnostic in diagnostics) == ("A:Xbad", "B:Xbad")
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
    )


def test_undefined_hierarchy_references_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xmissing",
                    "instance_type": "subcircuit",
                    "of": "missing child",
                    "nodes": ["in", "out", "0"],
                }
            ],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="undefined_ref.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED"
    assert diagnostic.element_id == "gain-stage:Xmissing"
    assert diagnostic.witness == {
        "instance_id": "Xmissing",
        "instance_type": "subcircuit",
        "normalized_target_id": "MISSING_CHILD",
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
        "target_id": "missing child",
    }


def test_unicode_canonical_equivalent_hierarchy_reference_resolves(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "Å",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 50.0},
        }
    ]
    design["subcircuits"] = [
        {
            "id": "gain-stage",
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
    path = _write_bundle(tmp_path, payload, name="unicode_reference.json")

    document = parse_design_bundle_document(path)

    assert tuple(instance.target_id for instance in document.subcircuits[0].instances) == ("A\u030A",)


def test_hierarchy_reference_type_mismatch_fails_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="reference_type_mismatch.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_REFERENCE_TYPE_MISMATCH"
    assert diagnostic.element_id == "gain-stage:XwrongType"
    assert diagnostic.witness == {
        "instance_id": "XwrongType",
        "instance_type": "macro",
        "normalized_target_id": "CHILD_STAGE",
        "resolved_definition_type": "subcircuit",
        "resolved_raw_id": "child-stage",
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
        "target_id": "child-stage",
    }


def test_macro_instance_arity_mismatch_fails_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "gain-stage",
            "ports": ["in", "out", "0"],
            "elements": [],
            "instances": [
                {
                    "id": "Xarity",
                    "instance_type": "macro",
                    "of": "res-load",
                    "nodes": ["out"],
                }
            ],
        }
    ]
    path = _write_bundle(tmp_path, payload, name="macro_instance_arity_mismatch.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_INSTANCE_ARITY_INVALID"
    assert diagnostic.element_id == "gain-stage:Xarity"
    assert diagnostic.witness == {
        "actual_node_count": 1,
        "expected_node_count": 2,
        "instance_id": "Xarity",
        "instance_type": "macro",
        "normalized_target_id": "RES_LOAD",
        "resolved_definition_type": "macro",
        "resolved_raw_id": "res-load",
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
        "target_id": "res-load",
    }


def test_subcircuit_instance_arity_mismatch_fails_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
                    "id": "Xchild",
                    "instance_type": "subcircuit",
                    "of": "child-stage",
                    "nodes": ["in", "out"],
                }
            ],
        },
    ]
    path = _write_bundle(tmp_path, payload, name="subcircuit_instance_arity_mismatch.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_INSTANCE_ARITY_INVALID"
    assert diagnostic.element_id == "gain-stage:Xchild"
    assert diagnostic.witness == {
        "actual_node_count": 2,
        "expected_node_count": 3,
        "instance_id": "Xchild",
        "instance_type": "subcircuit",
        "normalized_target_id": "CHILD_STAGE",
        "resolved_definition_type": "subcircuit",
        "resolved_raw_id": "child-stage",
        "scope_id": "gain-stage",
        "scope_type": "subcircuit",
        "target_id": "child-stage",
    }


def test_illegal_recursive_subcircuit_declarations_fail_deterministically(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
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
    path = _write_bundle(tmp_path, payload, name="recursive_subckt.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL"
    assert diagnostic.element_id == "hierarchy:STAGE_A"
    assert diagnostic.witness == {"component": ["STAGE_A", "STAGE_B"]}


def test_self_recursive_subcircuit_declaration_reports_single_component(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = [
        {
            "id": "stage-a",
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
        }
    ]
    path = _write_bundle(tmp_path, payload, name="self_recursive_subckt.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL"
    assert diagnostic.element_id == "hierarchy:STAGE_A"
    assert diagnostic.witness == {"component": ["STAGE_A"]}


def test_branched_recursive_subcircuit_declarations_report_recursive_component_once(
    tmp_path: Path,
) -> None:
    payload = _hierarchy_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
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
    path = _write_bundle(tmp_path, payload, name="branched_recursive_subckt.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
    )
    assert tuple(diagnostic.witness for diagnostic in diagnostics) == (
        {"component": ["STAGE_A", "STAGE_B", "STAGE_C", "STAGE_D"]},
    )


def test_mixed_definition_and_recursion_failures_accumulate(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="mixed_definition_recursion.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    assert tuple(diagnostic.code for diagnostic in exc_info.value.diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
    )


def test_duplicate_subcircuit_definition_still_emits_recursion_diagnostic(tmp_path: Path) -> None:
    payload = _hierarchy_bundle()
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
    path = _write_bundle(tmp_path, payload, name="duplicate_subcircuit_recursion.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
    )
    assert diagnostics[0].element_id == "hierarchy:STAGE_A"
    assert diagnostics[1].element_id == "stage a:Xself"
    assert diagnostics[2].witness == {"component": ["STAGE_A"]}

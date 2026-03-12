from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfmna.parser.design_bundle import (
    DESIGN_BUNDLE_SCHEMA_ID,
    canonical_bundle_parse_product_json,
    hash_bundle_parse_product,
    parse_design_bundle_document,
)

pytestmark = pytest.mark.property


def _base_payload() -> dict[str, object]:
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
                            "of": "load cell",
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
                            "id": "Xbias",
                            "instance_type": "macro",
                            "of": "bias block",
                            "nodes": ["out", "0"],
                        },
                        {
                            "id": "XstageB",
                            "instance_type": "subcircuit",
                            "of": "stage-b",
                            "nodes": ["in", "out", "0"],
                        },
                    ],
                },
            ],
            "instances": [
                {
                    "id": "Xtop-load",
                    "instance_type": "macro",
                    "of": "load cell",
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


def _write_payload(tmp_path: Path, payload: dict[str, object], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@given(
    macro_order=st.permutations((0, 1)),
    subcircuit_order=st.permutations((0, 1)),
    stage_a_instance_order=st.permutations((0, 1)),
    top_level_instance_order=st.permutations((0, 1)),
)
def test_hierarchy_parse_product_is_permutation_invariant(
    macro_order: tuple[int, int],
    subcircuit_order: tuple[int, int],
    stage_a_instance_order: tuple[int, int],
    top_level_instance_order: tuple[int, int],
) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_payload = _base_payload()
        baseline_path = _write_payload(tmp_path, baseline_payload, name="baseline.json")
        baseline_document = parse_design_bundle_document(baseline_path)
        baseline_json = canonical_bundle_parse_product_json(baseline_document)
        baseline_hash = hash_bundle_parse_product(baseline_document)

        payload = _base_payload()
        design = payload["design"]
        assert isinstance(design, dict)
        macros = design["macros"]
        subcircuits = design["subcircuits"]
        top_level_instances = design["instances"]
        assert isinstance(macros, list)
        assert isinstance(subcircuits, list)
        assert isinstance(top_level_instances, list)
        design["macros"] = [macros[index] for index in macro_order]
        reordered_subcircuits = [subcircuits[index] for index in subcircuit_order]
        for subcircuit in reordered_subcircuits:
            assert isinstance(subcircuit, dict)
            if subcircuit["id"] == "stage-a":
                instances = subcircuit["instances"]
                assert isinstance(instances, list)
                subcircuit["instances"] = [instances[index] for index in stage_a_instance_order]
        design["subcircuits"] = reordered_subcircuits
        design["instances"] = [top_level_instances[index] for index in top_level_instance_order]

        candidate_path = _write_payload(tmp_path, payload, name="candidate.json")
        candidate_document = parse_design_bundle_document(candidate_path)

        assert canonical_bundle_parse_product_json(candidate_document) == baseline_json
        assert hash_bundle_parse_product(candidate_document) == baseline_hash


@given(
    stage_a_element_order=st.permutations((0, 1)),
)
def test_hierarchy_parse_product_preserves_subcircuit_local_element_order_semantics(
    stage_a_element_order: tuple[int, int],
) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_payload = _base_payload()
        baseline_path = _write_payload(tmp_path, baseline_payload, name="nested_order_baseline.json")
        baseline_document = parse_design_bundle_document(baseline_path)
        baseline_json = canonical_bundle_parse_product_json(baseline_document)
        baseline_hash = hash_bundle_parse_product(baseline_document)

        payload = _base_payload()
        design = payload["design"]
        assert isinstance(design, dict)
        subcircuits = design["subcircuits"]
        assert isinstance(subcircuits, list)
        stage_a = subcircuits[1]
        assert isinstance(stage_a, dict)
        elements = stage_a["elements"]
        assert isinstance(elements, list)
        stage_a["elements"] = [elements[index] for index in stage_a_element_order]

        candidate_path = _write_payload(tmp_path, payload, name="nested_order_candidate.json")
        candidate_document = parse_design_bundle_document(candidate_path)

        if tuple(stage_a_element_order) == (0, 1):
            assert canonical_bundle_parse_product_json(candidate_document) == baseline_json
            assert hash_bundle_parse_product(candidate_document) == baseline_hash
        else:
            assert canonical_bundle_parse_product_json(candidate_document) != baseline_json
            assert hash_bundle_parse_product(candidate_document) != baseline_hash


@given(
    macro_order=st.permutations((0, 1)),
    subcircuit_order=st.permutations((0, 1)),
    stage_a_instance_order=st.permutations((0, 1)),
    top_level_instance_order=st.permutations((0, 1)),
)
def test_hierarchy_parse_product_preserves_flat_design_order_semantics(
    macro_order: tuple[int, int],
    subcircuit_order: tuple[int, int],
    stage_a_instance_order: tuple[int, int],
    top_level_instance_order: tuple[int, int],
) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_payload = _base_payload()
        baseline_design = baseline_payload["design"]
        assert isinstance(baseline_design, dict)
        baseline_design["nodes"] = ["n2", "n1"]
        baseline_design["ports"] = [
            {"id": "P2", "p_plus": "n2", "p_minus": "0", "z0_ohm": 50.0},
            {"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 50.0},
        ]
        baseline_path = _write_payload(tmp_path, baseline_payload, name="flat_order_baseline.json")
        baseline_document = parse_design_bundle_document(baseline_path)
        baseline_json = canonical_bundle_parse_product_json(baseline_document)
        baseline_hash = hash_bundle_parse_product(baseline_document)

        payload = _base_payload()
        design = payload["design"]
        assert isinstance(design, dict)
        design["nodes"] = ["n1", "n2"]
        design["ports"] = [
            {"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 50.0},
            {"id": "P2", "p_plus": "n2", "p_minus": "0", "z0_ohm": 50.0},
        ]
        macros = design["macros"]
        subcircuits = design["subcircuits"]
        top_level_instances = design["instances"]
        assert isinstance(macros, list)
        assert isinstance(subcircuits, list)
        assert isinstance(top_level_instances, list)
        design["macros"] = [macros[index] for index in macro_order]
        reordered_subcircuits = [subcircuits[index] for index in subcircuit_order]
        for subcircuit in reordered_subcircuits:
            assert isinstance(subcircuit, dict)
            if subcircuit["id"] == "stage-a":
                instances = subcircuit["instances"]
                assert isinstance(instances, list)
                subcircuit["instances"] = [instances[index] for index in stage_a_instance_order]
        design["subcircuits"] = reordered_subcircuits
        design["instances"] = [top_level_instances[index] for index in top_level_instance_order]

        candidate_path = _write_payload(tmp_path, payload, name="flat_order_candidate.json")
        candidate_document = parse_design_bundle_document(candidate_path)

        assert canonical_bundle_parse_product_json(candidate_document) != baseline_json
        assert hash_bundle_parse_product(candidate_document) != baseline_hash

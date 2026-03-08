from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.parser import DESIGN_BUNDLE_SCHEMA_ID, load_design_bundle_document

pytestmark = pytest.mark.conformance

runner = CliRunner()
_EXIT_FAILURE: Final[int] = 2
_TWO_POINTS: Final[int] = 2


def _base_bundle() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "elements": [
                {
                    "id": "R1",
                    "kind": "R",
                    "nodes": ["n1", "0"],
                    "params": {"resistance_ohm": 50.0},
                }
            ],
            "ports": [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 50.0}],
        },
        "analysis": {
            "type": "ac",
            "frequency_sweep": {
                "mode": "linear",
                "start": {"value": 1.0, "unit": "Hz"},
                "stop": {"value": 2.0, "unit": "Hz"},
                "points": 2,
            }
        },
    }


def _matched_two_port_bundle() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "elements": [
                {
                    "id": "R1",
                    "kind": "R",
                    "nodes": ["n1", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
                {
                    "id": "R2",
                    "kind": "R",
                    "nodes": ["n2", "0"],
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
            }
        },
    }


def _sentinel_bundle() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "elements": [
                {
                    "id": "C1",
                    "kind": "C",
                    "nodes": ["n1", "0"],
                    "params": {"capacitance_f": 1.0e-6},
                },
                {
                    "id": "I1",
                    "kind": "I",
                    "nodes": ["n1", "0"],
                    "params": {"current_a": 1.0},
                },
            ],
            "ports": [{"id": "P1", "p_plus": "n1", "p_minus": "0"}],
        },
        "analysis": {
            "type": "ac",
            "frequency_sweep": {
                "mode": "linear",
                "start": {"value": 0.0, "unit": "Hz"},
                "stop": {"value": 1.0, "unit": "Hz"},
                "points": 2,
            }
        },
    }


def _write_bundle(tmp_path: Path, payload: dict[str, object], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _solve_bundle(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...], object]:
    design_path = _write_bundle(tmp_path, payload, name=name)
    parsed = load_design_bundle_document(design_path)
    bundle = cli_main._load_design_bundle(design_path.as_posix())
    result = cli_main._execute_sweep(bundle.frequencies_hz, bundle.sweep_layout, bundle.assemble_point)
    node_ids = tuple(node.node_id for node in parsed.ir.nodes if not node.is_reference)
    aux_ids = tuple(aux.aux_id for aux in parsed.ir.aux_unknowns)
    return node_ids, aux_ids, result


def test_design_bundle_schema_artifact_location_and_version_are_canonical() -> None:
    schema_path = Path(__file__).resolve().parents[2] / DESIGN_BUNDLE_SCHEMA_ID
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"] == DESIGN_BUNDLE_SCHEMA_ID
    assert schema["properties"]["schema"]["const"] == DESIGN_BUNDLE_SCHEMA_ID
    assert schema["properties"]["schema_version"]["const"] == 1


def test_design_bundle_schema_artifact_encodes_supported_kind_arity_and_required_params() -> None:
    schema_path = Path(__file__).resolve().parents[2] / DESIGN_BUNDLE_SCHEMA_ID
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    element_rules = schema["$defs"]["element"]["allOf"]
    kind_tokens = schema["$defs"]["accepted_element_kind_token"]["enum"]

    assert schema["$defs"]["element"]["properties"]["kind"]["$ref"] == (
        "#/$defs/accepted_element_kind_token"
    )
    assert "R" in kind_tokens
    assert "Y1P" in kind_tokens
    assert "FD_LINEAR" in kind_tokens
    assert "UNSUPPORTED_KIND" not in kind_tokens

    def _then_for_kind(kind_token: str) -> dict[str, object]:
        for rule in element_rules:
            kind_selector = rule["if"]["properties"]["kind"]
            if kind_selector.get("const") == kind_token:
                return rule["then"]
            if kind_token in kind_selector.get("enum", []):
                return rule["then"]
        raise AssertionError(f"missing schema rule for kind token {kind_token}")

    assert _then_for_kind("R")["properties"]["nodes"]["$ref"] == "#/$defs/two_node_string_array"
    assert _then_for_kind("R")["properties"]["params"]["allOf"][1]["required"] == ["resistance_ohm"]
    assert _then_for_kind("C")["properties"]["params"]["allOf"][1]["required"] == ["capacitance_f"]
    assert _then_for_kind("G")["properties"]["params"]["allOf"][1]["required"] == ["conductance_s"]
    assert _then_for_kind("L")["properties"]["params"]["allOf"][1]["required"] == ["inductance_h"]
    assert _then_for_kind("I")["properties"]["params"]["allOf"][1]["required"] == ["current_a"]
    assert _then_for_kind("V")["properties"]["params"]["allOf"][1]["required"] == ["voltage_v"]
    assert _then_for_kind("VCCS")["properties"]["nodes"]["$ref"] == "#/$defs/four_node_string_array"
    assert _then_for_kind("VCCS")["properties"]["params"]["allOf"][1]["required"] == [
        "transconductance_s"
    ]
    assert _then_for_kind("VCVS")["properties"]["nodes"]["$ref"] == "#/$defs/four_node_string_array"
    assert _then_for_kind("VCVS")["properties"]["params"]["allOf"][1]["required"] == ["gain_mu"]
    assert _then_for_kind("RESISTOR")["properties"]["params"]["allOf"][1]["required"] == [
        "resistance_ohm"
    ]
    assert _then_for_kind("CURRENT_SOURCE")["properties"]["params"]["allOf"][1]["required"] == [
        "current_a"
    ]
    assert _then_for_kind("E")["properties"]["params"]["allOf"][1]["required"] == ["gain_mu"]
    assert schema["$defs"]["nonempty_string"]["pattern"] == "\\S"
    assert schema["$defs"]["scalar_token"]["anyOf"][1]["$ref"] == "#/$defs/nonempty_string"
    assert schema["$defs"]["design"]["properties"]["reference_node"]["$ref"] == (
        "#/$defs/nonempty_string"
    )


def test_loader_backed_check_and_run_paths_accept_supported_bundle(tmp_path: Path) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])
    run_result = runner.invoke(cli_main.app, ["run", design_path.as_posix(), "--analysis", "ac"])

    assert check_result.exit_code == 0
    check_payload = json.loads(check_result.stdout)
    assert check_payload["status"] == "pass"
    assert check_payload["diagnostics"] == []

    assert run_result.exit_code == 0
    assert "POINT index=0 freq_hz=1 status=pass" in run_result.stdout
    assert "POINT index=1 freq_hz=2 status=pass" in run_result.stdout


def test_loader_rejects_non_schema_kind_tokens_even_if_runtime_normalization_would_accept_them(
    tmp_path: Path,
) -> None:
    payload = _base_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    elements = design["elements"]
    assert isinstance(elements, list)
    element = elements[0]
    assert isinstance(element, dict)
    element["kind"] = "resistor"
    design_path = _write_bundle(tmp_path, payload, name="non_schema_kind.json")

    result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert result.exit_code == _EXIT_FAILURE
    payload_out = json.loads(result.stdout)
    diagnostic = payload_out["diagnostics"][0]
    assert diagnostic["code"] == "E_CLI_DESIGN_SCHEMA_INVALID"
    assert diagnostic["witness"] == {"path": "$.design.elements[0].kind"}


def test_loader_rejects_supported_kind_shape_that_violates_per_kind_schema_rules(
    tmp_path: Path,
) -> None:
    payload = _base_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["elements"] = [
        {
            "id": "R1",
            "kind": "R",
            "nodes": ["n1", "n2", "0"],
            "params": {},
        }
    ]
    design_path = _write_bundle(tmp_path, payload, name="invalid_supported_shape.json")

    result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert result.exit_code == _EXIT_FAILURE
    payload_out = json.loads(result.stdout)
    diagnostic = payload_out["diagnostics"][0]
    assert diagnostic["code"] == "E_CLI_DESIGN_SCHEMA_INVALID"
    assert diagnostic["witness"] == {"path": "$.design.elements[0].nodes"}


def test_loader_backed_supported_element_kinds_execute_with_expected_orientation(
    tmp_path: Path,
) -> None:
    r_i_nodes, r_i_aux, r_i_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "R1",
                        "kind": "R",
                        "nodes": ["n1", "0"],
                        "params": {"resistance_ohm": 2.0},
                    },
                    {
                        "id": "I1",
                        "kind": "I",
                        "nodes": ["n1", "0"],
                        "params": {"current_a": 1.0},
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
        },
        name="r_i_supported.json",
    )
    assert list(r_i_result.status.tolist()) == ["pass"]
    assert r_i_nodes == ("n1",)
    assert r_i_aux == ()
    np.testing.assert_allclose(r_i_result.V_nodes[0, 0], -2.0 + 0.0j)

    c_nodes, _, c_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "C1",
                        "kind": "C",
                        "nodes": ["n1", "0"],
                        "params": {"capacitance_f": 1.0e-6},
                    },
                    {
                        "id": "I1",
                        "kind": "I",
                        "nodes": ["0", "n1"],
                        "params": {"current_a": 1.0},
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
        },
        name="c_supported.json",
    )
    assert c_nodes == ("n1",)
    assert list(c_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(c_result.V_nodes[0, 0], 0.0 - 1j / (2.0 * np.pi * 1.0e-6))

    g_nodes, _, g_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "G1",
                        "kind": "G",
                        "nodes": ["n1", "0"],
                        "params": {"conductance_s": 0.5},
                    },
                    {
                        "id": "I1",
                        "kind": "I",
                        "nodes": ["0", "n1"],
                        "params": {"current_a": 1.0},
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
        },
        name="g_supported.json",
    )
    assert g_nodes == ("n1",)
    assert list(g_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(g_result.V_nodes[0, 0], 2.0 + 0.0j)

    l_nodes, l_aux, l_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "L1",
                        "kind": "L",
                        "nodes": ["n1", "0"],
                        "params": {"inductance_h": 1.0e-3},
                    },
                    {
                        "id": "I1",
                        "kind": "I",
                        "nodes": ["0", "n1"],
                        "params": {"current_a": 1.0},
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
        },
        name="l_supported.json",
    )
    assert l_nodes == ("n1",)
    assert l_aux == ("L1:i",)
    assert list(l_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(l_result.V_nodes[0, 0], 0.0 + 1j * 2.0 * np.pi * 1.0e-3)
    np.testing.assert_allclose(l_result.I_aux[0, 0], 1.0 + 0.0j)

    v_nodes, v_aux, v_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "R1",
                        "kind": "R",
                        "nodes": ["n1", "0"],
                        "params": {"resistance_ohm": 1.0},
                    },
                    {
                        "id": "V1",
                        "kind": "V",
                        "nodes": ["n1", "0"],
                        "params": {"voltage_v": 3.0},
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
        },
        name="v_supported.json",
    )
    assert v_nodes == ("n1",)
    assert v_aux == ("V1:i",)
    assert list(v_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(v_result.V_nodes[0, 0], 3.0 + 0.0j)
    np.testing.assert_allclose(v_result.I_aux[0, 0], -3.0 + 0.0j)

    vccs_nodes, vccs_aux, vccs_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "VCTRL",
                        "kind": "V",
                        "nodes": ["nc", "0"],
                        "params": {"voltage_v": 2.0},
                    },
                    {
                        "id": "RL",
                        "kind": "R",
                        "nodes": ["nout", "0"],
                        "params": {"resistance_ohm": 4.0},
                    },
                    {
                        "id": "G1",
                        "kind": "VCCS",
                        "nodes": ["nout", "0", "nc", "0"],
                        "params": {"transconductance_s": 0.5},
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
        },
        name="vccs_supported.json",
    )
    assert vccs_nodes == ("nc", "nout")
    assert vccs_aux == ("VCTRL:i",)
    assert list(vccs_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(vccs_result.V_nodes[0], np.asarray([2.0 + 0.0j, -4.0 + 0.0j]))
    np.testing.assert_allclose(vccs_result.I_aux[0, 0], 0.0 + 0.0j)

    vcvs_nodes, vcvs_aux, vcvs_result = _solve_bundle(
        tmp_path,
        {
            "schema": DESIGN_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "design": {
                "reference_node": "0",
                "elements": [
                    {
                        "id": "VCTRL",
                        "kind": "V",
                        "nodes": ["nc", "0"],
                        "params": {"voltage_v": 2.0},
                    },
                    {
                        "id": "RL",
                        "kind": "R",
                        "nodes": ["nout", "0"],
                        "params": {"resistance_ohm": 4.0},
                    },
                    {
                        "id": "E1",
                        "kind": "VCVS",
                        "nodes": ["nout", "0", "nc", "0"],
                        "params": {"gain_mu": 1.5},
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
        },
        name="vcvs_supported.json",
    )
    assert vcvs_nodes == ("nc", "nout")
    assert vcvs_aux == ("E1:i", "VCTRL:i")
    assert list(vcvs_result.status.tolist()) == ["pass"]
    np.testing.assert_allclose(vcvs_result.V_nodes[0], np.asarray([2.0 + 0.0j, 3.0 + 0.0j]))
    np.testing.assert_allclose(vcvs_result.I_aux[0], np.asarray([-0.75 + 0.0j, 0.0 + 0.0j]))


def test_loader_backed_rf_path_preserves_port_order_and_wave_conventions(tmp_path: Path) -> None:
    design_path = _write_bundle(tmp_path, _matched_two_port_bundle(), name="matched_two_port.json")

    result = runner.invoke(
        cli_main.app,
        [
            "run",
            design_path.as_posix(),
            "--analysis",
            "ac",
            "--rf",
            "y",
            "--rf",
            "s",
        ],
    )

    assert result.exit_code == 0
    rf_lines = [line for line in result.stdout.splitlines() if line.startswith("RF ")]
    assert any("metric=y" in line and "row_port=P1" in line and "col_port=P1" in line for line in rf_lines)
    assert any("metric=y" in line and "row_port=P2" in line and "col_port=P2" in line for line in rf_lines)
    assert any(
        "metric=y" in line and "row_port=P1" in line and "col_port=P1" in line and "value_re=0.02" in line
        for line in rf_lines
    )
    assert any(
        "metric=s" in line and "row_port=P1" in line and "col_port=P1" in line and "value_re=0" in line and "value_im=0" in line
        for line in rf_lines
    )
    assert any(
        "metric=s" in line and "row_port=P2" in line and "col_port=P2" in line and "value_re=0" in line and "value_im=0" in line
        for line in rf_lines
    )
    # The first matrix line should already reflect canonical port sorting, not declaration order.
    assert "row_port=P1" in rf_lines[0]
    assert "col_port=P1" in rf_lines[0]


def test_loader_frequency_grammar_matches_frozen_non_regression_rules(tmp_path: Path) -> None:
    payload = _base_bundle()
    payload["analysis"] = {
        "type": "ac",
        "frequency_sweep": {
            "mode": "log",
            "start": {"value": 1.0, "unit": "MHz"},
            "stop": {"value": 1000.0, "unit": "MHz"},
            "points": 5,
        }
    }
    design_path = _write_bundle(tmp_path, payload, name="log_grid.json")

    parsed = load_design_bundle_document(design_path)

    expected_log10 = np.asarray([6.0, 6.75, 7.5, 8.25, 9.0], dtype=np.float64)
    np.testing.assert_allclose(np.log10(parsed.frequencies_hz), expected_log10)
    assert parsed.frequencies_hz[0] == pytest.approx(1.0e6)
    assert parsed.frequencies_hz[-1] == pytest.approx(1.0e9)


def test_loader_backed_run_preserves_fail_sentinel_and_no_point_omission_in_conformance(
    tmp_path: Path,
) -> None:
    design_path = _write_bundle(tmp_path, _sentinel_bundle(), name="sentinel.json")
    loaded_bundle = cli_main._load_design_bundle(design_path.as_posix())

    def _assemble_point(point_index: int, frequency_hz: float):
        if point_index == 0:
            raise RuntimeError("planned loader-backed assembly failure")
        return loaded_bundle.assemble_point(point_index, frequency_hz)

    result = cli_main._execute_sweep(
        loaded_bundle.frequencies_hz,
        loaded_bundle.sweep_layout,
        _assemble_point,
    )

    assert result.n_points == _TWO_POINTS
    assert list(result.status.tolist()) == ["fail", "pass"]
    assert np.isnan(result.res_l2[0])
    assert np.isnan(result.res_linf[0])
    assert np.isnan(result.res_rel[0])
    assert np.isnan(result.cond_ind[0])
    assert np.isnan(result.V_nodes[0, 0].real)
    assert np.isnan(result.V_nodes[0, 0].imag)
    assert result.diagnostics_by_point[0]
    assert result.diagnostics_by_point[0][0].code in {"E_NUM_SOLVE_FAILED", "E_NUM_SINGULAR_MATRIX"}


def test_loader_failure_diagnostics_are_taxonomy_complete_in_check_json(tmp_path: Path) -> None:
    payload = _base_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["elements"] = [
        {
            "id": "Y1",
            "kind": "Y1P",
            "nodes": ["n1", "0"],
            "params": {"y11_s": 0.02},
        }
    ]
    design_path = _write_bundle(tmp_path, payload, name="excluded.json")

    result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert result.exit_code == _EXIT_FAILURE
    payload_out = json.loads(result.stdout)
    diagnostic = payload_out["diagnostics"][0]
    assert diagnostic["code"] == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"
    assert diagnostic["severity"] == "error"
    assert diagnostic["message"]
    assert diagnostic["suggested_action"]
    assert diagnostic["solver_stage"] == "parse"
    assert diagnostic["element_id"] == "Y1"
    assert diagnostic["witness"] == {
        "capability_id": "y_block_elements",
        "element_id": "Y1",
        "kind": "Y1P",
        "normalized_kind": "Y1P",
    }

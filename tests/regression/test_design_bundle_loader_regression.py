from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.parser.design_bundle import (
    DESIGN_BUNDLE_SCHEMA_ID,
    DesignBundleLoadError,
    parse_design_bundle_document,
)
from rfmna.viz_io import build_manifest

pytestmark = pytest.mark.regression

runner = CliRunner()
_EXIT_FAILURE: Final[int] = 2
_TWO_POINTS: Final[int] = 2
_DEEP_HIERARCHY_DEPTH: Final[int] = 1300


def _supported_bundle() -> dict[str, object]:
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


def _parameterized_bundle(*, resistance_ohm: float) -> dict[str, object]:
    payload = _supported_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["parameters"] = {"r_load": resistance_ohm}
    elements = design["elements"]
    assert isinstance(elements, list)
    element = elements[0]
    assert isinstance(element, dict)
    params = element["params"]
    assert isinstance(params, dict)
    params["resistance_ohm"] = "r_load"
    return payload


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


def _excluded_bundle() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "elements": [
                {
                    "id": "FD1",
                    "kind": "FD_LINEAR",
                    "nodes": ["n1", "0"],
                    "params": {"numerator": 1.0},
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


def _invalid_unused_hierarchy_bundle() -> dict[str, object]:
    payload = _supported_bundle()
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
    return payload


def _invalid_macro_default_key_bundle() -> dict[str, object]:
    payload = _supported_bundle()
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
    return payload


def _invalid_unused_macro_default_value_bundle() -> dict[str, object]:
    payload = _supported_bundle()
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
    return payload


def _top_level_instance_bundle() -> dict[str, object]:
    payload = _supported_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["macros"] = [
        {
            "id": "res-load",
            "kind": "R",
            "node_formals": ["p", "n"],
            "params": {"resistance_ohm": 75.0},
        }
    ]
    design["instances"] = [
        {
            "id": "Xload",
            "instance_type": "macro",
            "of": "res-load",
            "nodes": ["n1", "0"],
        }
    ]
    return payload


def _invalid_port_z0_bundle() -> dict[str, object]:
    payload = _supported_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": -5.0}]
    return payload


def _deep_hierarchy_bundle(*, depth: int, cyclic: bool) -> dict[str, object]:
    subcircuits: list[dict[str, object]] = []
    for index in range(depth):
        instances: list[dict[str, object]] = []
        if index + 1 < depth:
            instances.append(
                {
                    "id": f"X{index + 1}",
                    "instance_type": "subcircuit",
                    "of": f"stage-{index + 1}",
                    "nodes": ["in", "out", "0"],
                }
            )
        elif cyclic:
            instances.append(
                {
                    "id": "X0",
                    "instance_type": "subcircuit",
                    "of": "stage-0",
                    "nodes": ["in", "out", "0"],
                }
            )
        subcircuits.append(
            {
                "id": f"stage-{index}",
                "ports": ["in", "out", "0"],
                "elements": [],
                "instances": instances,
            }
        )
    payload = _supported_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["subcircuits"] = subcircuits
    return payload


def _write_bundle(tmp_path: Path, payload: dict[str, object], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_loader_backed_check_exit_mapping_remains_nonzero_for_excluded_inputs(tmp_path: Path) -> None:
    supported_path = _write_bundle(tmp_path, _supported_bundle(), name="supported.json")
    excluded_path = _write_bundle(tmp_path, _excluded_bundle(), name="excluded.json")

    supported = runner.invoke(cli_main.app, ["check", supported_path.as_posix(), "--format", "json"])
    excluded = runner.invoke(cli_main.app, ["check", excluded_path.as_posix(), "--format", "json"])

    assert supported.exit_code == 0
    assert json.loads(supported.stdout)["status"] == "pass"

    assert excluded.exit_code == _EXIT_FAILURE
    excluded_payload = json.loads(excluded.stdout)
    assert excluded_payload["status"] == "fail"
    assert excluded_payload["exit_code"] == _EXIT_FAILURE
    assert excluded_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"


def test_loader_backed_run_preserves_fail_sentinel_and_no_point_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _sentinel_bundle(), name="sentinel.json")

    loaded_bundle = cli_main._load_design_bundle(design_path.as_posix())

    def _assemble_point(point_index: int, frequency_hz: float):
        if point_index == 0:
            raise RuntimeError("planned loader-backed assembly failure")
        return loaded_bundle.assemble_point(point_index, frequency_hz)

    bundle = cli_main.CliDesignBundle(
        preflight_input=loaded_bundle.preflight_input,
        frequencies_hz=loaded_bundle.frequencies_hz,
        sweep_layout=loaded_bundle.sweep_layout,
        assemble_point=_assemble_point,
        rf_ports=loaded_bundle.rf_ports,
        rf_z0_ohm=loaded_bundle.rf_z0_ohm,
    )
    result = cli_main._execute_sweep(bundle.frequencies_hz, bundle.sweep_layout, bundle.assemble_point)

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

    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    cli_result = runner.invoke(cli_main.app, ["run", design_path.as_posix(), "--analysis", "ac"])
    assert cli_result.exit_code == _EXIT_FAILURE
    assert "POINT index=0 freq_hz=0 status=fail" in cli_result.stdout
    assert "POINT index=1 freq_hz=1 status=pass" in cli_result.stdout


def test_loader_backed_check_preserves_specific_port_z0_failure_code(tmp_path: Path) -> None:
    design_path = _write_bundle(tmp_path, _invalid_port_z0_bundle(), name="invalid_port_z0.json")

    result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert result.exit_code == _EXIT_FAILURE
    payload = json.loads(result.stdout)
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "E_MODEL_PORT_Z0_NONPOSITIVE"
    assert diagnostic["solver_stage"] == "assemble"
    assert diagnostic["port_context"] == {"port_id": "P1"}
    assert diagnostic["witness"] == {
        "path": "design.ports[P1].z0_ohm",
        "port_id": "P1",
    }


def test_loader_backed_commands_reject_invalid_unused_hierarchy_declarations(
    tmp_path: Path,
) -> None:
    design_path = _write_bundle(
        tmp_path,
        _invalid_unused_hierarchy_bundle(),
        name="invalid_unused_hierarchy.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])
    run_result = runner.invoke(cli_main.app, ["run", design_path.as_posix(), "--analysis", "ac"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["status"] == "fail"
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_VALUE_INVALID"
    assert check_payload["diagnostics"][0]["witness"] == {
        "input_text": "missing_param + 1",
        "path": "design.subcircuits[gain-stage].elements[Rbias].params.resistance_ohm",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }

    assert run_result.exit_code == _EXIT_FAILURE
    assert "E_CLI_DESIGN_VALUE_INVALID" in run_result.stdout


def test_loader_backed_commands_reject_invalid_macro_default_keys(tmp_path: Path) -> None:
    design_path = _write_bundle(
        tmp_path,
        _invalid_macro_default_key_bundle(),
        name="invalid_macro_default_key.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_VALUE_INVALID"
    assert check_payload["diagnostics"][0]["witness"] == {
        "allowed_keys": ["resistance_ohm"],
        "path": "design.macros[res-load].params",
        "unexpected_keys": ["bogus"],
    }


def test_loader_backed_commands_reject_invalid_unused_macro_default_values(tmp_path: Path) -> None:
    design_path = _write_bundle(
        tmp_path,
        _invalid_unused_macro_default_value_bundle(),
        name="invalid_unused_macro_default_value.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])
    run_result = runner.invoke(cli_main.app, ["run", design_path.as_posix(), "--analysis", "ac"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["status"] == "fail"
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_VALUE_INVALID"
    assert check_payload["diagnostics"][0]["witness"] == {
        "issue_code": "E_MODEL_R_NONPOSITIVE",
        "issue_context": {"element_id": "res-load", "resistance_ohm": 0.0},
        "path": "design.macros[res-load]",
    }

    assert run_result.exit_code == _EXIT_FAILURE
    assert "E_CLI_DESIGN_VALUE_INVALID" in run_result.stdout


def test_loader_backed_commands_report_exclusions_before_top_level_hierarchy_unsupported(
    tmp_path: Path,
) -> None:
    payload = _top_level_instance_bundle()
    analysis = payload["analysis"]
    assert isinstance(analysis, dict)
    analysis["parameter_sweeps"] = [{"parameter": "bias", "values": [1.0, 2.0]}]
    design_path = _write_bundle(
        tmp_path,
        payload,
        name="top_level_instance_with_exclusion.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"


def test_loader_backed_commands_report_exclusions_before_hierarchy_value_validation(
    tmp_path: Path,
) -> None:
    payload = _invalid_unused_macro_default_value_bundle()
    analysis = payload["analysis"]
    assert isinstance(analysis, dict)
    analysis["parameter_sweeps"] = [{"parameter": "bias", "values": [1.0, 2.0]}]
    design_path = _write_bundle(
        tmp_path,
        payload,
        name="excluded_before_hierarchy_value_validation.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"


def test_loader_backed_commands_report_invalid_top_level_instance_override_before_unsupported(
    tmp_path: Path,
) -> None:
    payload = _top_level_instance_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    instances = design["instances"]
    assert isinstance(instances, list)
    instance = instances[0]
    assert isinstance(instance, dict)
    instance["params"] = {"resistance_ohm": "missing_param + 1"}
    design_path = _write_bundle(
        tmp_path,
        payload,
        name="top_level_instance_invalid_override.json",
    )

    check_result = runner.invoke(cli_main.app, ["check", design_path.as_posix(), "--format", "json"])

    assert check_result.exit_code == _EXIT_FAILURE
    check_payload = json.loads(check_result.stdout)
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_VALUE_INVALID"
    assert check_payload["diagnostics"][0]["witness"] == {
        "input_text": "missing_param + 1",
        "path": "design.instances[Xload].params.resistance_ohm",
        "source_code": "E_PARSE_PARAM_UNDEFINED",
        "witness": ["missing_param"],
    }


def test_loader_backed_manifest_hashes_track_design_contents_and_resolved_params(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_parameterized_bundle(resistance_ohm=50.0), indent=2), encoding="utf-8")
    first_bundle = cli_main._load_design_bundle(design_path.as_posix())

    design_path.write_text(json.dumps(_parameterized_bundle(resistance_ohm=75.0), indent=2), encoding="utf-8")
    second_bundle = cli_main._load_design_bundle(design_path.as_posix())

    first_manifest = build_manifest(
        input_payload=first_bundle.manifest_input_payload,
        resolved_params_payload=first_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(first_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )
    second_manifest = build_manifest(
        input_payload=second_bundle.manifest_input_payload,
        resolved_params_payload=second_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(second_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )

    assert first_manifest.input_hash != second_manifest.input_hash
    assert first_manifest.resolved_params_hash != second_manifest.resolved_params_hash


def test_parse_surface_handles_deep_acyclic_hierarchy_without_python_recursion_crash(
    tmp_path: Path,
) -> None:
    path = _write_bundle(
        tmp_path,
        _deep_hierarchy_bundle(depth=_DEEP_HIERARCHY_DEPTH, cyclic=False),
        name="deep_acyclic_hierarchy.json",
    )

    document = parse_design_bundle_document(path)

    assert len(document.subcircuits) == _DEEP_HIERARCHY_DEPTH


def test_parse_surface_reports_deep_recursive_hierarchy_with_cataloged_diagnostic(
    tmp_path: Path,
) -> None:
    path = _write_bundle(
        tmp_path,
        _deep_hierarchy_bundle(depth=_DEEP_HIERARCHY_DEPTH, cyclic=True),
        name="deep_recursive_hierarchy.json",
    )

    with pytest.raises(DesignBundleLoadError) as exc_info:
        parse_design_bundle_document(path)

    diagnostics = exc_info.value.diagnostics
    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
    )
    assert diagnostics[0].element_id == "hierarchy:STAGE_0"
    assert diagnostics[0].witness == {
        "component": sorted(f"STAGE_{index}" for index in range(_DEEP_HIERARCHY_DEPTH))
    }


def test_parse_surface_reports_each_equivalent_invalid_instance_path(tmp_path: Path) -> None:
    payload = _supported_bundle()
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
    path = _write_bundle(
        tmp_path,
        payload,
        name="equivalent_invalid_instance_paths.json",
    )

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


def test_loader_backed_manifest_hashes_ignore_design_file_path_for_identical_contents(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "nested" / "right.json"
    right_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _parameterized_bundle(resistance_ohm=50.0)
    left_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    right_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    left_bundle = cli_main._load_design_bundle(left_path.as_posix())
    right_bundle = cli_main._load_design_bundle(right_path.as_posix())

    left_manifest = build_manifest(
        input_payload=left_bundle.manifest_input_payload,
        resolved_params_payload=left_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(left_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )
    right_manifest = build_manifest(
        input_payload=right_bundle.manifest_input_payload,
        resolved_params_payload=right_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(right_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )

    assert left_manifest.input_hash == right_manifest.input_hash
    assert left_manifest.resolved_params_hash == right_manifest.resolved_params_hash


def test_loader_backed_manifest_hashes_preserve_unused_hierarchy_declaration_order(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left_hierarchy.json"
    right_path = tmp_path / "right_hierarchy.json"
    baseline = _parameterized_bundle(resistance_ohm=50.0)
    left_design = baseline["design"]
    assert isinstance(left_design, dict)
    left_design["macros"] = [
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
    ]
    left_design["subcircuits"] = [
        {
            "id": "stage-a",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "Rmid",
                    "kind": "R",
                    "nodes": ["mid", "0"],
                    "params": {"resistance_ohm": 50.0},
                },
                {
                    "id": "Rout",
                    "kind": "R",
                    "nodes": ["out", "0"],
                    "params": {"resistance_ohm": 75.0},
                },
            ],
            "instances": [
                {
                    "id": "Xmid",
                    "instance_type": "macro",
                    "of": "bias block",
                    "nodes": ["mid", "0"],
                },
                {
                    "id": "Xout",
                    "instance_type": "macro",
                    "of": "load cell",
                    "nodes": ["out", "0"],
                },
            ],
        },
        {
            "id": "stage-b",
            "ports": ["in", "out", "0"],
            "elements": [
                {
                    "id": "Rkeep",
                    "kind": "R",
                    "nodes": ["in", "0"],
                    "params": {"resistance_ohm": 60.0},
                }
            ],
            "instances": [
                {
                    "id": "Xkeep",
                    "instance_type": "macro",
                    "of": "bias block",
                    "nodes": ["out", "0"],
                }
            ],
        }
    ]
    reordered = json.loads(json.dumps(baseline))
    right_design = reordered["design"]
    assert isinstance(right_design, dict)
    right_subcircuits = right_design["subcircuits"]
    assert isinstance(right_subcircuits, list)
    right_stage = right_subcircuits[0]
    assert isinstance(right_stage, dict)
    right_stage["elements"] = list(reversed(right_stage["elements"]))
    right_stage["instances"] = list(reversed(right_stage["instances"]))
    right_design["subcircuits"] = list(reversed(right_subcircuits))
    right_design["macros"] = list(reversed(right_design["macros"]))
    left_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    right_path.write_text(json.dumps(reordered, indent=2), encoding="utf-8")

    left_bundle = cli_main._load_design_bundle(left_path.as_posix())
    right_bundle = cli_main._load_design_bundle(right_path.as_posix())
    left_manifest = build_manifest(
        input_payload=left_bundle.manifest_input_payload,
        resolved_params_payload=left_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(left_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )
    right_manifest = build_manifest(
        input_payload=right_bundle.manifest_input_payload,
        resolved_params_payload=right_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(right_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )

    assert left_manifest.input_hash != right_manifest.input_hash
    assert left_manifest.resolved_params_hash == right_manifest.resolved_params_hash


def test_loader_backed_manifest_hashes_preserve_flat_input_order_semantics(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "flat_left.json"
    right_path = tmp_path / "flat_right.json"
    baseline = _supported_bundle()
    left_design = baseline["design"]
    assert isinstance(left_design, dict)
    left_design["elements"] = [
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
            "params": {"resistance_ohm": 75.0},
        },
    ]
    left_design["ports"] = [
        {"id": "P1", "p_plus": "n1", "p_minus": "0"},
        {"id": "P2", "p_plus": "n2", "p_minus": "0"},
    ]
    reordered = json.loads(json.dumps(baseline))
    right_design = reordered["design"]
    assert isinstance(right_design, dict)
    right_design["elements"] = list(reversed(right_design["elements"]))
    right_design["ports"] = list(reversed(right_design["ports"]))
    left_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    right_path.write_text(json.dumps(reordered, indent=2), encoding="utf-8")

    left_bundle = cli_main._load_design_bundle(left_path.as_posix())
    right_bundle = cli_main._load_design_bundle(right_path.as_posix())
    left_manifest = build_manifest(
        input_payload=left_bundle.manifest_input_payload,
        resolved_params_payload=left_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(left_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )
    right_manifest = build_manifest(
        input_payload=right_bundle.manifest_input_payload,
        resolved_params_payload=right_bundle.manifest_resolved_params_payload,
        solver_config_snapshot={},
        frequency_grid_metadata={"n_points": len(right_bundle.frequencies_hz)},
        timestamp="2026-03-07T00:00:00+00:00",
        timezone="UTC",
    )

    assert left_manifest.input_hash != right_manifest.input_hash

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.parser import DESIGN_BUNDLE_SCHEMA_ID
from rfmna.viz_io import build_manifest

pytestmark = pytest.mark.regression

runner = CliRunner()
_EXIT_FAILURE: Final[int] = 2
_TWO_POINTS: Final[int] = 2


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


def _invalid_port_z0_bundle() -> dict[str, object]:
    payload = _supported_bundle()
    design = payload["design"]
    assert isinstance(design, dict)
    design["ports"] = [{"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": -5.0}]
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

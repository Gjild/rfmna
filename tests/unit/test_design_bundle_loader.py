from __future__ import annotations

import errno
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from rfmna.cli import design_loader as cli_design_loader
from rfmna.cli import main as cli_main
from rfmna.diagnostics import build_diagnostic_event
from rfmna.ir import hash_canonical_ir
from rfmna.parser import design_bundle as parser_design_bundle_module
from rfmna.parser._loader_exclusions_runtime import (
    LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
    load_loader_temp_exclusions_payload_text,
    load_packaged_loader_temp_exclusions_payload_text,
    repo_loader_temp_exclusions_artifact_path,
    source_tree_loader_temp_exclusions_resource_path,
)
from rfmna.parser.design_bundle import (
    DESIGN_BUNDLE_SCHEMA_ID,
    DesignBundleLoadError,
    load_design_bundle_document,
)

pytestmark = pytest.mark.unit

runner = CliRunner()
_DEFAULT_RF_Z0_OHM: Final[float] = 50.0
_TWO_POINTS: Final[int] = 2
_EXIT_FAILURE: Final[int] = 2
_EXPECTED_INTERIM_EXCLUSIONS: Final[tuple[str, ...]] = (
    "frequency_dependent_compact_linear_forms",
    "parameter_sweep_support",
    "y_block_elements",
    "z_block_elements",
)


def _base_bundle() -> dict[str, object]:
    return {
        "schema": DESIGN_BUNDLE_SCHEMA_ID,
        "schema_version": 1,
        "design": {
            "reference_node": "0",
            "parameters": {
                "r_load": "50",
            },
            "elements": [
                {
                    "id": "R1",
                    "kind": "R",
                    "nodes": ["n1", "0"],
                    "params": {"resistance_ohm": "r_load"},
                }
            ],
            "ports": [
                {
                    "id": "P1",
                    "p_plus": "n1",
                    "p_minus": "0",
                    "z0_ohm": 50.0,
                }
            ],
        },
        "analysis": {
            "type": "ac",
            "frequency_sweep": {
                "mode": "linear",
                "start": {"value": 1.0, "unit": "Hz"},
                "stop": {"value": 3.0, "unit": "Hz"},
                "points": 3,
            }
        },
    }


def _write_bundle(tmp_path: Path, payload: dict[str, object], *, name: str = "design.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_exclusion_artifact(
    tmp_path: Path,
    exclusions: list[dict[str, object]],
    *,
    include_notes: bool,
    name: str = "p3_loader_temporary_exclusions.yaml",
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "exclusions": exclusions,
    }
    if include_notes:
        payload["notes"] = "unit test fixture"
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _canonical_exclusion_entries(
    *,
    per_capability_updates: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    payload = yaml.safe_load(load_packaged_loader_temp_exclusions_payload_text())
    assert isinstance(payload, dict)
    exclusions = payload["exclusions"]
    assert isinstance(exclusions, list)
    updates = per_capability_updates or {}
    entries: list[dict[str, object]] = []
    for exclusion in exclusions:
        assert isinstance(exclusion, dict)
        entry = dict(exclusion)
        capability_id = entry["capability_id"]
        assert isinstance(capability_id, str)
        entry.update(dict(updates.get(capability_id, {})))
        entries.append(entry)
    return entries


def _override_packaged_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    exclusion_artifact: Path,
) -> None:
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_read_governed_loader_exclusions_payload",
        lambda: cast(
            dict[str, object],
            yaml.safe_load(exclusion_artifact.read_text(encoding="utf-8")),
        ),
    )


def test_parse_design_bundle_document_builds_expected_frequency_grid_and_rf_z0(tmp_path: Path) -> None:
    payload = _base_bundle()
    payload["analysis"] = {
        "type": "ac",
        "frequency_sweep": {
            "mode": "log",
            "start": {"value": 1.0, "unit": "MHz"},
            "stop": {"value": 1.0, "unit": "GHz"},
            "points": 4,
        }
    }
    path = _write_bundle(tmp_path, payload)

    parsed = load_design_bundle_document(path)

    np.testing.assert_allclose(
        parsed.frequencies_hz,
        np.asarray([1.0e6, 1.0e7, 1.0e8, 1.0e9], dtype=np.float64),
    )
    assert parsed.rf_z0_ohm == _DEFAULT_RF_Z0_OHM
    assert tuple(node.node_id for node in parsed.ir.nodes) == ("0", "n1")
    assert tuple(port.port_id for port in parsed.rf_ports) == ("P1",)
    assert tuple(port.port_id for port in parsed.ir.ports) == ("P1",)


def test_canonical_ir_hash_tracks_rf_port_declarations(tmp_path: Path) -> None:
    low_z0_payload = _base_bundle()
    low_z0_path = _write_bundle(tmp_path, low_z0_payload, name="z0_50.json")

    high_z0_payload = _base_bundle()
    high_z0_design = cast(dict[str, object], high_z0_payload["design"])
    high_z0_ports = cast(list[dict[str, object]], high_z0_design["ports"])
    high_z0_ports[0]["z0_ohm"] = 75.0
    high_z0_path = _write_bundle(tmp_path, high_z0_payload, name="z0_75.json")

    low_z0_parsed = load_design_bundle_document(low_z0_path)
    high_z0_parsed = load_design_bundle_document(high_z0_path)

    assert low_z0_parsed.ir.ports == low_z0_parsed.rf_ports
    assert high_z0_parsed.ir.ports == high_z0_parsed.rf_ports
    assert hash_canonical_ir(low_z0_parsed.ir) != hash_canonical_ir(high_z0_parsed.ir)


def test_loader_canonicalizes_rf_port_order_and_heterogeneous_z0_together(tmp_path: Path) -> None:
    payload = {
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
                {"id": "P1", "p_plus": "n1", "p_minus": "0", "z0_ohm": 75.0},
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
    path = _write_bundle(tmp_path, payload, name="heterogeneous_ports.json")

    parsed = load_design_bundle_document(path)
    bundle = cli_design_loader.load_design_bundle(path.as_posix())

    assert tuple(port.port_id for port in parsed.rf_ports) == ("P1", "P2")
    assert parsed.rf_z0_ohm == (75.0, 50.0)
    assert tuple(port.port_id for port in bundle.rf_ports) == ("P1", "P2")
    assert bundle.rf_z0_ohm == (75.0, 50.0)


def test_cli_loader_compiles_sparse_pattern_once_and_fills_each_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bundle(tmp_path, _base_bundle())
    compile_calls = {"count": 0}
    fill_calls = {"count": 0}
    original_compile_pattern = cli_design_loader.compile_pattern
    original_fill_numeric = cli_design_loader.fill_numeric

    def _compile_pattern(*args: object, **kwargs: object):
        compile_calls["count"] += 1
        return original_compile_pattern(*args, **kwargs)

    def _fill_numeric(*args: object, **kwargs: object):
        fill_calls["count"] += 1
        return original_fill_numeric(*args, **kwargs)

    monkeypatch.setattr(cli_design_loader, "compile_pattern", _compile_pattern)
    monkeypatch.setattr(cli_design_loader, "fill_numeric", _fill_numeric)

    bundle = cli_design_loader.load_design_bundle(path.as_posix())
    _ = bundle.assemble_point(0, 1.0)
    _ = bundle.assemble_point(1, 2.0)

    assert compile_calls["count"] == 1
    assert fill_calls["count"] == _TWO_POINTS


def test_schema_invalid_diagnostic_is_deterministic(tmp_path: Path) -> None:
    payload = _base_bundle()
    analysis = cast(dict[str, object], payload["analysis"])
    analysis.pop("type")
    path = _write_bundle(tmp_path, payload)

    with pytest.raises(DesignBundleLoadError) as first_exc:
        load_design_bundle_document(path)
    with pytest.raises(DesignBundleLoadError) as second_exc:
        load_design_bundle_document(path)

    first = first_exc.value.diagnostics[0].model_dump(mode="json", exclude_none=True)
    second = second_exc.value.diagnostics[0].model_dump(mode="json", exclude_none=True)
    assert first == second
    assert first["code"] == "E_CLI_DESIGN_SCHEMA_INVALID"
    assert first["witness"] == {"path": "$.analysis"}


def test_read_failure_diagnostic_message_is_stable_and_os_detail_stays_in_witness(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(missing_path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_READ_FAILED"
    assert diagnostic.message == "design bundle read failed"
    assert diagnostic.witness["design"] == missing_path.as_posix()
    assert diagnostic.witness["errno"] == errno.ENOENT
    assert diagnostic.witness["error_type"] == "FileNotFoundError"


def test_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_schema.json"
    path.write_text(
        """
        {
          "schema": "docs/spec/schemas/design_bundle_v0.json",
          "schema": "docs/spec/schemas/design_bundle_v1.json",
          "schema_version": 1,
          "design": {
            "reference_node": "0",
            "elements": []
          },
          "analysis": {
            "type": "ac",
            "frequency_sweep": {
              "mode": "linear",
              "start": {"value": 1.0, "unit": "Hz"},
              "stop": {"value": 1.0, "unit": "Hz"},
              "points": 1
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_PARSE_FAILED"
    assert diagnostic.message == "design bundle JSON contains duplicate object keys"
    assert diagnostic.witness == {
        "design": path.as_posix(),
        "key": "schema",
    }


def test_loader_rejects_duplicate_json_parameter_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_parameters.json"
    path.write_text(
        """
        {
          "schema": "docs/spec/schemas/design_bundle_v1.json",
          "schema_version": 1,
          "design": {
            "reference_node": "0",
            "parameters": {
              "r": 50,
              "r": 75
            },
            "elements": [
              {
                "id": "R1",
                "kind": "R",
                "nodes": ["n1", "0"],
                "params": {"resistance_ohm": "r"}
              }
            ]
          },
          "analysis": {
            "type": "ac",
            "frequency_sweep": {
              "mode": "linear",
              "start": {"value": 1.0, "unit": "Hz"},
              "stop": {"value": 1.0, "unit": "Hz"},
              "points": 1
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_PARSE_FAILED"
    assert diagnostic.message == "design bundle JSON contains duplicate object keys"
    assert diagnostic.witness == {
        "design": path.as_posix(),
        "key": "r",
    }


def test_loader_runtime_schema_contract_is_derived_from_design_bundle_schema_artifact() -> None:
    contract = parser_design_bundle_module._load_design_bundle_schema_contract()

    assert contract.root_allowed_keys == ("analysis", "design", "schema", "schema_version")
    assert contract.root_required_keys == ("schema", "schema_version", "design", "analysis")
    assert contract.design_allowed_keys == (
        "elements",
        "instances",
        "macros",
        "nodes",
        "parameters",
        "ports",
        "reference_node",
        "subcircuits",
    )
    assert contract.design_required_keys == ("reference_node", "elements")
    assert contract.analysis_allowed_keys == ("frequency_sweep", "parameter_sweeps", "type")
    assert contract.analysis_required_keys == ("type", "frequency_sweep")
    assert contract.analysis_type == "ac"
    assert "RESISTOR" in contract.accepted_element_kind_tokens
    assert "Y1P" in contract.accepted_element_kind_tokens
    assert contract.supported_kind_node_counts["R"] == parser_design_bundle_module._TWO_NODE_COUNT
    assert contract.supported_kind_node_counts["VCVS"] == parser_design_bundle_module._FOUR_NODE_COUNT
    assert contract.supported_kind_required_params["R"] == ("resistance_ohm",)
    assert contract.supported_kind_required_params["VCVS"] == ("gain_mu",)
    assert contract.frequency_sweep_modes == ("linear", "log")
    assert contract.frequency_units == ("Hz", "kHz", "MHz", "GHz")
    assert contract.hierarchy_instance_types == ("macro", "subcircuit")


def test_schema_artifact_and_loader_both_reject_whitespace_only_strings(tmp_path: Path) -> None:
    payload = _base_bundle()
    design = cast(dict[str, object], payload["design"])
    design["reference_node"] = "   "
    path = _write_bundle(tmp_path, payload, name="whitespace_reference.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_SCHEMA_INVALID"
    assert diagnostic.witness == {"path": "$.design.reference_node"}


def test_loader_rejects_whitespace_only_scalar_mapping_keys(tmp_path: Path) -> None:
    payload = _base_bundle()
    design = cast(dict[str, object], payload["design"])
    design["parameters"] = {"   ": 1.0}
    path = _write_bundle(tmp_path, payload, name="whitespace_parameter_key.json")

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_SCHEMA_INVALID"
    assert diagnostic.witness == {"path": "$.design.parameters"}


def test_parameter_sweep_exclusion_emits_cataloged_loader_diagnostic(tmp_path: Path) -> None:
    payload = _base_bundle()
    analysis = cast(dict[str, object], payload["analysis"])
    analysis["parameter_sweeps"] = [{"parameter": "r_load", "values": [10.0, 20.0]}]
    path = _write_bundle(tmp_path, payload)

    with pytest.raises(DesignBundleLoadError) as exc_info:
        load_design_bundle_document(path)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"
    assert diagnostic.solver_stage == "parse"
    assert diagnostic.element_id == "cli.design_loader"
    assert diagnostic.witness == {
        "capability_id": "parameter_sweep_support",
        "parameter": "r_load",
        "path": "analysis.parameter_sweeps",
    }


def test_unsupported_kind_loader_diagnostic_preserves_canonical_assemble_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _base_bundle()
    design = cast(dict[str, object], payload["design"])
    design["elements"] = [
        {
            "id": "X1",
            "kind": "Y1P",
            "nodes": ["n1", "0"],
            "params": {"y11_s": 0.02},
        }
    ]
    path = _write_bundle(tmp_path, payload)
    exclusion_artifact = _write_exclusion_artifact(tmp_path, [], include_notes=False)
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_expected_interim_governed_exclusions",
        lambda: (),
    )
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_IR_KIND_UNKNOWN"
    assert diagnostic.solver_stage == "assemble"
    assert diagnostic.witness == {
        "element_id": "X1",
        "deferred_capability_id": "y_block_elements",
        "normalized_candidate": "Y1P",
        "policy_state": "not_deferred",
        "raw_kind": "Y1P",
        "supported_kinds": ["C", "G", "I", "L", "R", "V", "VCCS", "VCVS"],
    }


def test_loader_blocks_when_governed_exclusion_artifact_drifts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle())
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        [
            {
                "capability_id": "y_block_elements",
                "label": "duplicate y block",
                "status": "interim_deferred",
                "check_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "run_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "witness_capability_id": "y_block_elements",
            },
            {
                "capability_id": "y_block_elements",
                "label": "duplicate y block again",
                "status": "interim_deferred",
                "check_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "run_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "witness_capability_id": "y_block_elements",
            },
        ],
        include_notes=True,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "capability_id": "y_block_elements",
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
    }


def test_loader_accepts_empty_exclusion_artifact_as_closure_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supported_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        [],
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(supported_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "actual_capability_ids": [],
        "drifted_capability_ids": [],
        "expected_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "missing_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "mismatched_fields_by_capability": {},
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
        "unexpected_capability_ids": [],
    }


def test_loader_rejects_missing_governed_artifact_in_source_checkout_while_interim_exclusions_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    missing_artifact_path = tmp_path / "docs" / "dev" / "p3_loader_temporary_exclusions.yaml"
    missing_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "rfmna.parser._loader_exclusions_runtime.repo_loader_temp_exclusions_artifact_path",
        lambda: missing_artifact_path,
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "repo_loader_temp_exclusions_artifact_path",
        lambda: missing_artifact_path,
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "load_loader_temp_exclusions_payload_text",
        load_loader_temp_exclusions_payload_text,
    )
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "expected_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
    }


def test_loader_reads_source_tree_resource_when_governed_artifact_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "no-such-parent" / "p3_loader_temporary_exclusions.yaml"
    monkeypatch.setattr(
        "rfmna.parser._loader_exclusions_runtime.repo_loader_temp_exclusions_artifact_path",
        lambda: missing_path,
    )

    payload = yaml.safe_load(load_loader_temp_exclusions_payload_text())
    resource_payload = yaml.safe_load(
        source_tree_loader_temp_exclusions_resource_path().read_text(encoding="utf-8")
    )

    assert payload == resource_payload


def test_loader_reads_packaged_schema_resource_when_repo_schema_and_source_tree_schema_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "no-such-parent" / "design_bundle_v1.json"
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_repo_design_bundle_schema_artifact_path",
        lambda: missing_path,
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_source_tree_design_bundle_schema_resource_path",
        lambda: missing_path,
    )
    parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    try:
        payload = json.loads(
            parser_design_bundle_module._load_packaged_design_bundle_schema_payload_text()
        )
    finally:
        parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    resource_payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "src/rfmna/parser/resources/design_bundle_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert payload == resource_payload


def test_loader_uses_governed_schema_artifact_when_packaged_schema_mirror_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    governed_schema_path = tmp_path / "design_bundle_v1.json"
    packaged_schema_path = tmp_path / "packaged_design_bundle_v1.json"
    governed_schema = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/spec/schemas/design_bundle_v1.json").read_text(
            encoding="utf-8"
        )
    )
    packaged_schema = json.loads(json.dumps(governed_schema))
    governed_properties = governed_schema["properties"]
    assert isinstance(governed_properties, dict)
    governed_properties["test_only"] = {
        "type": "object",
        "additionalProperties": True,
    }
    governed_schema["additionalProperties"] = False
    payload = _base_bundle()
    payload["test_only"] = {"flag": True}
    design_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    governed_schema_path.write_text(json.dumps(governed_schema, indent=2), encoding="utf-8")
    packaged_schema_path.write_text(json.dumps(packaged_schema, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_repo_design_bundle_schema_artifact_path",
        lambda: governed_schema_path,
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_load_packaged_design_bundle_schema_payload_text",
        lambda: packaged_schema_path.read_text(encoding="utf-8"),
    )
    parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    try:
        parsed = load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    assert tuple(node.node_id for node in parsed.ir.nodes) == ("0", "n1")


def test_loader_uses_governed_schema_artifact_when_packaged_schema_mirror_is_missing_in_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    governed_schema_path = tmp_path / "design_bundle_v1.json"
    governed_schema_path.write_text(
        (
            Path(__file__).resolve().parents[2] / "docs/spec/schemas/design_bundle_v1.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_repo_design_bundle_schema_artifact_path",
        lambda: governed_schema_path,
    )
    monkeypatch.setattr(
        parser_design_bundle_module,
        "_load_packaged_design_bundle_schema_payload_text",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing packaged schema mirror")),
    )
    parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    try:
        parsed = load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_design_bundle_schema_contract.cache_clear()

    assert parsed.rf_z0_ohm == _DEFAULT_RF_Z0_OHM


def test_wheel_force_includes_tracked_runtime_schema_mirror_from_src_tree() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert (
        force_include["src/rfmna/parser/resources/design_bundle_v1.json"]
        == "rfmna/parser/resources/design_bundle_v1.json"
    )
    assert "docs/spec/schemas/design_bundle_v1.json" not in force_include
    assert tuple(force_include.values()).count("rfmna/parser/resources/design_bundle_v1.json") == 1


def test_loader_fails_closed_when_no_runtime_exclusions_source_exists_in_installed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parser_design_bundle_module,
        "load_loader_temp_exclusions_payload_text",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing exclusions artifact")),
    )
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            parser_design_bundle_module._read_governed_loader_exclusions_payload()
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE}


def test_loader_accepts_normal_yaml_exclusion_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = tmp_path / "p3_loader_temporary_exclusions.yaml"
    exclusion_artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "exclusions": _canonical_exclusion_entries(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        parsed = load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    assert parsed.rf_z0_ohm == _DEFAULT_RF_Z0_OHM


def test_loader_rejects_duplicate_yaml_exclusion_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    duplicate_yaml = """
schema_version: 1
exclusions:
  - capability_id: y_block_elements
    label: y block
    status: interim_deferred
    status: retired
    check_diagnostic_code: E_CLI_DESIGN_EXCLUDED_CAPABILITY
    run_diagnostic_code: E_CLI_DESIGN_EXCLUDED_CAPABILITY
    witness_capability_id: y_block_elements
""".lstrip()
    monkeypatch.setattr(
        parser_design_bundle_module,
        "load_loader_temp_exclusions_payload_text",
        lambda: duplicate_yaml,
    )
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.message == "loader temporary exclusions artifact contains duplicate mapping keys"
    assert diagnostic.witness == {
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
        "key": "status",
        "line": 6,
        "column": 5,
    }


def test_loader_rejects_partial_exclusion_artifact_while_interim_exclusions_are_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        [entry for entry in _canonical_exclusion_entries() if entry["capability_id"] == "y_block_elements"],
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "actual_capability_ids": ["y_block_elements"],
        "drifted_capability_ids": [],
        "expected_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "missing_capability_ids": [
            "frequency_dependent_compact_linear_forms",
            "parameter_sweep_support",
            "z_block_elements",
        ],
        "mismatched_fields_by_capability": {},
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
        "unexpected_capability_ids": [],
    }


def test_loader_rejects_exclusion_artifact_extra_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = tmp_path / "p3_loader_temporary_exclusions.yaml"
    exclusion_artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "exclusions": [
                    {
                        "capability_id": "y_block_elements",
                        "label": "y block",
                        "status": "interim_deferred",
                        "check_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                        "run_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                        "witness_capability_id": "y_block_elements",
                        "unexpected": "extra",
                    }
                ],
                "unexpected_root": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "path": "$",
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
        "unsupported_keys": ["unexpected_root"],
    }


def test_loader_accepts_schema_valid_exclusion_artifact_without_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle())
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(),
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        parsed = load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    assert parsed.rf_z0_ohm == _DEFAULT_RF_Z0_OHM


def test_packaged_loader_exclusion_policy_matches_governed_artifact() -> None:
    packaged = yaml.safe_load(load_packaged_loader_temp_exclusions_payload_text())
    governed = yaml.safe_load(
        repo_loader_temp_exclusions_artifact_path().read_text(encoding="utf-8")
    )

    assert packaged == governed


def test_loader_run_path_uses_governed_run_diagnostic_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _base_bundle()
    design = cast(dict[str, object], payload["design"])
    design["elements"] = [
        {
            "id": "Y1",
            "kind": "Y1P",
            "nodes": ["n1", "0"],
            "params": {"y11_s": 0.02},
        }
    ]
    design_path = _write_bundle(tmp_path, payload, name="excluded.json")
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(),
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        check_result = runner.invoke(
            cli_main.app,
            ["check", design_path.as_posix(), "--format", "json"],
        )
        run_result = runner.invoke(
            cli_main.app,
            ["run", design_path.as_posix(), "--analysis", "ac"],
        )
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    check_payload = json.loads(check_result.stdout)
    assert check_payload["diagnostics"][0]["code"] == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"
    assert "code=E_CLI_DESIGN_EXCLUDED_CAPABILITY" in run_result.stdout


def test_loader_run_invalid_exclusion_policy_returns_typed_exit_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        [
            {
                "capability_id": "y_block_elements",
                "label": "duplicate y block",
                "status": "interim_deferred",
                "check_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "run_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "witness_capability_id": "y_block_elements",
            },
            {
                "capability_id": "y_block_elements",
                "label": "duplicate y block again",
                "status": "interim_deferred",
                "check_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "run_diagnostic_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                "witness_capability_id": "y_block_elements",
            },
        ],
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        result = runner.invoke(cli_main.app, ["run", design_path.as_posix(), "--analysis", "ac"])
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    assert result.exit_code == _EXIT_FAILURE
    assert not isinstance(result.exception, DesignBundleLoadError)
    assert "code=E_CLI_DESIGN_EXCLUSION_POLICY_INVALID" in result.stdout


def test_adapt_loader_diagnostics_for_run_uses_structured_exclusion_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(),
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        adapted = parser_design_bundle_module.adapt_loader_diagnostics_for_command(
            (
                build_diagnostic_event(
                    code="E_CLI_DESIGN_EXCLUDED_CAPABILITY",
                    message="governed capability deferred by policy",
                    element_id="Y1",
                    solver_stage="parse",
                    severity="error",
                    suggested_action="remove the deferred capability",
                    witness={"capability_id": "y_block_elements"},
                ),
            ),
            command="run",
        )
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    assert adapted[0].code == "E_CLI_DESIGN_EXCLUDED_CAPABILITY"


def test_loader_rejects_non_exclusion_runtime_code_in_exclusion_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    exclusion_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(
            per_capability_updates={
                "y_block_elements": {
                    "run_diagnostic_code": "E_CLI_DESIGN_SCHEMA_INVALID",
                }
            }
        ),
        include_notes=False,
    )
    _override_packaged_exclusions(monkeypatch, exclusion_artifact)
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "capability_id": "y_block_elements",
        "code": "E_CLI_DESIGN_SCHEMA_INVALID",
        "expected_code": "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
        "field_name": "run_diagnostic_code",
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
    }


def test_loader_blocks_when_governed_exclusion_artifact_drifts_from_packaged_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design_path = _write_bundle(tmp_path, _base_bundle(), name="supported.json")
    governed_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(
            per_capability_updates={
                "y_block_elements": {
                    "label": "changed y block label",
                }
            }
        ),
        include_notes=False,
    )
    packaged_artifact = _write_exclusion_artifact(
        tmp_path,
        _canonical_exclusion_entries(),
        include_notes=False,
        name="packaged_p3_loader_temporary_exclusions.yaml",
    )
    _override_packaged_exclusions(monkeypatch, governed_artifact)
    monkeypatch.setattr(
        parser_design_bundle_module,
        "load_packaged_loader_temp_exclusions_payload_text",
        lambda: packaged_artifact.read_text(encoding="utf-8"),
    )
    parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    try:
        with pytest.raises(DesignBundleLoadError) as exc_info:
            load_design_bundle_document(design_path)
    finally:
        parser_design_bundle_module._load_governed_loader_exclusions.cache_clear()

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
    assert diagnostic.witness == {
        "actual_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "drifted_capability_ids": ["y_block_elements"],
        "expected_capability_ids": list(_EXPECTED_INTERIM_EXCLUSIONS),
        "missing_capability_ids": [],
        "mismatched_fields_by_capability": {"y_block_elements": ["label"]},
        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
        "unexpected_capability_ids": [],
    }

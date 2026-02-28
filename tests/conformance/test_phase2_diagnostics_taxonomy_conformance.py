from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rfmna.diagnostics import CANONICAL_DIAGNOSTIC_CATALOG
from rfmna.governance import diagnostics_taxonomy
from rfmna.governance.diagnostics_taxonomy import (
    load_runtime_diagnostic_inventory,
    load_typed_error_mapping_matrix,
    load_typed_error_registry,
    validate_runtime_diagnostic_inventory,
    validate_typed_error_registry,
)

pytestmark = pytest.mark.conformance


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_inventory_path() -> Path:
    return _repo_root() / "docs/dev/diagnostic_runtime_code_inventory.yaml"


def _typed_registry_path() -> Path:
    return _repo_root() / "docs/dev/typed_error_code_registry.yaml"


def _typed_matrix_path() -> Path:
    return _repo_root() / "docs/dev/typed_error_mapping_matrix.yaml"


def _catalog_codes() -> set[str]:
    return set(CANONICAL_DIAGNOSTIC_CATALOG)


def test_track_a_runtime_inventory_guard_passes_baseline() -> None:
    inventory = load_runtime_diagnostic_inventory(_runtime_inventory_path())
    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )
    assert errors == ()


def test_track_a_runtime_inventory_guard_fails_when_catalog_lacks_emitted_runtime_code() -> None:
    inventory = load_runtime_diagnostic_inventory(_runtime_inventory_path())
    catalog_codes = _catalog_codes()
    catalog_codes.remove("E_NUM_SOLVE_FAILED")

    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=catalog_codes,
    )

    assert errors
    assert any("uncataloged codes" in error for error in errors)
    assert any("E_NUM_SOLVE_FAILED" in error for error in errors)


def test_track_a_runtime_inventory_guard_fails_on_missing_runtime_emission_path(
    tmp_path: Path,
) -> None:
    baseline_inventory = json.loads(_runtime_inventory_path().read_text(encoding="utf-8"))
    runtime_paths = baseline_inventory["runtime_emission_paths"]
    assert isinstance(runtime_paths, list)
    baseline_inventory["runtime_emission_paths"] = [
        path for path in runtime_paths if path != "src/rfmna/rf_metrics/y_params.py"
    ]

    inventory_path = tmp_path / "diagnostic_runtime_code_inventory.yaml"
    inventory_path.write_text(json.dumps(baseline_inventory, indent=2), encoding="utf-8")

    inventory = load_runtime_diagnostic_inventory(inventory_path)
    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing runtime emission path" in error for error in errors)
    assert any("src/rfmna/rf_metrics/y_params.py" in error for error in errors)


def test_track_a_runtime_inventory_guard_fails_on_misclassified_runtime_path(
    tmp_path: Path,
) -> None:
    baseline_inventory = json.loads(_runtime_inventory_path().read_text(encoding="utf-8"))
    runtime_paths = baseline_inventory["runtime_emission_paths"]
    non_diag_paths = baseline_inventory["non_diagnostic_scoped_paths"]
    assert isinstance(runtime_paths, list)
    assert isinstance(non_diag_paths, list)
    moved_path = "src/rfmna/rf_metrics/y_params.py"
    baseline_inventory["runtime_emission_paths"] = [
        path for path in runtime_paths if path != moved_path
    ]
    baseline_inventory["non_diagnostic_scoped_paths"] = sorted(set((*non_diag_paths, moved_path)))

    inventory_path = tmp_path / "diagnostic_runtime_code_inventory.yaml"
    inventory_path.write_text(json.dumps(baseline_inventory, indent=2), encoding="utf-8")

    inventory = load_runtime_diagnostic_inventory(inventory_path)
    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("misclassifies runtime emission path" in error for error in errors)
    assert any("src/rfmna/rf_metrics/y_params.py" in error for error in errors)


def test_track_a_runtime_inventory_guard_fails_on_unscoped_code_bearing_path(
    tmp_path: Path,
) -> None:
    baseline_inventory = json.loads(_runtime_inventory_path().read_text(encoding="utf-8"))
    non_diag_paths = baseline_inventory["non_diagnostic_scoped_paths"]
    assert isinstance(non_diag_paths, list)
    baseline_inventory["non_diagnostic_scoped_paths"] = [
        path for path in non_diag_paths if path != "src/rfmna/parser/errors.py"
    ]

    inventory_path = tmp_path / "diagnostic_runtime_code_inventory.yaml"
    inventory_path.write_text(json.dumps(baseline_inventory, indent=2), encoding="utf-8")

    inventory = load_runtime_diagnostic_inventory(inventory_path)
    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("scope missing code-bearing path" in error for error in errors)
    assert any("src/rfmna/parser/errors.py" in error for error in errors)


def test_track_a_runtime_inventory_guard_fails_on_discovered_uncataloged_runtime_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = load_runtime_diagnostic_inventory(_runtime_inventory_path())

    monkeypatch.setattr(
        diagnostics_taxonomy,
        "_discover_runtime_emission_candidates",
        lambda *, repo_root, typed_error_only_families: (
            ("src/rfmna/rf_metrics/y_params.py",),
            ("E_NUM_RUNTIME_UNCATALOGED",),
            (),
        ),
    )

    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("uncataloged emitted runtime codes" in error for error in errors)
    assert any("E_NUM_RUNTIME_UNCATALOGED" in error for error in errors)


def test_track_a_runtime_derivation_ignores_comment_and_docstring_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_text = "\n".join(
        (
            '"""E_NUM_DOCSTRING_TOKEN"""',
            "# E_NUM_COMMENT_TOKEN",
            'code = "E_NUM_REAL_EMITTED"',
        )
    )
    monkeypatch.setattr(
        diagnostics_taxonomy,
        "_read_repo_file",
        lambda *, repo_root, rel_path: source_text,
    )

    derived = diagnostics_taxonomy.derive_runtime_emitted_codes(
        repo_root=_repo_root(),
        runtime_emission_paths=("src/rfmna/rf_metrics/fake_runtime.py",),
        typed_error_only_families=(),
    )

    assert derived == ("E_NUM_REAL_EMITTED",)


def test_track_a_runtime_derivation_filters_typed_only_families_from_matrix_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_text = "\n".join(
        (
            'typed_only = "E_DYNAMIC_TYPED_ONLY"',
            'runtime_diag = "E_NUM_RUNTIME_EMITTED"',
        )
    )
    monkeypatch.setattr(
        diagnostics_taxonomy,
        "_read_repo_file",
        lambda *, repo_root, rel_path: source_text,
    )

    derived = diagnostics_taxonomy.derive_runtime_emitted_codes(
        repo_root=_repo_root(),
        runtime_emission_paths=("src/rfmna/rf_metrics/fake_runtime.py",),
        typed_error_only_families=("E_DYNAMIC_*",),
    )

    assert derived == ("E_NUM_RUNTIME_EMITTED",)


def test_track_a_runtime_inventory_guard_fails_on_typed_only_runtime_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = load_runtime_diagnostic_inventory(_runtime_inventory_path())
    monkeypatch.setattr(
        diagnostics_taxonomy,
        "_discover_typed_only_codes_on_runtime_paths",
        lambda *, repo_root, runtime_emission_paths, typed_error_only_families: (
            "E_PARSE_UNIT_INVALID@src/rfmna/rf_metrics/y_params.py",
        ),
    )

    errors = validate_runtime_diagnostic_inventory(
        repo_root=_repo_root(),
        inventory=inventory,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("typed_error_only code emissions" in error for error in errors)
    assert any("E_PARSE_UNIT_INVALID@src/rfmna/rf_metrics/y_params.py" in error for error in errors)


def test_track_b_typed_registry_matrix_guard_passes_baseline() -> None:
    matrix = load_typed_error_mapping_matrix(_typed_matrix_path())
    registry = load_typed_error_registry(_typed_registry_path())
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )
    assert errors == ()


def test_minimum_taxonomy_runtime_codes_are_cataloged_and_typed_only_parse_codes_are_registered() -> (
    None
):
    catalog_codes = _catalog_codes()
    registry = load_typed_error_registry(_typed_registry_path())
    registry_codes = {entry.code for entry in registry.entries}

    minimum_runtime_subset = {
        "E_MODEL_PORT_Z0_COMPLEX",
        "E_MODEL_PORT_Z0_NONPOSITIVE",
        "E_TOPO_FLOATING_NODE",
        "E_TOPO_HARD_CONSTRAINT_CONFLICT",
        "E_TOPO_PORT_INVALID",
        "E_TOPO_REFERENCE_INVALID",
        "E_TOPO_VSRC_LOOP_INCONSISTENT",
        "E_NUM_SOLVE_FAILED",
        "E_NUM_SINGULAR_MATRIX",
        "E_NUM_ZBLOCK_SINGULAR",
        "E_NUM_ZBLOCK_ILL_CONDITIONED",
        "E_NUM_S_CONVERSION_SINGULAR",
        "W_NUM_COND_UNAVAILABLE",
        "W_NUM_ILL_CONDITIONED",
        "W_RF_RECIPROCITY",
        "W_RF_PASSIVITY",
    }
    minimum_typed_only_subset = {
        "E_MODEL_PARAM_CYCLE",
        "E_MODEL_FREQ_GRID_INVALID",
        "E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN",
    }

    assert minimum_runtime_subset <= catalog_codes
    assert minimum_typed_only_subset <= registry_codes


def test_track_b_registry_loader_rejects_duplicate_typed_codes(tmp_path: Path) -> None:
    baseline = json.loads(_typed_registry_path().read_text(encoding="utf-8"))
    entries = baseline["entries"]
    assert isinstance(entries, list)
    duplicate_entry = copy.deepcopy(entries[0])
    entries.append(duplicate_entry)
    entries.sort(key=lambda item: item["code"])

    target = tmp_path / "typed_error_code_registry.yaml"
    target.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate code"):
        load_typed_error_registry(target)


def test_track_b_registry_loader_accepts_factory_family_wildcard_code(tmp_path: Path) -> None:
    baseline = json.loads(_typed_registry_path().read_text(encoding="utf-8"))
    entries = baseline["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        if entry["code"] == "FACTORY_CONSTRUCTOR_FAILED":
            entry["code"] = "FACTORY_CONSTRUCTOR_INVALID"
            break
    else:  # pragma: no cover - deterministic fixture guard
        raise AssertionError("baseline registry missing FACTORY_CONSTRUCTOR_FAILED")
    entries.sort(key=lambda item: item["code"])

    target = tmp_path / "typed_error_code_registry.yaml"
    target.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    loaded = load_typed_error_registry(target)
    assert any(entry.code == "FACTORY_CONSTRUCTOR_INVALID" for entry in loaded.entries)


def test_track_b_guard_fails_on_missing_discovered_registry_code(tmp_path: Path) -> None:
    baseline_registry = json.loads(_typed_registry_path().read_text(encoding="utf-8"))
    entries = baseline_registry["entries"]
    assert isinstance(entries, list)
    baseline_registry["entries"] = [
        entry for entry in entries if entry["code"] != "E_PARSE_UNIT_INVALID"
    ]

    registry_path = tmp_path / "typed_error_code_registry.yaml"
    registry_path.write_text(json.dumps(baseline_registry, indent=2), encoding="utf-8")

    matrix = load_typed_error_mapping_matrix(_typed_matrix_path())
    registry = load_typed_error_registry(registry_path)
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing discovered code" in error for error in errors)
    assert any("E_PARSE_UNIT_INVALID" in error for error in errors)


def test_track_b_guard_fails_on_required_family_mode_violation(tmp_path: Path) -> None:
    baseline_matrix = json.loads(_typed_matrix_path().read_text(encoding="utf-8"))
    families = baseline_matrix["families"]
    assert isinstance(families, list)
    baseline_matrix["families"] = [row for row in families if row["family"] != "E_PARSE_*"]

    matrix_path = tmp_path / "typed_error_mapping_matrix.yaml"
    matrix_path.write_text(json.dumps(baseline_matrix, indent=2), encoding="utf-8")

    matrix = load_typed_error_mapping_matrix(matrix_path)
    registry = load_typed_error_registry(_typed_registry_path())
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing required family: E_PARSE_*" in error for error in errors)


def test_track_b_guard_fails_when_matrix_omits_registry_source_path(tmp_path: Path) -> None:
    baseline_matrix = json.loads(_typed_matrix_path().read_text(encoding="utf-8"))
    families = baseline_matrix["families"]
    assert isinstance(families, list)
    for family in families:
        if family["family"] == "E_MODEL_STAMP_*":
            paths = family["source_paths"]
            assert isinstance(paths, list)
            family["source_paths"] = [
                path for path in paths if path != "src/rfmna/elements/controlled.py"
            ]
            break
    else:  # pragma: no cover - deterministic fixture guard
        raise AssertionError("baseline matrix missing E_MODEL_STAMP_* family")

    matrix_path = tmp_path / "typed_error_mapping_matrix.yaml"
    matrix_path.write_text(json.dumps(baseline_matrix, indent=2), encoding="utf-8")

    matrix = load_typed_error_mapping_matrix(matrix_path)
    registry = load_typed_error_registry(_typed_registry_path())
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing source path" in error for error in errors)
    assert any("src/rfmna/elements/controlled.py" in error for error in errors)


def test_track_b_guard_fails_when_registry_and_matrix_omit_discovered_source_path(
    tmp_path: Path,
) -> None:
    baseline_matrix = json.loads(_typed_matrix_path().read_text(encoding="utf-8"))
    matrix_families = baseline_matrix["families"]
    assert isinstance(matrix_families, list)
    for family in matrix_families:
        if family["family"] == "E_MODEL_STAMP_*":
            paths = family["source_paths"]
            assert isinstance(paths, list)
            family["source_paths"] = [
                path for path in paths if path != "src/rfmna/elements/controlled.py"
            ]
            break
    else:  # pragma: no cover - deterministic fixture guard
        raise AssertionError("baseline matrix missing E_MODEL_STAMP_* family")

    baseline_registry = json.loads(_typed_registry_path().read_text(encoding="utf-8"))
    registry_entries = baseline_registry["entries"]
    assert isinstance(registry_entries, list)
    for entry in registry_entries:
        if entry["family"] == "E_MODEL_STAMP_*":
            paths = entry["source_paths"]
            assert isinstance(paths, list)
            entry["source_paths"] = [
                path for path in paths if path != "src/rfmna/elements/controlled.py"
            ]

    matrix_path = tmp_path / "typed_error_mapping_matrix.yaml"
    matrix_path.write_text(json.dumps(baseline_matrix, indent=2), encoding="utf-8")
    registry_path = tmp_path / "typed_error_code_registry.yaml"
    registry_path.write_text(json.dumps(baseline_registry, indent=2), encoding="utf-8")

    matrix = load_typed_error_mapping_matrix(matrix_path)
    registry = load_typed_error_registry(registry_path)
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing discovered source path" in error for error in errors)
    assert any("src/rfmna/elements/controlled.py" in error for error in errors)


def test_track_b_guard_fails_on_unmapped_diagnostic_equivalent_required_entry(
    tmp_path: Path,
) -> None:
    baseline_registry = json.loads(_typed_registry_path().read_text(encoding="utf-8"))
    entries = baseline_registry["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        if entry["code"] == "E_IR_KIND_UNKNOWN":
            entry["diagnostic_equivalent_codes"] = []
            break
    else:  # pragma: no cover - deterministic fixture guard
        raise AssertionError("baseline registry missing E_IR_KIND_UNKNOWN")

    registry_path = tmp_path / "typed_error_code_registry.yaml"
    registry_path.write_text(json.dumps(baseline_registry, indent=2), encoding="utf-8")

    matrix = load_typed_error_mapping_matrix(_typed_matrix_path())
    registry = load_typed_error_registry(registry_path)
    errors = validate_typed_error_registry(
        repo_root=_repo_root(),
        registry=registry,
        matrix=matrix,
        catalog_codes=_catalog_codes(),
    )

    assert errors
    assert any("missing diagnostic mapping" in error for error in errors)
    assert any("E_IR_KIND_UNKNOWN" in error for error in errors)

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.assembler import UnknownIndexing, build_unknown_indexing, compile_pattern, fill_numeric
from rfmna.elements import (
    CapacitorStamp,
    ConductanceStamp,
    ElementStamp,
    InductorStamp,
    ResistorStamp,
    StampContext,
    VCCSStamp,
)
from rfmna.governance.regression_fixture_schema import (
    validate_fixture_schema_document,
    validate_json_against_schema,
)
from rfmna.parser import parse_scalar_number
from rfmna.rf_metrics import (
    PortBoundary,
    convert_y_to_s,
    convert_z_to_s,
    extract_y_parameters,
    extract_z_parameters,
)

pytestmark = pytest.mark.regression

_FIXTURE_SCHEMA_PATH = "tests/regression/schemas/rf_regression_fixture_v1.schema.json"
_FIXTURE_HASH_LOCK_PATH = "tests/fixtures/regression/approved_hashes_v1.json"
_FIXTURE_DIR_PATH = "tests/fixtures/regression"
_TOLERANCE_PATH = "docs/dev/tolerances/regression_baseline_v1.yaml"
_CLASSIFICATION_PATH = "docs/dev/threshold_tolerance_classification.yaml"
_APPROVAL_SCRIPT_PATH = "scripts/regression/approve_fixture_hashes.py"
_REGRESSION_TOLERANCE_SOURCE = "docs/dev/tolerances/regression_baseline_v1.yaml"
_COMPLEX_PAIR_SIZE = 2


class _RFResultLike(Protocol):
    status: np.ndarray
    diagnostics_by_point: tuple[tuple[object, ...], ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path.as_posix()}")
    return payload


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fixture_paths() -> tuple[Path, ...]:
    fixture_dir = _repo_root() / _FIXTURE_DIR_PATH
    paths = tuple(
        path
        for path in sorted(fixture_dir.glob("*.json"))
        if path.name != Path(_FIXTURE_HASH_LOCK_PATH).name
    )
    assert paths, "expected at least one regression fixture under tests/fixtures/regression"
    return paths


def _load_hash_lock() -> dict[str, str]:
    payload = _load_json_object(_repo_root() / _FIXTURE_HASH_LOCK_PATH)
    assert payload["schema_version"] == 1
    assert payload["policy"] == "canonical_json_sha256"
    raw_entries = payload.get("hashes")
    assert isinstance(raw_entries, list) and raw_entries

    entries: dict[str, str] = {}
    for entry in raw_entries:
        assert isinstance(entry, dict)
        fixture_path = entry.get("fixture")
        digest = entry.get("canonical_sha256")
        assert isinstance(fixture_path, str) and fixture_path
        assert isinstance(digest, str) and digest
        entries[fixture_path] = digest
    return entries


def _load_tolerance_table() -> dict[str, dict[str, float]]:
    payload = _load_json_object(_repo_root() / _TOLERANCE_PATH)
    assert payload["schema_version"] == 1
    assert payload["artifact_id"] == "regression_baseline_v1"
    assert payload["status"] == "normative_gating"
    profiles_raw = payload.get("profiles")
    assert isinstance(profiles_raw, dict) and profiles_raw
    profiles: dict[str, dict[str, float]] = {}
    for profile_name, raw_profile in profiles_raw.items():
        assert isinstance(profile_name, str) and profile_name
        assert isinstance(raw_profile, dict)
        rtol = raw_profile.get("rtol")
        atol = raw_profile.get("atol")
        assert isinstance(rtol, (float, int))
        assert isinstance(atol, (float, int))
        profiles[profile_name] = {"rtol": float(rtol), "atol": float(atol)}
    return profiles


def _load_fixture_schema() -> dict[str, object]:
    payload = _load_json_object(_repo_root() / _FIXTURE_SCHEMA_PATH)
    schema_errors = validate_fixture_schema_document(payload)
    assert schema_errors == ()
    return payload


def _decode_complex_cube(raw: object) -> np.ndarray:
    if not isinstance(raw, list) or not raw:
        raise TypeError("complex cube must be a non-empty list")

    points: list[list[list[complex]]] = []
    for raw_point in raw:
        if not isinstance(raw_point, list) or not raw_point:
            raise TypeError("complex cube point must be a non-empty list")
        point_rows: list[list[complex]] = []
        for raw_row in raw_point:
            if not isinstance(raw_row, list) or not raw_row:
                raise TypeError("complex cube row must be a non-empty list")
            row_values: list[complex] = []
            for raw_pair in raw_row:
                if not isinstance(raw_pair, list) or len(raw_pair) != _COMPLEX_PAIR_SIZE:
                    raise TypeError("complex cube value must be [real, imag]")
                real, imag = raw_pair
                if real is None and imag is None:
                    row_values.append(np.nan + 1j * np.nan)
                    continue
                if real is None or imag is None:
                    raise TypeError("complex sentinel value must be [null, null]")
                if not isinstance(real, (float, int)) or not isinstance(imag, (float, int)):
                    raise TypeError("complex value components must be numbers or null")
                row_values.append(complex(float(real), float(imag)))
            point_rows.append(row_values)
        points.append(point_rows)

    return np.asarray(points, dtype=np.complex128)


def _diag_codes(diags_by_point: tuple[tuple[object, ...], ...]) -> list[list[str]]:
    return [[diag.code for diag in point] for point in diags_by_point]


def _assert_hash_approved(
    fixture_path: Path, fixture: dict[str, object], approved: dict[str, str]
) -> None:
    rel_path = fixture_path.relative_to(_repo_root()).as_posix()
    assert rel_path in approved
    assert _canonical_sha256(fixture) == approved[rel_path]


def _assert_hash_lock_fixture_set_parity(
    *, fixture_paths: tuple[Path, ...], approved: dict[str, str]
) -> None:
    expected_paths = {path.relative_to(_repo_root()).as_posix() for path in fixture_paths}
    assert set(approved) == expected_paths


def _build_ports(port_ids: list[str], *, scenario: str) -> tuple[PortBoundary, ...]:
    if scenario == "controlled_source_case":
        # Use reversed declaration order to lock canonical port sorting behavior.
        return (
            PortBoundary(port_id=port_ids[1], p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id=port_ids[0], p_plus_index=0, p_minus_index=None),
        )
    if len(port_ids) == 1:
        return (PortBoundary(port_id=port_ids[0], p_plus_index=0, p_minus_index=None),)
    return (
        PortBoundary(port_id=port_ids[0], p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id=port_ids[1], p_plus_index=1, p_minus_index=None),
    )


def _assemble_from_elements(
    *,
    node_ids: tuple[str, ...],
    aux_ids: tuple[str, ...],
    fail_indices: tuple[int, ...],
    element_factory: Callable[[UnknownIndexing], tuple[ElementStamp, ...]],
) -> Callable[[int, float], tuple[csc_matrix, np.ndarray]]:
    indexing = build_unknown_indexing(node_ids=node_ids, reference_node="0", aux_ids=aux_ids)
    elements = element_factory(indexing)
    pattern = compile_pattern(
        indexing.total_unknowns,
        elements,
        StampContext(omega_rad_s=0.0, resolved_params={}),
    )
    fail_set = set(fail_indices)

    def _assemble(point_index: int, frequency_hz: float) -> tuple[csc_matrix, np.ndarray]:
        if point_index in fail_set:
            raise RuntimeError("fixture planned fail")
        ctx = StampContext(
            omega_rad_s=2.0 * np.pi * frequency_hz,
            resolved_params={},
            frequency_index=point_index,
        )
        filled = fill_numeric(pattern, elements, ctx)
        return (
            filled.A.tocsc(),
            np.asarray(filled.b, dtype=np.complex128),
        )

    return _assemble


def _assemble_for_scenario(scenario: str, *, fail_indices: tuple[int, ...]):
    if scenario == "rc_rlc_sanity":
        r_ohm = parse_scalar_number("50")
        c_f = parse_scalar_number("1e-9")
        l_h = parse_scalar_number("1e-3")
        return _assemble_from_elements(
            node_ids=("0", "n1"),
            aux_ids=("L1:i",),
            fail_indices=fail_indices,
            element_factory=lambda indexing: (
                ResistorStamp("R1", "n1", "0", r_ohm, indexing),
                CapacitorStamp("C1", "n1", "0", c_f, indexing),
                InductorStamp("L1", "n1", "0", "L1:i", l_h, indexing),
            ),
        )

    if scenario == "controlled_source_case":
        g1_s = parse_scalar_number("0.11")
        g2_s = parse_scalar_number("0.08")
        r12_ohm = parse_scalar_number("25")
        gm_s = parse_scalar_number("0.07")
        return _assemble_from_elements(
            node_ids=("0", "n1", "n2"),
            aux_ids=(),
            fail_indices=fail_indices,
            element_factory=lambda indexing: (
                ConductanceStamp("G1", "n1", "0", g1_s, indexing),
                ConductanceStamp("G2", "n2", "0", g2_s, indexing),
                ResistorStamp("R12", "n1", "n2", r12_ohm, indexing),
                VCCSStamp("GM1", "n2", "0", "n1", "0", gm_s, indexing),
            ),
        )

    if scenario == "two_port_yzs_consistency":
        g1_s = parse_scalar_number("0.016")
        g2_s = parse_scalar_number("0.026")
        r12_ohm = parse_scalar_number("250")
        c1_f = parse_scalar_number("1e-10")
        c2_f = parse_scalar_number("0.5e-10")
        return _assemble_from_elements(
            node_ids=("0", "n1", "n2"),
            aux_ids=(),
            fail_indices=fail_indices,
            element_factory=lambda indexing: (
                ConductanceStamp("G1", "n1", "0", g1_s, indexing),
                ConductanceStamp("G2", "n2", "0", g2_s, indexing),
                ResistorStamp("R12", "n1", "n2", r12_ohm, indexing),
                CapacitorStamp("C1", "n1", "0", c1_f, indexing),
                CapacitorStamp("C2", "n2", "0", c2_f, indexing),
            ),
        )

    if scenario == "failed_point_sentinel":
        g_s = parse_scalar_number("0.5")
        return _assemble_from_elements(
            node_ids=("0", "n1"),
            aux_ids=(),
            fail_indices=fail_indices,
            element_factory=lambda indexing: (ConductanceStamp("G1", "n1", "0", g_s, indexing),),
        )

    raise ValueError(f"unsupported scenario: {scenario}")


def _assert_metric_match(
    *,
    expected: dict[str, object],
    result: _RFResultLike,
    expected_key: str,
    result_attr: str,
    tolerance: dict[str, float],
) -> None:
    expected_matrix = _decode_complex_cube(expected[expected_key])
    actual_matrix = np.asarray(getattr(result, result_attr), dtype=np.complex128)
    np.testing.assert_allclose(
        actual_matrix,
        expected_matrix,
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
        equal_nan=True,
    )
    expected_status = cast(list[str], cast(dict[str, object], expected["status"])[expected_key])
    expected_diag_codes = cast(
        list[list[str]],
        cast(dict[str, object], expected["diagnostic_codes"])[expected_key],
    )
    assert list(result.status.astype(str)) == expected_status
    assert _diag_codes(result.diagnostics_by_point) == expected_diag_codes


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda path: path.stem)
def test_regression_goldens_are_stable_and_tolerance_enforced(fixture_path: Path) -> None:
    fixture = _load_json_object(fixture_path)
    schema = _load_fixture_schema()
    approved_hashes = _load_hash_lock()
    fixture_paths = _fixture_paths()
    tolerance_table = _load_tolerance_table()
    _assert_hash_lock_fixture_set_parity(fixture_paths=fixture_paths, approved=approved_hashes)
    schema_errors = validate_json_against_schema(fixture, schema)
    assert schema_errors == ()
    _assert_hash_approved(fixture_path, fixture, approved_hashes)

    profile_name = cast(str, fixture["tolerance_profile"])
    assert profile_name in tolerance_table
    tolerance = tolerance_table[profile_name]
    frequencies = np.asarray(cast(list[float], fixture["frequencies_hz"]), dtype=np.float64)
    scenario = cast(str, fixture["scenario"])
    ports = _build_ports(cast(list[str], fixture["ports"]), scenario=scenario)
    expected = cast(dict[str, object], fixture["expected"])
    fail_indices = tuple(cast(list[int], expected.get("fail_point_indices", [])))
    assemble = _assemble_for_scenario(scenario, fail_indices=fail_indices)

    y_result = extract_y_parameters(frequencies, ports, assemble)
    z_result = extract_z_parameters(frequencies, ports, assemble)
    s_from_z_result = convert_z_to_s(z_result, z0_ohm=50.0)
    assert list(y_result.port_ids) == cast(list[str], fixture["ports"])
    assert list(z_result.port_ids) == cast(list[str], fixture["ports"])
    assert list(s_from_z_result.port_ids) == cast(list[str], fixture["ports"])

    _assert_metric_match(
        expected=expected,
        result=y_result,
        expected_key="y",
        result_attr="y",
        tolerance=tolerance,
    )
    _assert_metric_match(
        expected=expected,
        result=z_result,
        expected_key="z",
        result_attr="z",
        tolerance=tolerance,
    )
    _assert_metric_match(
        expected=expected,
        result=s_from_z_result,
        expected_key="s",
        result_attr="s",
        tolerance=tolerance,
    )

    if "s_from_y" in expected:
        s_from_y_result = convert_y_to_s(y_result, z0_ohm=50.0)
        _assert_metric_match(
            expected=expected,
            result=s_from_y_result,
            expected_key="s_from_y",
            result_attr="s",
            tolerance=tolerance,
        )

    for fail_index in fail_indices:
        assert np.isnan(y_result.y[fail_index].real).all()
        assert np.isnan(y_result.y[fail_index].imag).all()
        assert np.isnan(z_result.z[fail_index].real).all()
        assert np.isnan(z_result.z[fail_index].imag).all()
        assert np.isnan(s_from_z_result.s[fail_index].real).all()
        assert np.isnan(s_from_z_result.s[fail_index].imag).all()


def test_regression_tolerance_exceedance_raises_assertion() -> None:
    tolerance_table = _load_tolerance_table()
    tolerance = tolerance_table["rf_matrix_tight"]
    expected = np.asarray([[[1.0 + 0.0j]]], dtype=np.complex128)
    actual = np.asarray([[[1.0 + (10.0 * tolerance["atol"]) + 0.0j]]], dtype=np.complex128)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            equal_nan=True,
        )


def test_regression_hash_mismatch_detection_is_explicit() -> None:
    approved = _load_hash_lock()
    fixture_path = _fixture_paths()[0]
    fixture = _load_json_object(fixture_path)
    mutated = copy.deepcopy(fixture)
    mutated["fixture_id"] = f"{fixture['fixture_id']}-mutated"
    with pytest.raises(AssertionError):
        _assert_hash_approved(fixture_path, mutated, approved)


def test_golden_hash_updates_require_explicit_approve_flag() -> None:
    completed = subprocess.run(
        [sys.executable, _APPROVAL_SCRIPT_PATH],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--approve" in (completed.stdout + completed.stderr)


def test_regression_tolerance_source_is_classified_normative_merge_gating() -> None:
    classification = _load_json_object(_repo_root() / _CLASSIFICATION_PATH)
    entries_raw = classification.get("entries")
    merge_gating_raw = classification.get("merge_gating_tolerance_sources")
    assert isinstance(entries_raw, list)
    assert isinstance(merge_gating_raw, list)

    entry_map: dict[str, str] = {}
    for entry in entries_raw:
        assert isinstance(entry, dict)
        path = entry.get("path")
        state = entry.get("classification")
        assert isinstance(path, str) and isinstance(state, str)
        entry_map[path] = state

    assert entry_map[_REGRESSION_TOLERANCE_SOURCE] == "normative_gating"
    assert _REGRESSION_TOLERANCE_SOURCE in merge_gating_raw


def test_partial_null_complex_pair_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"\[null, null\]"):
        _decode_complex_cube([[[[None, 1.0]]]])


def test_fixture_schema_validation_rejects_missing_required_key() -> None:
    schema = _load_fixture_schema()
    fixture = _load_json_object(_fixture_paths()[0])
    fixture.pop("scenario", None)
    schema_errors = validate_json_against_schema(fixture, schema)
    assert any("missing required key 'scenario'" in error for error in schema_errors)


def test_fixture_schema_validation_rejects_malformed_expected_payload() -> None:
    schema = _load_fixture_schema()
    fixture = copy.deepcopy(_load_json_object(_fixture_paths()[0]))
    expected = cast(dict[str, object], fixture["expected"])
    expected["unexpected_key"] = 1
    status = cast(dict[str, object], expected["status"])
    status["y"] = "pass"
    schema_errors = validate_json_against_schema(fixture, schema)
    assert any("additional key not allowed: 'unexpected_key'" in error for error in schema_errors)
    assert any(
        "$.expected.status.y: expected type in ('array',)" in error for error in schema_errors
    )

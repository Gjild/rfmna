from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.governance.regression_fixture_schema import (
    load_json_mapping,
    validate_fixture_schema_document,
    validate_json_against_schema,
)
from rfmna.rf_metrics import PortBoundary, SParameterResult, YParameterResult, ZParameterResult
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

pytestmark = pytest.mark.cross_check

_FIXTURE_DIR_PATH = "tests/fixtures/cross_check"
_FIXTURE_HASH_LOCK_PATH = "tests/fixtures/cross_check/approved_hashes_v1.json"
_FIXTURE_SCHEMA_PATH = "tests/cross_check/schemas/rf_cross_check_fixture_v1.schema.json"
_TOLERANCE_PATH = "docs/dev/tolerances/regression_baseline_v1.yaml"
_CLASSIFICATION_PATH = "docs/dev/threshold_tolerance_classification.yaml"
_CALIBRATION_SEED_SOURCE = "docs/dev/tolerances/calibration_seed_v1.yaml"
_APPROVED_PROFILE_MAP = {
    "y": "rf_matrix_tight",
    "z": "rf_matrix_tight",
    "s": "rf_matrix_loose",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_fixture_paths() -> tuple[Path, ...]:
    fixture_dir = _repo_root() / _FIXTURE_DIR_PATH
    paths = tuple(
        path
        for path in sorted(fixture_dir.glob("*.json"))
        if path.name != Path(_FIXTURE_HASH_LOCK_PATH).name
    )
    assert paths, "expected at least one cross-check fixture"
    return paths


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_hash_lock() -> dict[str, str]:
    payload = load_json_mapping(_repo_root() / _FIXTURE_HASH_LOCK_PATH)
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


def _assert_hash_lock_fixture_set_parity(
    *, fixture_paths: tuple[Path, ...], approved_hashes: dict[str, str]
) -> None:
    expected = {path.relative_to(_repo_root()).as_posix() for path in fixture_paths}
    assert set(approved_hashes) == expected


def _assert_hash_approved(
    *, fixture_path: Path, fixture_payload: dict[str, object], approved_hashes: dict[str, str]
) -> None:
    rel_path = fixture_path.relative_to(_repo_root()).as_posix()
    assert rel_path in approved_hashes
    assert _canonical_sha256(fixture_payload) == approved_hashes[rel_path]


def _load_fixture_schema() -> dict[str, object]:
    payload = load_json_mapping(_repo_root() / _FIXTURE_SCHEMA_PATH)
    schema_errors = validate_fixture_schema_document(payload)
    assert schema_errors == ()
    return payload


def _load_tolerance_table() -> dict[str, dict[str, float]]:
    payload = load_json_mapping(_repo_root() / _TOLERANCE_PATH)
    assert payload["schema_version"] == 1
    assert payload["artifact_id"] == "regression_baseline_v1"
    assert payload["status"] == "normative_gating"
    raw_profiles = payload.get("profiles")
    assert isinstance(raw_profiles, dict)
    profiles: dict[str, dict[str, float]] = {}
    for profile_name, raw_profile in raw_profiles.items():
        assert isinstance(profile_name, str) and profile_name
        assert isinstance(raw_profile, dict)
        rtol = raw_profile.get("rtol")
        atol = raw_profile.get("atol")
        assert isinstance(rtol, (float, int))
        assert isinstance(atol, (float, int))
        profiles[profile_name] = {"rtol": float(rtol), "atol": float(atol)}
    return profiles


def _load_classification() -> tuple[dict[str, str], tuple[str, ...]]:
    payload = load_json_mapping(_repo_root() / _CLASSIFICATION_PATH)
    raw_entries = payload.get("entries")
    raw_merge_gating = payload.get("merge_gating_tolerance_sources")
    assert isinstance(raw_entries, list)
    assert isinstance(raw_merge_gating, list)

    entries: dict[str, str] = {}
    for raw_entry in raw_entries:
        assert isinstance(raw_entry, dict)
        path = raw_entry.get("path")
        classification = raw_entry.get("classification")
        assert isinstance(path, str) and path
        assert isinstance(classification, str) and classification
        entries[path] = classification
    merge_gating_sources = tuple(cast(str, source) for source in raw_merge_gating)
    return entries, merge_gating_sources


def _get_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    assert isinstance(value, (float, int))
    return float(value)


def _fixture_ports(fixture: dict[str, object]) -> tuple[str, ...]:
    raw_ports = fixture.get("ports")
    assert isinstance(raw_ports, list) and raw_ports
    ports: list[str] = []
    for raw_port in raw_ports:
        assert isinstance(raw_port, str) and raw_port
        ports.append(raw_port)
    return tuple(ports)


def _fixture_frequencies(fixture: dict[str, object]) -> np.ndarray:
    raw_frequencies = fixture.get("frequencies_hz")
    assert isinstance(raw_frequencies, list) and raw_frequencies
    frequencies = np.asarray(raw_frequencies, dtype=np.float64)
    assert frequencies.ndim == 1
    assert np.isfinite(frequencies).all()
    return frequencies


def _fixture_tolerance_profiles(fixture: dict[str, object]) -> dict[str, str]:
    raw_profiles = fixture.get("tolerance_profiles")
    assert isinstance(raw_profiles, dict)
    profiles: dict[str, str] = {}
    for metric in ("y", "z", "s"):
        profile = raw_profiles.get(metric)
        assert isinstance(profile, str) and profile
        profiles[metric] = profile
    return profiles


def _fixture_tolerance_source(fixture: dict[str, object]) -> str:
    source = fixture.get("tolerance_source")
    assert isinstance(source, str) and source
    return source


def _fixture_z0_ohm(fixture: dict[str, object]) -> float:
    z0_ohm = fixture.get("z0_ohm")
    assert isinstance(z0_ohm, (float, int))
    return float(z0_ohm)


def _fixture_scenario(fixture: dict[str, object]) -> str:
    scenario = fixture.get("scenario")
    assert isinstance(scenario, str) and scenario
    return scenario


def _fixture_parameters(fixture: dict[str, object]) -> dict[str, object]:
    parameters = fixture.get("parameters")
    assert isinstance(parameters, dict)
    return cast(dict[str, object], parameters)


def _analytic_y_point(
    *,
    scenario: str,
    parameters: dict[str, object],
    frequency_hz: float,
) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_hz

    if scenario == "one_port_shunt_rc":
        g_s = _get_float(parameters, "g_s")
        c_f = _get_float(parameters, "c_f")
        y11 = g_s + (1j * omega * c_f)
        return np.asarray([[y11]], dtype=np.complex128)

    if scenario == "two_port_coupled_rc":
        g1_s = _get_float(parameters, "g1_s")
        g2_s = _get_float(parameters, "g2_s")
        g12_s = _get_float(parameters, "g12_s")
        c1_f = _get_float(parameters, "c1_f")
        c2_f = _get_float(parameters, "c2_f")
        c12_f = _get_float(parameters, "c12_f")
        y11 = (g1_s + g12_s) + (1j * omega * (c1_f + c12_f))
        y22 = (g2_s + g12_s) + (1j * omega * (c2_f + c12_f))
        y12 = -(g12_s + (1j * omega * c12_f))
        return np.asarray(((y11, y12), (y12, y22)), dtype=np.complex128)

    raise AssertionError(f"unsupported cross-check scenario: {scenario}")


def _analytic_y_cube(
    *,
    scenario: str,
    parameters: dict[str, object],
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    points = [
        _analytic_y_point(
            scenario=scenario,
            parameters=parameters,
            frequency_hz=float(frequency_hz),
        )
        for frequency_hz in frequencies_hz
    ]
    return np.asarray(points, dtype=np.complex128)


def _analytic_z_from_y(y_values: np.ndarray) -> np.ndarray:
    z_points = [np.linalg.inv(y_point) for y_point in y_values]
    return np.asarray(z_points, dtype=np.complex128)


def _analytic_s_from_z(*, z_values: np.ndarray, z0_ohm: float) -> np.ndarray:
    n_ports = int(z_values.shape[1])
    identity = np.eye(n_ports, dtype=np.complex128)
    z0_matrix = z0_ohm * identity
    s_points: list[np.ndarray] = []
    for z_point in z_values:
        numerator = z_point - z0_matrix
        denominator = z_point + z0_matrix
        s_point = np.linalg.solve(denominator.T, numerator.T).T
        s_points.append(np.asarray(s_point, dtype=np.complex128))
    return np.asarray(s_points, dtype=np.complex128)


def _build_assemble_point(
    *,
    scenario: str,
    parameters: dict[str, object],
    n_ports: int,
) -> Callable[[int, float], tuple[csc_matrix, np.ndarray]]:
    def _assemble(_: int, frequency_hz: float) -> tuple[csc_matrix, np.ndarray]:
        y_point = _analytic_y_point(
            scenario=scenario,
            parameters=parameters,
            frequency_hz=frequency_hz,
        )
        rhs = np.zeros(n_ports, dtype=np.complex128)
        return csc_matrix(y_point), rhs

    return _assemble


def _assert_matrix_match(
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: dict[str, float],
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.complex128),
        np.asarray(expected, dtype=np.complex128),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
        equal_nan=True,
    )


@pytest.mark.parametrize("fixture_path", _load_fixture_paths(), ids=lambda path: path.stem)
def test_cross_check_fixture_matches_analytic_reference(fixture_path: Path) -> None:
    fixture = load_json_mapping(fixture_path)
    schema = _load_fixture_schema()
    schema_errors = validate_json_against_schema(fixture, schema)
    assert schema_errors == ()

    fixture_paths = _load_fixture_paths()
    approved_hashes = _load_hash_lock()
    _assert_hash_lock_fixture_set_parity(
        fixture_paths=fixture_paths, approved_hashes=approved_hashes
    )
    _assert_hash_approved(
        fixture_path=fixture_path,
        fixture_payload=fixture,
        approved_hashes=approved_hashes,
    )

    ports = _fixture_ports(fixture)
    frequencies_hz = _fixture_frequencies(fixture)
    scenario = _fixture_scenario(fixture)
    parameters = _fixture_parameters(fixture)
    z0_ohm = _fixture_z0_ohm(fixture)
    tolerance_profiles = _fixture_tolerance_profiles(fixture)
    tolerance_table = _load_tolerance_table()

    port_boundaries = tuple(
        PortBoundary(port_id=port_id, p_plus_index=port_index, p_minus_index=None)
        for port_index, port_id in enumerate(ports)
    )
    assemble_point = _build_assemble_point(
        scenario=scenario,
        parameters=parameters,
        n_ports=len(ports),
    )
    sweep_result = run_sweep(
        frequencies_hz,
        SweepLayout(n_nodes=len(ports), n_aux=0),
        assemble_point,
        rf_request=SweepRFRequest(ports=port_boundaries, metrics=("y", "z", "s")),
    )
    assert sweep_result.status.tolist() == ["pass"] * int(frequencies_hz.shape[0])
    assert sweep_result.rf_payloads is not None

    y_payload = sweep_result.rf_payloads.get("y")
    z_payload = sweep_result.rf_payloads.get("z")
    s_payload = sweep_result.rf_payloads.get("s")
    assert isinstance(y_payload, YParameterResult)
    assert isinstance(z_payload, ZParameterResult)
    assert isinstance(s_payload, SParameterResult)
    assert y_payload.status.tolist() == ["pass"] * int(frequencies_hz.shape[0])
    assert z_payload.status.tolist() == ["pass"] * int(frequencies_hz.shape[0])
    assert s_payload.status.tolist() == ["pass"] * int(frequencies_hz.shape[0])

    expected_y = _analytic_y_cube(
        scenario=scenario,
        parameters=parameters,
        frequencies_hz=frequencies_hz,
    )
    expected_z = _analytic_z_from_y(expected_y)
    expected_s = _analytic_s_from_z(z_values=expected_z, z0_ohm=z0_ohm)

    _assert_matrix_match(
        actual=y_payload.y,
        expected=expected_y,
        tolerance=tolerance_table[tolerance_profiles["y"]],
    )
    _assert_matrix_match(
        actual=z_payload.z,
        expected=expected_z,
        tolerance=tolerance_table[tolerance_profiles["z"]],
    )
    _assert_matrix_match(
        actual=s_payload.s,
        expected=expected_s,
        tolerance=tolerance_table[tolerance_profiles["s"]],
    )


def test_cross_check_tolerance_exceedance_is_explicit() -> None:
    tolerance = _load_tolerance_table()["rf_matrix_tight"]
    expected = np.asarray([[[1.0 + 0.0j]]], dtype=np.complex128)
    actual = np.asarray([[[1.0 + (10.0 * tolerance["atol"]) + 0.0j]]], dtype=np.complex128)
    with pytest.raises(AssertionError):
        _assert_matrix_match(actual=actual, expected=expected, tolerance=tolerance)


def test_cross_check_variability_profile_policy_is_enforced() -> None:
    tolerance_table = _load_tolerance_table()
    assert tolerance_table["rf_matrix_loose"]["rtol"] >= tolerance_table["rf_matrix_tight"]["rtol"]
    assert tolerance_table["rf_matrix_loose"]["atol"] >= tolerance_table["rf_matrix_tight"]["atol"]

    for fixture_path in _load_fixture_paths():
        fixture = load_json_mapping(fixture_path)
        assert _fixture_tolerance_profiles(fixture) == _APPROVED_PROFILE_MAP


def test_cross_check_gating_sources_are_normative_and_not_calibration_only() -> None:
    classification_entries, merge_gating_sources = _load_classification()
    calibration_classification = classification_entries.get(_CALIBRATION_SEED_SOURCE)
    assert calibration_classification == "calibration_only"

    fixture_sources = {
        _fixture_tolerance_source(load_json_mapping(path)) for path in _load_fixture_paths()
    }
    assert _CALIBRATION_SEED_SOURCE not in fixture_sources

    for source in fixture_sources:
        assert classification_entries.get(source) == "normative_gating"
        assert source in merge_gating_sources
        source_payload = load_json_mapping(_repo_root() / source)
        assert source_payload["status"] == "normative_gating"


def test_cross_check_marker_strictness_contract_has_no_p2_00_drift() -> None:
    pytest_ini = (_repo_root() / "pytest.ini").read_text(encoding="utf-8")
    assert "--strict-markers" in pytest_ini
    marker_lines = [line.strip() for line in pytest_ini.splitlines()]
    assert "cross_check" in marker_lines

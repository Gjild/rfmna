from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.governance.regression_fixture_schema import load_json_mapping
from rfmna.rf_metrics import PortBoundary, SParameterResult, YParameterResult, ZParameterResult
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep

_FIXTURE_DIR_PATH = "tests/fixtures/cross_check"
_HASH_LOCK_FILE = "approved_hashes_v1.json"
_TOLERANCE_PATH = "docs/dev/tolerances/regression_baseline_v1.yaml"


def _fixture_paths(repo_root: Path) -> tuple[Path, ...]:
    fixture_dir = repo_root / _FIXTURE_DIR_PATH
    paths = tuple(
        path for path in sorted(fixture_dir.glob("*.json")) if path.name != _HASH_LOCK_FILE
    )
    if not paths:
        raise ValueError("expected at least one cross-check fixture")
    return paths


def _load_tolerance_table(repo_root: Path) -> dict[str, dict[str, float]]:
    payload = load_json_mapping(repo_root / _TOLERANCE_PATH)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("tolerance profiles must be a mapping")

    profiles: dict[str, dict[str, float]] = {}
    for profile_name, raw_profile in raw_profiles.items():
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError("tolerance profile names must be non-empty strings")
        if not isinstance(raw_profile, dict):
            raise ValueError("tolerance profile payload must be a mapping")
        rtol = raw_profile.get("rtol")
        atol = raw_profile.get("atol")
        if not isinstance(rtol, (float, int)) or not isinstance(atol, (float, int)):
            raise ValueError("tolerance profile values must include numeric rtol/atol")
        profiles[profile_name] = {"rtol": float(rtol), "atol": float(atol)}
    return profiles


def _get_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (float, int)):
        raise ValueError(f"missing numeric parameter: {key}")
    return float(value)


def _scenario(fixture: dict[str, object]) -> str:
    scenario = fixture.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario must be a non-empty string")
    return scenario


def _parameters(fixture: dict[str, object]) -> dict[str, object]:
    parameters = fixture.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping")
    return cast(dict[str, object], parameters)


def _frequencies_hz(fixture: dict[str, object]) -> np.ndarray:
    raw_frequencies = fixture.get("frequencies_hz")
    if not isinstance(raw_frequencies, list) or not raw_frequencies:
        raise ValueError("frequencies_hz must be a non-empty list")
    frequencies = np.asarray(raw_frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be a finite one-dimensional numeric array")
    return frequencies


def _ports(fixture: dict[str, object]) -> tuple[str, ...]:
    raw_ports = fixture.get("ports")
    if not isinstance(raw_ports, list) or not raw_ports:
        raise ValueError("ports must be a non-empty list")
    ports = tuple(raw_port for raw_port in raw_ports if isinstance(raw_port, str) and raw_port)
    if len(ports) != len(raw_ports):
        raise ValueError("ports entries must be non-empty strings")
    return ports


def _z0_ohm(fixture: dict[str, object]) -> float:
    z0_ohm = fixture.get("z0_ohm")
    if not isinstance(z0_ohm, (float, int)):
        raise ValueError("z0_ohm must be numeric")
    return float(z0_ohm)


def _tolerance_profiles(fixture: dict[str, object]) -> dict[str, str]:
    raw_profiles = fixture.get("tolerance_profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("tolerance_profiles must be a mapping")
    profiles: dict[str, str] = {}
    for metric in ("y", "z", "s"):
        profile = raw_profiles.get(metric)
        if not isinstance(profile, str) or not profile:
            raise ValueError(f"missing tolerance profile for metric: {metric}")
        profiles[metric] = profile
    return profiles


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
        return np.asarray([[g_s + (1j * omega * c_f)]], dtype=np.complex128)
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
    raise ValueError(f"unsupported scenario: {scenario}")


def _analytic_y_cube(
    *,
    scenario: str,
    parameters: dict[str, object],
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _analytic_y_point(
                scenario=scenario,
                parameters=parameters,
                frequency_hz=float(frequency_hz),
            )
            for frequency_hz in frequencies_hz
        ],
        dtype=np.complex128,
    )


def _analytic_z_from_y(y_values: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.inv(point) for point in y_values], dtype=np.complex128)


def _analytic_s_from_z(*, z_values: np.ndarray, z0_ohm: float) -> np.ndarray:
    n_ports = int(z_values.shape[1])
    identity = np.eye(n_ports, dtype=np.complex128)
    z0_matrix = z0_ohm * identity
    points: list[np.ndarray] = []
    for z_point in z_values:
        numerator = z_point - z0_matrix
        denominator = z_point + z0_matrix
        points.append(np.linalg.solve(denominator.T, numerator.T).T)
    return np.asarray(points, dtype=np.complex128)


def _assemble_point(
    *,
    scenario: str,
    parameters: dict[str, object],
    n_ports: int,
):
    def _assemble(_: int, frequency_hz: float) -> tuple[csc_matrix, np.ndarray]:
        y_point = _analytic_y_point(
            scenario=scenario,
            parameters=parameters,
            frequency_hz=frequency_hz,
        )
        return csc_matrix(y_point), np.zeros(n_ports, dtype=np.complex128)

    return _assemble


def _max_abs_error(*, actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)))


def _max_rel_error(*, actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(actual - expected) / denominator))


def _calibrate_fixture(
    *,
    fixture_path: Path,
    tolerance_table: dict[str, dict[str, float]],
) -> dict[str, object]:
    fixture = load_json_mapping(fixture_path)
    scenario = _scenario(fixture)
    parameters = _parameters(fixture)
    frequencies_hz = _frequencies_hz(fixture)
    ports = _ports(fixture)
    z0_ohm = _z0_ohm(fixture)
    tolerance_profiles = _tolerance_profiles(fixture)

    expected_y = _analytic_y_cube(
        scenario=scenario,
        parameters=parameters,
        frequencies_hz=frequencies_hz,
    )
    expected_z = _analytic_z_from_y(expected_y)
    expected_s = _analytic_s_from_z(z_values=expected_z, z0_ohm=z0_ohm)

    port_boundaries = tuple(
        PortBoundary(port_id=port_id, p_plus_index=index, p_minus_index=None)
        for index, port_id in enumerate(ports)
    )
    sweep_result = run_sweep(
        frequencies_hz,
        SweepLayout(n_nodes=len(ports), n_aux=0),
        _assemble_point(scenario=scenario, parameters=parameters, n_ports=len(ports)),
        rf_request=SweepRFRequest(ports=port_boundaries, metrics=("y", "z", "s")),
    )
    if sweep_result.rf_payloads is None:
        raise ValueError("cross-check sweep did not produce RF payloads")

    y_payload = sweep_result.rf_payloads.get("y")
    z_payload = sweep_result.rf_payloads.get("z")
    s_payload = sweep_result.rf_payloads.get("s")
    if not isinstance(y_payload, YParameterResult):
        raise ValueError("missing Y payload")
    if not isinstance(z_payload, ZParameterResult):
        raise ValueError("missing Z payload")
    if not isinstance(s_payload, SParameterResult):
        raise ValueError("missing S payload")

    max_abs_error = {
        "y": _max_abs_error(actual=y_payload.y, expected=expected_y),
        "z": _max_abs_error(actual=z_payload.z, expected=expected_z),
        "s": _max_abs_error(actual=s_payload.s, expected=expected_s),
    }
    max_rel_error = {
        "y": _max_rel_error(actual=y_payload.y, expected=expected_y),
        "z": _max_rel_error(actual=z_payload.z, expected=expected_z),
        "s": _max_rel_error(actual=s_payload.s, expected=expected_s),
    }

    within_profile: dict[str, bool] = {}
    for metric in ("y", "z", "s"):
        profile = tolerance_profiles[metric]
        tolerance = tolerance_table[profile]
        within_profile[metric] = (
            max_abs_error[metric] <= tolerance["atol"]
            and max_rel_error[metric] <= tolerance["rtol"]
        )

    return {
        "fixture_id": fixture["fixture_id"],
        "scenario": scenario,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "tolerance_profiles": tolerance_profiles,
        "within_profile": within_profile,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate deterministic cross-check tolerances.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    tolerance_table = _load_tolerance_table(repo_root)
    fixture_results = [
        _calibrate_fixture(fixture_path=path, tolerance_table=tolerance_table)
        for path in _fixture_paths(repo_root)
    ]
    max_by_metric = {
        metric: max(float(result["max_abs_error"][metric]) for result in fixture_results)
        for metric in ("y", "z", "s")
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_glob": "tests/fixtures/cross_check/*.json",
        "tolerance_source_path": _TOLERANCE_PATH,
        "fixture_results": fixture_results,
        "max_abs_error_by_metric": max_by_metric,
        "recommended_profiles": {
            "y": "rf_matrix_tight",
            "z": "rf_matrix_tight",
            "s": "rf_matrix_loose",
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output:
        (repo_root / args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

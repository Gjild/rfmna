from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from rfmna.cli import main as cli_main
from rfmna.parser import PreflightInput
from rfmna.sweep_engine import SweepLayout, SweepResult
from rfmna.viz_io import (
    ManifestError,
    RunManifest,
    build_manifest,
    stable_manifest_hash,
    stable_projection,
    to_canonical_dict,
    to_canonical_json,
)
from rfmna.viz_io.manifest import CORE_DEPENDENCIES, THREAD_ENV_KEYS

pytestmark = pytest.mark.unit

runner = CliRunner()


def _build_minimal_manifest(
    *,
    timestamp: str = "2026-02-08T00:00:00+00:00",
    timezone: str = "UTC",
    solver_config_snapshot: dict[str, object] | None = None,
) -> RunManifest:
    return build_manifest(
        input_payload={"design": {"name": "amp", "nodes": ["n1", "0"]}},
        resolved_params_payload={"r": 50.0, "c": 1.0e-12},
        solver_config_snapshot=solver_config_snapshot or {"solver": {"gmin": [0.0, 1.0e-12]}},
        frequency_grid_metadata={"n_points": 3, "f_start_hz": 1.0, "f_stop_hz": 10.0},
        tool_version="0.1.0",
        git_commit_sha=None,
        source_hash_fallback=None,
        dependency_versions={dep: None for dep in CORE_DEPENDENCIES},
        platform_fingerprint={
            "system": "Linux",
            "release": "test",
            "version": "test",
            "machine": "x86_64",
            "processor": None,
            "python_implementation": "CPython",
        },
        thread_runtime_fingerprint={key: None for key in THREAD_ENV_KEYS},
        numeric_backend_fingerprint={
            "numpy_version": "2.4.2",
            "scipy_version": "1.17.0",
            "numpy_blas_opt_info": None,
            "scipy_blas_opt_info": None,
        },
        timestamp=timestamp,
        timezone=timezone,
    )


def _dummy_sweep_result(statuses: Sequence[str]) -> SweepResult:
    n_points = len(statuses)
    return SweepResult(
        n_points=n_points,
        n_nodes=1,
        n_aux=0,
        V_nodes=np.zeros((n_points, 1), dtype=np.complex128),
        I_aux=np.zeros((n_points, 0), dtype=np.complex128),
        res_l2=np.zeros(n_points, dtype=np.float64),
        res_linf=np.zeros(n_points, dtype=np.float64),
        res_rel=np.zeros(n_points, dtype=np.float64),
        cond_ind=np.ones(n_points, dtype=np.float64),
        status=np.asarray(tuple(statuses), dtype=np.dtype("<U8")),
        diagnostics_by_point=tuple(() for _ in range(n_points)),
    )


def test_schema_presence_and_placeholders() -> None:
    manifest = _build_minimal_manifest()
    canonical = to_canonical_dict(manifest)
    required = {
        "tool_version",
        "git_commit_sha",
        "source_hash_fallback",
        "python_version",
        "dependency_versions",
        "platform",
        "input_hash",
        "resolved_params_hash",
        "timestamp",
        "timezone",
        "solver_config_snapshot",
        "thread_runtime_fingerprint",
        "numeric_backend_fingerprint",
        "frequency_grid_metadata",
    }
    assert required.issubset(set(canonical))
    assert canonical["git_commit_sha"] is None
    assert canonical["source_hash_fallback"] is None


def test_canonical_serialization_determinism_and_nested_order_independence() -> None:
    left = _build_minimal_manifest(solver_config_snapshot={"z": {"b": 2, "a": 1}, "a": [3, 2, 1]})
    right = _build_minimal_manifest(solver_config_snapshot={"a": [3, 2, 1], "z": {"a": 1, "b": 2}})
    assert to_canonical_json(left) == to_canonical_json(right)


def test_hash_stability_for_identical_nonvolatile_content() -> None:
    first = _build_minimal_manifest()
    second = _build_minimal_manifest()
    assert first.input_hash == second.input_hash
    assert first.resolved_params_hash == second.resolved_params_hash
    assert stable_manifest_hash(first) == stable_manifest_hash(second)


def test_volatile_fields_excluded_from_stable_hash() -> None:
    first = _build_minimal_manifest(timestamp="2026-02-08T00:00:00+00:00", timezone="UTC")
    second = _build_minimal_manifest(timestamp="2026-02-08T00:01:00+00:00", timezone="PST")
    assert stable_manifest_hash(first) == stable_manifest_hash(second)
    assert stable_projection(first) == stable_projection(second)


def test_fingerprint_structure_has_deterministic_keys() -> None:
    manifest = build_manifest(
        input_payload={"x": 1},
        resolved_params_payload={"y": 2},
        solver_config_snapshot={},
        frequency_grid_metadata={},
        timestamp="2026-02-08T00:00:00+00:00",
        timezone="UTC",
    )
    canonical = to_canonical_dict(manifest)
    dependencies = canonical["dependency_versions"]
    threads = canonical["thread_runtime_fingerprint"]
    numeric = canonical["numeric_backend_fingerprint"]
    assert isinstance(dependencies, dict)
    assert tuple(sorted(dependencies)) == tuple(sorted(CORE_DEPENDENCIES))
    assert isinstance(threads, dict)
    assert tuple(sorted(threads)) == tuple(sorted(THREAD_ENV_KEYS))
    assert isinstance(numeric, dict)
    assert "numpy_version" in numeric
    assert "scipy_version" in numeric
    assert "numpy_blas_opt_info" in numeric
    assert "scipy_blas_opt_info" in numeric


def test_purity_inputs_not_mutated() -> None:
    input_payload = {"netlist": {"nodes": ["n1", "0"], "ports": {"p1": {"n+": "n1", "n-": "0"}}}}
    resolved = {"r": 50.0, "c": 1.0e-12}
    config = {"solver": {"gmin": [0.0, 1.0e-12]}}
    metadata = {"freq": {"n": 11, "fmin": 1.0, "fmax": 10.0}}
    input_before = copy.deepcopy(input_payload)
    resolved_before = copy.deepcopy(resolved)
    config_before = copy.deepcopy(config)
    metadata_before = copy.deepcopy(metadata)

    _ = build_manifest(
        input_payload=input_payload,
        resolved_params_payload=resolved,
        solver_config_snapshot=config,
        frequency_grid_metadata=metadata,
        timestamp="2026-02-08T00:00:00+00:00",
        timezone="UTC",
    )
    assert input_payload == input_before
    assert resolved == resolved_before
    assert config == config_before
    assert metadata == metadata_before


def test_invalid_required_payload_raises_typed_error() -> None:
    with pytest.raises(ManifestError) as exc:
        _ = build_manifest(
            input_payload={"bad": object()},
            resolved_params_payload={"ok": 1},
            solver_config_snapshot={},
            frequency_grid_metadata={},
            timestamp="2026-02-08T00:00:00+00:00",
            timezone="UTC",
        )
    assert exc.value.code == "E_MANIFEST_PAYLOAD_INVALID"


def test_run_path_emits_manifest_every_run_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, NDArray[np.complex128]]:
        del index, frequency_hz
        raise AssertionError("assemble_point should not be used in this test")

    bundle = cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=(1.0, 2.0),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _dummy_sweep_result(("pass", "pass")),
    )

    manifest_calls = {"build": 0, "attach": 0}
    original_attach = cli_main.attach_manifest_to_run_payload

    def _build_manifest(**kwargs: object) -> RunManifest:
        manifest_calls["build"] += 1
        return _build_minimal_manifest()

    def _attach(payload: object, manifest: RunManifest) -> object:
        manifest_calls["attach"] += 1
        return original_attach(payload, manifest)

    monkeypatch.setattr(cli_main, "build_manifest", _build_manifest)
    monkeypatch.setattr(cli_main, "attach_manifest_to_run_payload", _attach)

    first = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    second = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert manifest_calls == {"build": 2, "attach": 2}


def test_run_path_emits_solver_snapshot_with_explicit_default_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_input = PreflightInput(nodes=("0",), reference_node="0")

    def assemble_point(index: int, frequency_hz: float) -> tuple[object, NDArray[np.complex128]]:
        del index, frequency_hz
        raise AssertionError("assemble_point should not be used in this test")

    bundle = cli_main.CliDesignBundle(
        preflight_input=preflight_input,
        frequencies_hz=(1.0,),
        sweep_layout=SweepLayout(n_nodes=1, n_aux=0),
        assemble_point=assemble_point,
    )
    monkeypatch.setattr(cli_main, "_load_design_bundle", lambda design: bundle)
    monkeypatch.setattr(cli_main, "_execute_preflight", lambda _: ())
    monkeypatch.setattr(
        cli_main,
        "_execute_sweep",
        lambda freq, layout, assemble: _dummy_sweep_result(("pass",)),
    )

    captured: dict[str, object] = {}

    def _build_manifest(**kwargs: object) -> RunManifest:
        captured["solver_config_snapshot"] = kwargs["solver_config_snapshot"]
        return _build_minimal_manifest(
            solver_config_snapshot=kwargs["solver_config_snapshot"]  # type: ignore[arg-type]
        )

    monkeypatch.setattr(cli_main, "build_manifest", _build_manifest)

    result = runner.invoke(cli_main.app, ["run", "design.net", "--analysis", "ac"])
    assert result.exit_code == 0

    snapshot = captured["solver_config_snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["schema"] == "solver_repro_snapshot_v1"
    assert snapshot["analysis"] == "ac"
    assert snapshot["conversion_math_controls"] == {"enable_gmin_regularization": False}
    assert snapshot["attempt_trace_summary"]["total_solve_calls"] == 0
    assert snapshot["attempt_trace_summary"]["skip_reason_counts"] == {}

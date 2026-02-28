from __future__ import annotations

import pytest

from rfmna.viz_io import (
    build_manifest,
    stable_manifest_hash,
    to_canonical_dict,
    to_canonical_json,
)
from rfmna.viz_io.manifest import VOLATILE_STABLE_EXCLUDE_KEYS

pytestmark = pytest.mark.conformance


def _manifest(
    *, timestamp: str, timezone: str, solver_config_snapshot: dict[str, object]
) -> object:
    return build_manifest(
        input_payload={"design": {"a": 1, "b": [2, 3]}},
        resolved_params_payload={"param": {"x": 1.0, "y": 2.0}},
        solver_config_snapshot=solver_config_snapshot,
        frequency_grid_metadata={"n_points": 2, "f_start_hz": 1.0, "f_stop_hz": 2.0},
        timestamp=timestamp,
        timezone=timezone,
        dependency_versions={
            "numpy": None,
            "scipy": None,
            "pandas": None,
            "matplotlib": None,
            "pydantic": None,
            "typer": None,
        },
        thread_runtime_fingerprint={
            "PYTHONHASHSEED": None,
            "OPENBLAS_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OMP_NUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        numeric_backend_fingerprint={
            "numpy_version": "2.4.2",
            "scipy_version": "1.17.0",
            "numpy_blas_opt_info": None,
            "scipy_blas_opt_info": None,
        },
    )


def test_required_fields_present_canonical_deterministic_and_volatile_exclusion() -> None:
    left = _manifest(
        timestamp="2026-02-08T00:00:00+00:00",
        timezone="UTC",
        solver_config_snapshot={"outer": {"b": 2, "a": 1}},
    )
    right = _manifest(
        timestamp="2026-02-08T01:00:00+00:00",
        timezone="PST",
        solver_config_snapshot={"outer": {"a": 1, "b": 2}},
    )

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
    assert required.issubset(set(to_canonical_dict(left)))
    assert to_canonical_json(left) != ""
    assert stable_manifest_hash(left) == stable_manifest_hash(right)
    assert "timestamp" in VOLATILE_STABLE_EXCLUDE_KEYS
    assert "timezone" in VOLATILE_STABLE_EXCLUDE_KEYS

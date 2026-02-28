from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfmna.solver import FallbackRunConfig
from rfmna.solver.repro_snapshot import build_solver_config_snapshot

pytestmark = pytest.mark.conformance


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs/spec/schemas/solver_repro_snapshot_v1.json"


def test_solver_repro_snapshot_schema_artifact_has_required_contract_keys() -> None:
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    assert schema["$id"] == "docs/spec/schemas/solver_repro_snapshot_v1.json"
    assert schema["type"] == "object"
    required = set(schema["required"])
    assert required == {
        "schema",
        "retry_controls",
        "conversion_math_controls",
        "attempt_trace_summary",
    }


def test_solver_repro_snapshot_defaults_and_inactive_controls_are_explicit_and_deterministic() -> (
    None
):
    left = build_solver_config_snapshot()
    right = build_solver_config_snapshot()
    assert left == right

    inactive = build_solver_config_snapshot(
        run_config=FallbackRunConfig(
            enable_alt_pivot=False, enable_scaling=False, enable_gmin=False
        )
    )
    retry_controls = inactive["retry_controls"]
    summary = inactive["attempt_trace_summary"]
    assert retry_controls["enable_alt_pivot"] is False
    assert retry_controls["enable_scaling"] is False
    assert retry_controls["enable_gmin"] is False
    assert summary["total_solve_calls"] == 0
    assert summary["stage_run_counts"] == {
        "baseline": 0,
        "alt_pivot": 0,
        "scaling": 0,
        "gmin": 0,
        "final_fail": 0,
    }

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_regression_smoke_fixture_matches_schema_contract() -> None:
    root = _repo_root()
    fixture_path = root / "tests/regression/fixtures/smoke/resistor_y_smoke_v1.json"
    schema_path = root / "tests/regression/schemas/resistor_y_smoke_v1.schema.json"

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    required_keys = schema["required"]
    assert required_keys == ["schema_version", "fixture_id", "metric", "expected"]

    assert fixture["schema_version"] == 1
    assert fixture["fixture_id"] == "reg-smoke-resistor-y-v1"
    assert fixture["metric"] == "y11"
    assert isinstance(fixture["expected"], float)
    assert fixture["expected"] == pytest.approx(0.02)

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

pytestmark = pytest.mark.conformance

_EXPECTED_THREAD_DEFAULTS: dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ci_test_job() -> dict[str, object]:
    workflow_path = _repo_root() / ".github/workflows/ci.yml"
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    test_job = jobs.get("test")
    assert isinstance(test_job, dict)
    return test_job


def test_ci_workflow_declares_deterministic_thread_defaults() -> None:
    test_job = _ci_test_job()
    env_block = test_job.get("env")
    assert isinstance(env_block, dict)

    for key, expected in _EXPECTED_THREAD_DEFAULTS.items():
        assert str(env_block.get(key)) == expected


def test_envrc_declares_deterministic_thread_defaults() -> None:
    envrc_path = _repo_root() / ".envrc"
    lines = envrc_path.read_text(encoding="utf-8").splitlines()
    exports: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith("export "):
            continue
        key_value = stripped.removeprefix("export ")
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", maxsplit=1)
        exports[key] = value

    for key, expected in _EXPECTED_THREAD_DEFAULTS.items():
        assert exports.get(key) == expected


def test_ci_workflow_always_surfaces_phase1_gate_status() -> None:
    test_job = _ci_test_job()
    steps = test_job.get("steps")
    assert isinstance(steps, list)

    named_steps: dict[str, tuple[int, dict[str, object]]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if isinstance(name, str):
            named_steps[name] = (index, step)

    status_name = "Phase 1 gate status (informational)"
    upload_name = "Upload Phase 1 gate status (informational)"
    assert status_name in named_steps
    assert upload_name in named_steps

    status_index, status_step = named_steps[status_name]
    upload_index, upload_step = named_steps[upload_name]
    assert status_index < upload_index

    assert status_step.get("if") == "${{ always() }}"
    status_run = status_step.get("run")
    assert isinstance(status_run, str)
    assert "checklist_status=present" in status_run
    assert "checklist_status=missing" in status_run
    assert "phase1_gate_status.txt" in status_run
    assert "sed -n '1,220p' docs/dev/phase1_gate.md" in status_run
    assert "docs/dev/phase1_gate.md" in status_run

    assert upload_step.get("if") == "${{ always() }}"
    assert upload_step.get("uses") == "actions/upload-artifact@v4"
    with_block = upload_step.get("with")
    assert isinstance(with_block, dict)
    upload_path = with_block.get("path")
    assert isinstance(upload_path, str)
    assert "phase1_gate_status.txt" in upload_path
    assert "docs/dev/phase1_gate.md" in upload_path

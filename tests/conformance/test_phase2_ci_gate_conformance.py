from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from rfmna.governance.phase2_gate import evaluate_category_bootstrap_gate

pytestmark = pytest.mark.conformance


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ci_test_job() -> dict[str, object]:
    payload = yaml.safe_load(
        (_repo_root() / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    test_job = jobs.get("test")
    assert isinstance(test_job, dict)
    return test_job


def test_phase2_category_bootstrap_gate_passes_in_repo_baseline() -> None:
    result = evaluate_category_bootstrap_gate(repo_root=_repo_root())
    assert result.passed, result.errors


def test_phase2_category_bootstrap_gate_returns_structured_failure_when_required_files_missing(
    tmp_path: Path,
) -> None:
    result = evaluate_category_bootstrap_gate(repo_root=tmp_path)
    assert not result.passed
    assert result.sub_gate == "category-bootstrap"
    assert any("missing required file" in error for error in result.errors)
    assert any("missing required directory" in error for error in result.errors)


def test_ci_workflow_contains_phase2_governance_and_category_subgates() -> None:
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

    governance_name = "Phase 2 governance sub-gate (blocking)"
    category_name = "Phase 2 category bootstrap sub-gate (blocking)"
    cross_collect_name = "Tests (cross_check collection guard)"
    cross_name = "Tests (cross_check)"
    regression_name = "Tests (regression smoke selector)"

    for required in (
        governance_name,
        category_name,
        cross_collect_name,
        cross_name,
        regression_name,
    ):
        assert required in named_steps

    governance_run = named_steps[governance_name][1].get("run")
    assert isinstance(governance_run, str)
    assert "python -m rfmna.governance.phase2_gate" in governance_run
    assert "--sub-gate governance" in governance_run

    category_run = named_steps[category_name][1].get("run")
    assert isinstance(category_run, str)
    assert "--sub-gate category-bootstrap" in category_run

    cross_collect_run = named_steps[cross_collect_name][1].get("run")
    assert cross_collect_run == "uv run pytest -m cross_check --collect-only -q"
    cross_run = named_steps[cross_name][1].get("run")
    assert cross_run == "uv run pytest -m cross_check"
    regression_run = named_steps[regression_name][1].get("run")
    assert regression_run == "uv run pytest tests/regression -m regression -q"

    resolve_range_run = named_steps["Resolve governance diff range"][1].get("run")
    assert isinstance(resolve_range_run, str)
    assert 'ZERO_SHA="0000000000000000000000000000000000000000"' in resolve_range_run
    assert 'gov_base="${{ github.event.before }}"' in resolve_range_run
    assert 'git rev-parse --verify --quiet "${gov_base}^{commit}"' in resolve_range_run
    assert 'gov_base="$(git rev-parse "${{ github.sha }}^")"' in resolve_range_run


def test_ci_workflow_always_surfaces_phase2_gate_status_and_upload() -> None:
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

    status_name = "Phase 2 gate status (informational)"
    upload_name = "Upload Phase 2 gate status (informational)"
    assert status_name in named_steps
    assert upload_name in named_steps

    status_index, status_step = named_steps[status_name]
    upload_index, upload_step = named_steps[upload_name]
    assert status_index < upload_index

    assert status_step.get("if") == "${{ always() }}"
    status_run = status_step.get("run")
    assert isinstance(status_run, str)
    assert "phase2_gate_status.txt" in status_run
    assert "docs/dev/phase2_gate.md" in status_run
    assert "checklist_status=present" in status_run
    assert "checklist_status=missing" in status_run

    assert upload_step.get("if") == "${{ always() }}"
    assert upload_step.get("uses") == "actions/upload-artifact@v4"
    with_block = upload_step.get("with")
    assert isinstance(with_block, dict)
    upload_path = with_block.get("path")
    assert isinstance(upload_path, str)
    assert "phase2_gate_status.txt" in upload_path
    assert "phase2_governance_subgate_status.txt" in upload_path
    assert "phase2_category_bootstrap_subgate_status.txt" in upload_path


def test_ci_checkout_fetch_depth_is_full_for_deterministic_diff_detection() -> None:
    test_job = _ci_test_job()
    steps = test_job.get("steps")
    assert isinstance(steps, list)

    checkout_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("uses") == "actions/checkout@v4"
        ),
        None,
    )
    assert isinstance(checkout_step, dict)
    with_block = checkout_step.get("with")
    assert isinstance(with_block, dict)
    assert with_block.get("fetch-depth") == 0

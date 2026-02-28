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


def _ci_named_steps() -> dict[str, tuple[int, dict[str, object]]]:
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
    return named_steps


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _seed_minimal_category_bootstrap_repo(
    repo_root: Path,
    *,
    markers: tuple[str, ...],
) -> None:
    marker_lines = "\n".join(f"    {marker}" for marker in markers)
    _write_text(
        repo_root / "pytest.ini",
        "[pytest]\naddopts = -ra --strict-markers\nmarkers =\n" + marker_lines + "\n",
    )
    _write_text(
        repo_root / "tests/cross_check/test_smoke.py",
        "def test_cross_check_placeholder() -> None:\n    assert True\n",
    )
    _write_text(
        repo_root / "tests/regression/test_smoke.py",
        "def test_regression_placeholder() -> None:\n    assert True\n",
    )
    _write_text(
        repo_root / "docs/dev/regression_fixture_schema_convention.md",
        "regression fixture convention\n",
    )
    _write_text(
        repo_root / "docs/dev/phase2_ci_category_enforcement.md",
        "\n".join(
            (
                "`unit`",
                "`conformance`",
                "`property`",
                "`regression`",
                "`cross_check`",
                "non-empty",
                "--strict-markers",
            )
        )
        + "\n",
    )
    _write_text(
        repo_root / "AGENTS.md",
        "`cross_check` (mandatory for Phase 2 robustness work; reference-comparison tests with documented tolerances)\n",
    )
    _write_text(
        repo_root / ".github/workflows/ci.yml",
        "\n".join(
            (
                "Phase 2 governance sub-gate (blocking)",
                "Phase 2 category bootstrap sub-gate (blocking)",
                "Audit AGENTS Phase 2 cross_check policy alignment",
                "uv run pytest -m unit --collect-only -q",
                "uv run pytest -m unit --junitxml=test-reports/unit.xml",
                "uv run pytest -m conformance --collect-only -q",
                "Tests (thread-controls conformance guard)",
                "test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults",
                "test_thread_controls_conformance.py::test_envrc_declares_deterministic_thread_defaults",
                "uv run pytest -m conformance --junitxml=test-reports/conformance.xml",
                "uv run pytest -m property --collect-only -q",
                "uv run pytest -m property --junitxml=test-reports/property.xml",
                "uv run pytest -m regression --collect-only -q",
                "uv run pytest -m regression --junitxml=test-reports/regression.xml",
                "uv run pytest tests/regression -m regression -q",
                "uv run pytest -m cross_check --collect-only -q",
                "uv run pytest -m cross_check --junitxml=test-reports/cross_check.xml",
                "Upload calibration/regression/cross_check diagnostics (failure)",
            )
        )
        + "\n",
    )


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


def test_phase2_category_bootstrap_gate_fails_when_mandatory_marker_is_missing(
    tmp_path: Path,
) -> None:
    _seed_minimal_category_bootstrap_repo(
        tmp_path,
        markers=("unit", "conformance", "property", "cross_check"),
    )
    result = evaluate_category_bootstrap_gate(repo_root=tmp_path)
    assert not result.passed
    assert any(
        "mandatory Phase 2 categories" in error and "regression" in error for error in result.errors
    )


def test_ci_workflow_contains_phase2_governance_and_category_subgates() -> None:
    named_steps = _ci_named_steps()
    governance_name = "Phase 2 governance sub-gate (blocking)"
    category_name = "Phase 2 category bootstrap sub-gate (blocking)"
    agents_audit_name = "Audit AGENTS Phase 2 cross_check policy alignment"

    for required in (
        governance_name,
        category_name,
        agents_audit_name,
    ):
        assert required in named_steps

    governance_run = named_steps[governance_name][1].get("run")
    assert isinstance(governance_run, str)
    assert "python -m rfmna.governance.phase2_gate" in governance_run
    assert "--sub-gate governance" in governance_run

    category_run = named_steps[category_name][1].get("run")
    assert isinstance(category_run, str)
    assert "--sub-gate category-bootstrap" in category_run

    agents_audit_run = named_steps[agents_audit_name][1].get("run")
    assert isinstance(agents_audit_run, str)
    assert "grep -Fq '`cross_check` (mandatory for Phase 2 robustness work'" in agents_audit_run
    assert "separate governance-labeled task/PR" in agents_audit_run

    resolve_range_run = named_steps["Resolve governance diff range"][1].get("run")
    assert isinstance(resolve_range_run, str)
    assert 'ZERO_SHA="0000000000000000000000000000000000000000"' in resolve_range_run
    assert 'gov_base="${{ github.event.before }}"' in resolve_range_run
    assert 'git rev-parse --verify --quiet "${gov_base}^{commit}"' in resolve_range_run
    assert 'gov_base="$(git rev-parse "${{ github.sha }}^")"' in resolve_range_run


def test_ci_workflow_includes_non_empty_collection_guards_for_all_phase2_categories() -> None:
    named_steps = _ci_named_steps()
    expected_guards = {
        "Tests (unit collection guard)": "uv run pytest -m unit --collect-only -q",
        "Tests (conformance collection guard)": "uv run pytest -m conformance --collect-only -q",
        "Tests (property collection guard)": "uv run pytest -m property --collect-only -q",
        "Tests (regression collection guard)": "uv run pytest -m regression --collect-only -q",
        "Tests (cross_check collection guard)": "uv run pytest -m cross_check --collect-only -q",
    }
    for name, command in expected_guards.items():
        assert name in named_steps
        assert named_steps[name][1].get("run") == command


def test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard() -> None:
    named_steps = _ci_named_steps()
    expected_lanes = {
        "Tests (unit)": "uv run pytest -m unit --junitxml=test-reports/unit.xml",
        "Tests (conformance)": "uv run pytest -m conformance --junitxml=test-reports/conformance.xml",
        "Tests (property)": "uv run pytest -m property --junitxml=test-reports/property.xml",
        "Tests (regression)": "uv run pytest -m regression --junitxml=test-reports/regression.xml",
        "Tests (cross_check)": "uv run pytest -m cross_check --junitxml=test-reports/cross_check.xml",
    }
    for name, command in expected_lanes.items():
        assert name in named_steps
        assert named_steps[name][1].get("run") == command

    regression_scaffold_name = "Tests (regression scaffold selector)"
    assert regression_scaffold_name in named_steps
    assert (
        named_steps[regression_scaffold_name][1].get("run")
        == "uv run pytest tests/regression -m regression -q"
    )

    thread_guard_name = "Tests (thread-controls conformance guard)"
    assert thread_guard_name in named_steps
    thread_guard_run = named_steps[thread_guard_name][1].get("run")
    assert isinstance(thread_guard_run, str)
    assert (
        "test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults"
        in thread_guard_run
    )
    assert (
        "test_thread_controls_conformance.py::test_envrc_declares_deterministic_thread_defaults"
        in thread_guard_run
    )
    assert "-m conformance" in thread_guard_run
    assert "--junitxml=test-reports/thread_controls_conformance.xml" in thread_guard_run


def test_ci_workflow_always_surfaces_phase2_gate_status_and_upload() -> None:
    named_steps = _ci_named_steps()

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


def test_ci_uploads_phase2_failure_diagnostics_for_calibration_regression_cross_check() -> None:
    steps = _ci_test_job().get("steps")
    assert isinstance(steps, list)
    upload_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("name")
            == "Upload calibration/regression/cross_check diagnostics (failure)"
        ),
        None,
    )
    assert isinstance(upload_step, dict)
    assert upload_step.get("if") == "${{ failure() }}"
    assert upload_step.get("uses") == "actions/upload-artifact@v4"
    with_block = upload_step.get("with")
    assert isinstance(with_block, dict)
    assert with_block.get("name") == "phase2-calibration-regression-cross-check-diagnostics"
    upload_path = with_block.get("path")
    assert isinstance(upload_path, str)
    assert "test-reports/regression.xml" in upload_path
    assert "test-reports/cross_check.xml" in upload_path
    assert "docs/dev/p2_08_calibration_report.md" in upload_path
    assert "docs/dev/tolerances/calibration_seed_v1.yaml" in upload_path
    assert "docs/dev/tolerances/regression_baseline_v1.yaml" in upload_path
    assert "tests/fixtures/cross_check/*.json" in upload_path


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

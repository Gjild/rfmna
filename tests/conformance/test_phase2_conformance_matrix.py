from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

pytestmark = pytest.mark.conformance


@dataclass(frozen=True, slots=True)
class Phase2ConformanceRow:
    area: str
    conformance_id: str
    test_id: str
    status: str
    notes: str


PHASE2_CONFORMANCE_MATRIX: tuple[Phase2ConformanceRow, ...] = (
    Phase2ConformanceRow(
        area="backend_scaling_and_pivot_controls",
        conformance_id="P2CM-001",
        test_id=(
            "tests/conformance/test_solver_backend_controls_conformance.py::"
            "test_controlled_fixture_recovers_on_alt_stage_with_deterministic_metadata"
        ),
        status="covered",
        notes="real alt-pivot/scaling controls execute deterministically with explicit metadata",
    ),
    Phase2ConformanceRow(
        area="backend_scaling_and_pivot_controls",
        conformance_id="P2CM-002",
        test_id=(
            "tests/conformance/test_solver_backend_controls_conformance.py::"
            "test_permutation_equivalent_inputs_remain_tolerance_stable_under_alt_scaling_controls"
        ),
        status="covered",
        notes="permutation-equivalent solve behavior remains tolerance-stable under backend controls",
    ),
    Phase2ConformanceRow(
        area="ci_category_enforcement",
        conformance_id="P2CM-003",
        test_id=(
            "tests/conformance/test_phase2_ci_gate_conformance.py::"
            "test_phase2_category_bootstrap_gate_passes_in_repo_baseline"
        ),
        status="covered",
        notes="CI pass-status traceability: baseline category bootstrap gate passes in-repo",
    ),
    Phase2ConformanceRow(
        area="ci_category_enforcement",
        conformance_id="P2CM-004",
        test_id=(
            "tests/conformance/test_phase2_ci_gate_conformance.py::"
            "test_ci_workflow_includes_non_empty_collection_guards_for_all_phase2_categories"
        ),
        status="covered",
        notes="all Phase 2 categories are non-empty guarded in CI",
    ),
    Phase2ConformanceRow(
        area="ci_category_enforcement",
        conformance_id="P2CM-005",
        test_id=(
            "tests/conformance/test_phase2_ci_gate_conformance.py::"
            "test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard"
        ),
        status="covered",
        notes="unit/conformance/property/regression/cross_check lanes and thread-controls guard are enforced",
    ),
    Phase2ConformanceRow(
        area="deterministic_solver_evidence_bundle",
        conformance_id="P2CM-006",
        test_id=(
            "tests/conformance/test_solver_repro_snapshot_conformance.py::"
            "test_solver_repro_snapshot_schema_artifact_has_required_contract_keys"
        ),
        status="covered",
        notes="deterministic evidence bundle includes required solver snapshot contract keys",
    ),
    Phase2ConformanceRow(
        area="deterministic_solver_evidence_bundle",
        conformance_id="P2CM-007",
        test_id=(
            "tests/conformance/test_solver_repro_snapshot_conformance.py::"
            "test_solver_repro_snapshot_defaults_and_inactive_controls_are_explicit_and_deterministic"
        ),
        status="covered",
        notes="inactive/default retry controls serialize deterministically with explicit values",
    ),
    Phase2ConformanceRow(
        area="diagnostics_taxonomy_closure_and_ordering",
        conformance_id="P2CM-008",
        test_id=(
            "tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::"
            "test_track_a_runtime_inventory_guard_passes_baseline"
        ),
        status="covered",
        notes="Track A runtime inventory guard enforces catalog closure",
    ),
    Phase2ConformanceRow(
        area="diagnostics_taxonomy_closure_and_ordering",
        conformance_id="P2CM-009",
        test_id=(
            "tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::"
            "test_track_b_typed_registry_matrix_guard_passes_baseline"
        ),
        status="covered",
        notes="Track B typed error registry/matrix guard enforces deterministic mapping policy",
    ),
    Phase2ConformanceRow(
        area="diagnostics_taxonomy_closure_and_ordering",
        conformance_id="P2CM-010",
        test_id=(
            "tests/conformance/test_sweep_fail_sentinel_conformance.py::"
            "test_fail_point_diagnostics_sort_severity_before_stage"
        ),
        status="covered",
        notes="diagnostics ordering remains deterministic under failure/warning combinations",
    ),
    Phase2ConformanceRow(
        area="freeze_boundary_traceability",
        conformance_id="P2CM-011",
        test_id=(
            "tests/conformance/test_phase2_governance_conformance.py::"
            "test_frozen_rule_table_each_detection_entry_path_is_exercised"
        ),
        status="covered",
        notes="freeze-boundary detection paths are exercised across frozen IDs",
    ),
    Phase2ConformanceRow(
        area="freeze_boundary_traceability",
        conformance_id="P2CM-012",
        test_id=(
            "tests/conformance/test_phase2_governance_conformance.py::"
            "test_rule_table_declares_required_evidence_mapping_for_each_frozen_id"
        ),
        status="covered",
        notes="governance evidence mapping is explicit for frozen IDs 1..12",
    ),
    Phase2ConformanceRow(
        area="freeze_boundary_traceability",
        conformance_id="P2CM-013",
        test_id=(
            "tests/conformance/test_phase2_governance_conformance.py::"
            "test_rule_table_requires_frozen_id_5_sources_in_threshold_source_list"
        ),
        status="covered",
        notes="frozen threshold/status-band boundary (ID #5) is explicitly guarded",
    ),
    Phase2ConformanceRow(
        area="hardened_check_command_contract",
        conformance_id="P2CM-014",
        test_id=(
            "tests/conformance/test_check_command_contract_conformance.py::"
            "test_check_schema_artifact_location_and_version_are_canonical"
        ),
        status="covered",
        notes="`rfmna check` schema location and version contract are fixed and auditable",
    ),
    Phase2ConformanceRow(
        area="hardened_check_command_contract",
        conformance_id="P2CM-015",
        test_id=(
            "tests/conformance/test_check_command_contract_conformance.py::"
            "test_check_json_output_validates_against_canonical_schema"
        ),
        status="covered",
        notes="machine-consumable check payload validates against canonical schema",
    ),
    Phase2ConformanceRow(
        area="hardened_check_command_contract",
        conformance_id="P2CM-016",
        test_id=(
            "tests/conformance/test_check_command_contract_conformance.py::"
            "test_check_json_deterministic_ordering_and_witness_stability_under_permutations"
        ),
        status="covered",
        notes="check diagnostics ordering/witness payloads remain permutation-stable",
    ),
    Phase2ConformanceRow(
        area="rf_fallback_execution_paths",
        conformance_id="P2CM-017",
        test_id=(
            "tests/conformance/test_phase2_rf_fallback_execution_conformance.py::"
            "test_sweep_rf_payload_paths_keep_mna_gmin_eligible_and_conversion_no_gmin"
        ),
        status="covered",
        notes="sweep-integrated RF payload solves keep MNA gmin eligible and conversion no-gmin policy",
    ),
    Phase2ConformanceRow(
        area="rf_fallback_execution_paths",
        conformance_id="P2CM-018",
        test_id=(
            "tests/conformance/test_phase2_rf_fallback_execution_conformance.py::"
            "test_standalone_rf_extractors_keep_mna_gmin_eligible_and_point_alignment"
        ),
        status="covered",
        notes="standalone Y/Z/Zin/Zout solves keep node-count-driven fallback eligibility and point alignment",
    ),
    Phase2ConformanceRow(
        area="rf_warning_propagation_parity",
        conformance_id="P2CM-019",
        test_id=(
            "tests/conformance/test_rf_warning_propagation_conformance.py::"
            "test_rf_warning_propagation_parity_and_context_across_metrics"
        ),
        status="covered",
        notes="RF payload diagnostics preserve solver warning parity with deterministic context",
    ),
    Phase2ConformanceRow(
        area="run_exit_semantics_under_fallback",
        conformance_id="P2CM-020",
        test_id=(
            "tests/conformance/test_cli_exit_conformance.py::"
            "test_run_exit_semantics_remain_intact_when_fallback_controls_are_exercised"
        ),
        status="covered",
        notes="frozen CLI run exit semantics (0/1/2) remain intact under fallback-sensitive outcomes",
    ),
    Phase2ConformanceRow(
        area="sentinel_and_partial_sweep_invariants",
        conformance_id="P2CM-021",
        test_id=(
            "tests/conformance/test_solver_backend_controls_conformance.py::"
            "test_sweep_fail_sentinel_policy_preserved_when_backend_controls_are_exercised"
        ),
        status="covered",
        notes="base sweep keeps full fail-point sentinel/partial-sweep contract under retry/failure",
    ),
    Phase2ConformanceRow(
        area="sentinel_and_partial_sweep_invariants",
        conformance_id="P2CM-022",
        test_id=(
            "tests/conformance/test_rf_fixtures_conformance.py::"
            "test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment"
        ),
        status="covered",
        notes="RF payloads retain sentinel/index alignment for pass/degraded/fail bundles",
    ),
)

EXPECTED_PHASE2_AREAS = {
    "backend_scaling_and_pivot_controls",
    "ci_category_enforcement",
    "deterministic_solver_evidence_bundle",
    "diagnostics_taxonomy_closure_and_ordering",
    "freeze_boundary_traceability",
    "hardened_check_command_contract",
    "rf_fallback_execution_paths",
    "rf_warning_propagation_parity",
    "run_exit_semantics_under_fallback",
    "sentinel_and_partial_sweep_invariants",
}
EXPECTED_COLUMNS = ("area", "conformance_id", "test_id", "status", "notes")


def test_phase2_matrix_columns_are_explicit() -> None:
    assert tuple(field.name for field in fields(Phase2ConformanceRow)) == EXPECTED_COLUMNS


def test_phase2_matrix_declares_expected_areas_and_non_empty_rows() -> None:
    declared_areas = {row.area for row in PHASE2_CONFORMANCE_MATRIX}
    assert declared_areas == EXPECTED_PHASE2_AREAS
    assert len(PHASE2_CONFORMANCE_MATRIX) > 0
    for row in PHASE2_CONFORMANCE_MATRIX:
        assert row.status == "covered"
        assert row.notes.strip()


def test_phase2_matrix_rows_are_unique_and_deterministically_ordered() -> None:
    conformance_ids = [row.conformance_id for row in PHASE2_CONFORMANCE_MATRIX]
    assert len(conformance_ids) == len(set(conformance_ids))

    row_keys = [(row.area, row.conformance_id, row.test_id) for row in PHASE2_CONFORMANCE_MATRIX]
    assert len(row_keys) == len(set(row_keys))
    assert (
        tuple(
            sorted(
                PHASE2_CONFORMANCE_MATRIX,
                key=lambda row: (row.area, row.conformance_id, row.test_id),
            )
        )
        == PHASE2_CONFORMANCE_MATRIX
    )


def test_phase2_matrix_entries_resolve_to_existing_tests(request: pytest.FixtureRequest) -> None:
    collected_nodeids = {item.nodeid.split("[", maxsplit=1)[0] for item in request.session.items}

    missing: list[str] = []
    for row in PHASE2_CONFORMANCE_MATRIX:
        if row.test_id not in collected_nodeids:
            missing.append(f"{row.area}:{row.conformance_id}:{row.test_id}")

    assert missing == []

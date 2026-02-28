# Phase 2 Conformance Coverage Report (P2-10)

## P2-10 deliverables and acceptance (verbatim)

Deliverables:

- `tests/conformance/test_phase2_conformance_matrix.py`
- `docs/dev/phase2_conformance_coverage.md`

Acceptance:

- All matrix entries resolve to executable tests.
- Normative and governance-critical behavior is explicitly auditable.

## Phase 2 conformance matrix

Status values:

- `covered`: mapped to concrete executable tests in this repository.

| area | conformance_id | test_id (`test_file::test_name`) | status | notes |
| --- | --- | --- | --- | --- |
| `backend_scaling_and_pivot_controls` | `P2CM-001` | `tests/conformance/test_solver_backend_controls_conformance.py::test_controlled_fixture_recovers_on_alt_stage_with_deterministic_metadata` | `covered` | real alt-pivot/scaling controls execute deterministically with explicit metadata |
| `backend_scaling_and_pivot_controls` | `P2CM-002` | `tests/conformance/test_solver_backend_controls_conformance.py::test_permutation_equivalent_inputs_remain_tolerance_stable_under_alt_scaling_controls` | `covered` | permutation-equivalent solve behavior remains tolerance-stable under backend controls |
| `ci_category_enforcement` | `P2CM-003` | `tests/conformance/test_phase2_ci_gate_conformance.py::test_phase2_category_bootstrap_gate_passes_in_repo_baseline` | `covered` | CI pass-status traceability: baseline category bootstrap gate passes in-repo |
| `ci_category_enforcement` | `P2CM-004` | `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_includes_non_empty_collection_guards_for_all_phase2_categories` | `covered` | all Phase 2 categories are non-empty guarded in CI |
| `ci_category_enforcement` | `P2CM-005` | `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard` | `covered` | unit/conformance/property/regression/cross_check lanes and thread-controls guard are enforced |
| `deterministic_solver_evidence_bundle` | `P2CM-006` | `tests/conformance/test_solver_repro_snapshot_conformance.py::test_solver_repro_snapshot_schema_artifact_has_required_contract_keys` | `covered` | deterministic evidence bundle includes required solver snapshot contract keys |
| `deterministic_solver_evidence_bundle` | `P2CM-007` | `tests/conformance/test_solver_repro_snapshot_conformance.py::test_solver_repro_snapshot_defaults_and_inactive_controls_are_explicit_and_deterministic` | `covered` | inactive/default retry controls serialize deterministically with explicit values |
| `diagnostics_taxonomy_closure_and_ordering` | `P2CM-008` | `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_a_runtime_inventory_guard_passes_baseline` | `covered` | Track A runtime inventory guard enforces catalog closure |
| `diagnostics_taxonomy_closure_and_ordering` | `P2CM-009` | `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_b_typed_registry_matrix_guard_passes_baseline` | `covered` | Track B typed error registry/matrix guard enforces deterministic mapping policy |
| `diagnostics_taxonomy_closure_and_ordering` | `P2CM-010` | `tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_diagnostics_sort_severity_before_stage` | `covered` | diagnostics ordering remains deterministic under failure/warning combinations |
| `freeze_boundary_traceability` | `P2CM-011` | `tests/conformance/test_phase2_governance_conformance.py::test_frozen_rule_table_each_detection_entry_path_is_exercised` | `covered` | freeze-boundary detection paths are exercised across frozen IDs |
| `freeze_boundary_traceability` | `P2CM-012` | `tests/conformance/test_phase2_governance_conformance.py::test_rule_table_declares_required_evidence_mapping_for_each_frozen_id` | `covered` | governance evidence mapping is explicit for frozen IDs 1..12 |
| `freeze_boundary_traceability` | `P2CM-013` | `tests/conformance/test_phase2_governance_conformance.py::test_rule_table_requires_frozen_id_5_sources_in_threshold_source_list` | `covered` | frozen threshold/status-band boundary (ID #5) is explicitly guarded |
| `hardened_check_command_contract` | `P2CM-014` | `tests/conformance/test_check_command_contract_conformance.py::test_check_schema_artifact_location_and_version_are_canonical` | `covered` | `rfmna check` schema location and version contract are fixed and auditable |
| `hardened_check_command_contract` | `P2CM-015` | `tests/conformance/test_check_command_contract_conformance.py::test_check_json_output_validates_against_canonical_schema` | `covered` | machine-consumable check payload validates against canonical schema |
| `hardened_check_command_contract` | `P2CM-016` | `tests/conformance/test_check_command_contract_conformance.py::test_check_json_deterministic_ordering_and_witness_stability_under_permutations` | `covered` | check diagnostics ordering/witness payloads remain permutation-stable |
| `rf_fallback_execution_paths` | `P2CM-017` | `tests/conformance/test_phase2_rf_fallback_execution_conformance.py::test_sweep_rf_payload_paths_keep_mna_gmin_eligible_and_conversion_no_gmin` | `covered` | sweep-integrated RF payload solves keep MNA gmin eligible and conversion no-gmin policy |
| `rf_fallback_execution_paths` | `P2CM-018` | `tests/conformance/test_phase2_rf_fallback_execution_conformance.py::test_standalone_rf_extractors_keep_mna_gmin_eligible_and_point_alignment` | `covered` | standalone Y/Z/Zin/Zout solves keep node-count-driven fallback eligibility and point alignment |
| `rf_warning_propagation_parity` | `P2CM-019` | `tests/conformance/test_rf_warning_propagation_conformance.py::test_rf_warning_propagation_parity_and_context_across_metrics` | `covered` | RF payload diagnostics preserve solver warning parity with deterministic context |
| `run_exit_semantics_under_fallback` | `P2CM-020` | `tests/conformance/test_cli_exit_conformance.py::test_run_exit_semantics_remain_intact_when_fallback_controls_are_exercised` | `covered` | frozen CLI run exit semantics (0/1/2) remain intact under fallback-sensitive outcomes |
| `sentinel_and_partial_sweep_invariants` | `P2CM-021` | `tests/conformance/test_solver_backend_controls_conformance.py::test_sweep_fail_sentinel_policy_preserved_when_backend_controls_are_exercised` | `covered` | base sweep keeps full fail-point sentinel/partial-sweep contract under retry/failure |
| `sentinel_and_partial_sweep_invariants` | `P2CM-022` | `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment` | `covered` | RF payloads retain sentinel/index alignment for pass/degraded/fail bundles |

## Auditability hooks

- `tests/conformance/test_phase2_conformance_matrix.py` enforces explicit matrix columns (`area`, `conformance_id`, `test_id`, `status`, `notes`), deterministic row ordering, and executable nodeid resolution.
- `tests/conformance/test_phase2_ci_gate_conformance.py::test_phase2_category_bootstrap_gate_passes_in_repo_baseline` provides in-repo CI pass-status traceability evidence for the Phase 2 category gate.

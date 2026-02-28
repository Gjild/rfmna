# Phase 2 Usage (Implemented Surface)

This document describes only implemented behavior with executable evidence in this repository.

## 1) Executable Commands

```bash
# CLI surface
uv run rfmna --help
uv run rfmna check --help
uv run rfmna run --help

# mandatory post-P2-09 validation selectors
uv run pytest -m unit
uv run pytest -m conformance
uv run pytest -m property
uv run pytest -m regression
uv run pytest -m cross_check
```

Evidence:

- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard`
- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_includes_non_empty_collection_guards_for_all_phase2_categories`

## 2) Current Limits

- `rfmna run` and `rfmna check` depend on project-specific design-loader wiring for real design execution.
- `rfmna run` supports `--analysis ac` only.
- Without design-loader wiring, `rfmna check` returns a typed loader-boundary diagnostic (`E_CLI_CHECK_LOADER_FAILED`) and exits non-zero.

Evidence:

- `tests/unit/test_cli_semantics.py::test_run_analysis_guard_non_ac_is_nonzero`
- `tests/unit/test_cli_semantics.py::test_check_loader_boundary_failure_emits_typed_diagnostic`
- `tests/conformance/test_cli_exit_conformance.py::test_check_exit_mapping_covers_warning_json_and_loader_boundary_paths`

## 3) Hardened `check` Contract

- Output modes: `--format text` and `--format json`.
- JSON schema identifier is fixed to `docs/spec/schemas/check_output_v1.json`.
- Exit behavior is locked: `0` when no error diagnostics are present, `2` when any error diagnostic is present.
- JSON diagnostics are deterministically ordered with stable witness payload ordering.

Contract artifact:

- `docs/dev/check_command_contract.md`

Evidence:

- `tests/conformance/test_check_command_contract_conformance.py::test_check_schema_artifact_location_and_version_are_canonical`
- `tests/conformance/test_check_command_contract_conformance.py::test_check_json_output_validates_against_canonical_schema`
- `tests/conformance/test_check_command_contract_conformance.py::test_check_json_deterministic_ordering_and_witness_stability_under_permutations`
- `tests/conformance/test_cli_exit_conformance.py::test_check_exit_mapping`

## 4) Diagnostics Catalog Closure (Phase 2)

- Runtime diagnostic emission paths and codes are inventory-locked in `docs/dev/diagnostic_runtime_code_inventory.yaml`.
- Typed non-diagnostic error codes are registry-tracked in `docs/dev/typed_error_code_registry.yaml` with mapping policy from `docs/dev/typed_error_mapping_matrix.yaml`.
- Runtime diagnostic codes remain catalog-backed through `src/rfmna/diagnostics/catalog.py`.

Evidence:

- `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_a_runtime_inventory_guard_passes_baseline`
- `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_b_typed_registry_matrix_guard_passes_baseline`
- `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_minimum_taxonomy_runtime_codes_are_cataloged_and_typed_only_parse_codes_are_registered`
- `tests/unit/test_diagnostics_adapters.py::test_catalog_codes_are_unique_and_schema_complete`

## 5) Property/Regression/Cross-Check Workflow

Property workflow:

- Property selectors run under `tests/property/`.
- Determinism-sensitive invariants are exercised as property tests.

Regression workflow:

- Golden fixtures are hash-locked and schema-validated.
- Regression tolerances are loaded from `docs/dev/tolerances/regression_baseline_v1.yaml`.
- Regression merge-gating tolerance source classification is enforced.

Cross-check workflow:

- Cross-check fixtures are hash-locked and compared against analytic references.
- Cross-check selectors use normative gating tolerance sources and reject `calibration_only` tolerance sources.

Artifacts:

- `docs/dev/regression_fixture_schema_convention.md`
- `docs/dev/cross_check_reference_tolerance_policy.md`
- `docs/dev/phase2_ci_category_enforcement.md`

Evidence:

- `tests/property/test_solver_diagnostics_sweep_properties.py::test_sweep_fail_sentinel_and_point_presence_invariant`
- `tests/property/test_preflight_properties.py::test_preflight_input_order_permutation_invariance`
- `tests/regression/test_regression_golden_tolerance_suite.py::test_regression_goldens_are_stable_and_tolerance_enforced`
- `tests/regression/test_regression_golden_tolerance_suite.py::test_regression_tolerance_source_is_classified_normative_merge_gating`
- `tests/cross_check/test_cross_check_smoke.py::test_cross_check_fixture_matches_analytic_reference`
- `tests/cross_check/test_cross_check_smoke.py::test_cross_check_gating_sources_are_normative_and_not_calibration_only`

## 6) CI Lanes and Governance Traceability

- CI runs independent lanes for `unit`, `conformance`, `property`, `regression`, and `cross_check`.
- Each lane has a non-empty collection guard.
- CI includes deterministic thread-control conformance checks.
- CI uploads calibration/regression/cross-check diagnostics artifacts on failure.

Evidence:

- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard`
- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_includes_non_empty_collection_guards_for_all_phase2_categories`
- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_uploads_phase2_failure_diagnostics_for_calibration_regression_cross_check`
- `tests/conformance/test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults`

## 7) Frozen-Boundary Reminder

- Run/check exit semantics are governed by frozen artifact #9.
- Canonical API output-shape changes are governed by frozen artifact #10.
- Any frozen-boundary impact requires semver bump, decision record, conformance updates, migration note, and reproducibility impact statement.

Evidence:

- `tests/conformance/test_phase2_governance_conformance.py::test_rule_table_declares_required_evidence_mapping_for_each_frozen_id`
- `tests/conformance/test_cli_exit_conformance.py::test_run_exit_semantics_remain_intact_when_fallback_controls_are_exercised`

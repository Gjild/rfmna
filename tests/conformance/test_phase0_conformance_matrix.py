from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

pytestmark = pytest.mark.conformance

PHASE0_CONFORMANCE_MATRIX: Mapping[str, Sequence[str]] = {
    "canonical_stamps_phase0": (
        "tests/conformance/test_passive_stamps_conformance.py::test_two_node_sign_convention_matches_canonical_equations",
        "tests/conformance/test_sources_stamp_conformance.py::test_independent_current_source_rhs_sign_conventions",
        "tests/conformance/test_sources_stamp_conformance.py::test_independent_voltage_source_aux_current_formulation_and_reference_omission",
        "tests/conformance/test_sources_stamp_conformance.py::test_independent_voltage_source_orientation_sign_conventions",
        "tests/conformance/test_inductor_stamp_conformance.py::test_inductor_aux_current_formulation_matches_canonical_equations",
    ),
    "frequency_grid_behavior": (
        "tests/conformance/test_frequency_grid_conformance.py::test_frequency_grid_linear_log_formulas_endpoints_and_n_equals_one",
        "tests/conformance/test_frequency_grid_conformance.py::test_frequency_grid_invalid_domain_and_deterministic_order",
    ),
    "residual_contract": (
        "tests/conformance/test_solver_residual_conformance.py::test_residual_formula_uses_contract_epsilon_and_matrix_infinity_norm",
        "tests/conformance/test_solver_residual_conformance.py::test_status_band_mapping_is_exact",
    ),
    "fallback_ladder_and_conditioning_defaults": (
        "tests/conformance/test_solver_fallback_conformance.py::test_threshold_artifact_declares_normative_fallback_order",
        "tests/conformance/test_solver_fallback_conformance.py::test_normative_ladder_order_and_no_hidden_retries_and_final_status",
        "tests/conformance/test_solver_fallback_conformance.py::test_per_point_reset_and_gmin_restart",
        "tests/conformance/test_solver_fallback_conformance.py::test_unavailable_condition_indicator_emits_warning",
        "tests/conformance/test_solver_fallback_conformance.py::test_ill_conditioned_band_emits_warning_without_fail",
        "tests/conformance/test_solver_fallback_conformance.py::test_fail_condition_band_forces_numeric_failure_classification",
    ),
    "fail_point_sentinel_policy": (
        "tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence",
    ),
    "ir_serialization_hash_rules": (
        "tests/conformance/test_ir_serialization_conformance.py::test_ir_canonical_serialization_and_hash_are_permutation_invariant",
        "tests/conformance/test_ir_serialization_conformance.py::test_ir_hash_changes_when_canonical_semantics_change",
    ),
    "cli_exit_semantics_and_partial_sweep": (
        "tests/conformance/test_cli_exit_conformance.py::test_run_exit_mapping_and_preflight_gate_and_fail_visibility",
        "tests/conformance/test_cli_exit_conformance.py::test_check_exit_mapping",
    ),
    "deterministic_ordering_invariants": (
        "tests/conformance/test_assembler_pattern_fill_conformance.py::test_two_stage_separation_and_deterministic_slot_mapping",
        "tests/conformance/test_vsource_sign_conformance.py::test_vsource_loop_diagnostics_are_permutation_stable",
        "tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_diagnostics_sort_severity_before_stage",
        "tests/conformance/test_manifest_conformance.py::test_required_fields_present_canonical_deterministic_and_volatile_exclusion",
        "tests/conformance/test_thread_controls_conformance.py::test_ci_workflow_declares_deterministic_thread_defaults",
    ),
}

EXPECTED_FROZEN_ARTIFACT_KEYS = {
    "canonical_stamps_phase0",
    "frequency_grid_behavior",
    "residual_contract",
    "fallback_ladder_and_conditioning_defaults",
    "fail_point_sentinel_policy",
    "ir_serialization_hash_rules",
    "cli_exit_semantics_and_partial_sweep",
    "deterministic_ordering_invariants",
}


def test_phase0_matrix_declares_all_required_artifacts() -> None:
    assert set(PHASE0_CONFORMANCE_MATRIX) == EXPECTED_FROZEN_ARTIFACT_KEYS
    for artifact, entries in PHASE0_CONFORMANCE_MATRIX.items():
        assert entries, f"artifact '{artifact}' must map to at least one conformance test"


def test_phase0_matrix_entries_resolve_to_existing_tests(request: pytest.FixtureRequest) -> None:
    collected_nodeids = {item.nodeid.split("[", maxsplit=1)[0] for item in request.session.items}

    missing: list[str] = []
    for artifact, entries in PHASE0_CONFORMANCE_MATRIX.items():
        for nodeid in entries:
            if nodeid not in collected_nodeids:
                missing.append(f"{artifact}:{nodeid}")

    assert missing == []

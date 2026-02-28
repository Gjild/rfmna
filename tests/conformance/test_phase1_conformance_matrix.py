from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

pytestmark = pytest.mark.conformance

PHASE1_CONFORMANCE_MATRIX: Mapping[str, Sequence[str]] = {
    "vccs_vcvs_stamps": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_vccs_kcl_equations_match_stamp_appendix_section_5",
        "tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_stamp_matches_appendix_section_6_with_aux_row",
        "tests/conformance/test_controlled_stamps_conformance.py::test_full_polarity_orientation_matrix_matches_sign_conventions",
    ),
    "port_sign_conventions": (
        "tests/conformance/test_rf_boundary_conformance.py::test_rfbc_001_voltage_orientation_rows_follow_vp_definition",
        "tests/conformance/test_rf_boundary_conformance.py::test_rfbc_002_current_sign_positive_into_dut",
        "tests/conformance/test_y_params_conformance.py::test_rfy_002_two_port_columns_match_analytic_y_block",
        "tests/conformance/test_z_params_conformance.py::test_rfz_002_two_port_columns_match_analytic_z_block",
    ),
    "yzs_formulas_and_well_posedness_gates": (
        "tests/conformance/test_y_params_conformance.py::test_rfy_001_one_port_voltage_excitation_matches_analytic_y11",
        "tests/conformance/test_z_params_conformance.py::test_rfz_001_one_port_current_excitation_matches_analytic_z11",
        "tests/conformance/test_s_params_conformance.py::test_rfs_001_z_formula_matches_section_3_1",
        "tests/conformance/test_s_params_conformance.py::test_rfs_002_y_formula_matches_section_3_1",
        "tests/conformance/test_z_params_conformance.py::test_rfz_003_y_to_z_singular_gate_emits_explicit_code",
        "tests/conformance/test_z_params_conformance.py::test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code",
    ),
    "z0_validation_and_singular_conversion_diagnostics": (
        "tests/conformance/test_s_params_conformance.py::test_rfs_003_complex_z0_rejected_with_explicit_model_code",
        "tests/conformance/test_s_params_conformance.py::test_rfs_004_nonpositive_z0_rejected_with_explicit_model_code",
        "tests/conformance/test_s_params_conformance.py::test_rfs_005_singular_conversion_emits_explicit_code",
        "tests/conformance/test_z_params_conformance.py::test_rfz_003_y_to_z_singular_gate_emits_explicit_code",
        "tests/conformance/test_z_params_conformance.py::test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code",
    ),
    "deterministic_rf_output_and_diagnostic_ordering": (
        "tests/conformance/test_y_params_conformance.py::test_rfy_002_two_port_columns_match_analytic_y_block",
        "tests/conformance/test_z_params_conformance.py::test_rfz_002_two_port_columns_match_analytic_z_block",
        "tests/conformance/test_s_params_conformance.py::test_rfs_006_diagnostics_ordering_and_witnesses_are_deterministic",
        "tests/conformance/test_z_params_conformance.py::test_rfz_005_diagnostics_are_sorted_and_witness_stable",
        "tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic",
        "tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_diagnostics_sort_severity_before_stage",
    ),
    "pass_degraded_fail_fixtures": (
        "tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment",
        "tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic",
        "tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence",
        "tests/conformance/test_cli_exit_conformance.py::test_run_exit_mapping_and_preflight_gate_and_fail_visibility",
    ),
    "phase0_frozen_artifact_regression_fixture": (
        "tests/conformance/test_phase0_conformance_matrix.py::test_phase0_matrix_declares_all_required_artifacts",
        "tests/conformance/test_phase0_conformance_matrix.py::test_phase0_matrix_entries_resolve_to_existing_tests",
    ),
}

EXPECTED_PHASE1_AREA_KEYS = {
    "vccs_vcvs_stamps",
    "port_sign_conventions",
    "yzs_formulas_and_well_posedness_gates",
    "z0_validation_and_singular_conversion_diagnostics",
    "deterministic_rf_output_and_diagnostic_ordering",
    "pass_degraded_fail_fixtures",
    "phase0_frozen_artifact_regression_fixture",
}


def test_phase1_matrix_declares_all_required_areas() -> None:
    assert set(PHASE1_CONFORMANCE_MATRIX) == EXPECTED_PHASE1_AREA_KEYS
    for area, entries in PHASE1_CONFORMANCE_MATRIX.items():
        assert entries, f"area '{area}' must map to at least one conformance test"


def test_phase1_matrix_entries_resolve_to_existing_tests(request: pytest.FixtureRequest) -> None:
    collected_nodeids = {item.nodeid.split("[", maxsplit=1)[0] for item in request.session.items}

    missing: list[str] = []
    for area, entries in PHASE1_CONFORMANCE_MATRIX.items():
        for nodeid in entries:
            if nodeid not in collected_nodeids:
                missing.append(f"{area}:{nodeid}")

    assert missing == []

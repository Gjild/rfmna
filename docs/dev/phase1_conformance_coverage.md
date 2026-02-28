# Phase 1 Conformance Coverage Report (P1-14)

## P1-14 deliverables and acceptance (verbatim)

Deliverables:

- `tests/conformance/rf_*`
- `tests/fixtures/rf_*`
- Conformance coverage report (ID → test cases)

Acceptance:

- Detects convention drift.
- `unit + conformance` passes.
- No frozen-artifact drift without DR.
- Normative clause coverage is explicit and auditable.

## Acceptance-area coverage matrix

Status values:

- `covered`: mapped to concrete executable tests in this repository.

| P1-14 acceptance area | Tests (`test_file::test_name`) | Status |
| --- | --- | --- |
| VCCS/VCVS stamps | `tests/conformance/test_controlled_stamps_conformance.py::test_vccs_kcl_equations_match_stamp_appendix_section_5`; `tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_stamp_matches_appendix_section_6_with_aux_row`; `tests/conformance/test_controlled_stamps_conformance.py::test_full_polarity_orientation_matrix_matches_sign_conventions` | covered |
| Port sign conventions | `tests/conformance/test_rf_boundary_conformance.py::test_rfbc_001_voltage_orientation_rows_follow_vp_definition`; `tests/conformance/test_rf_boundary_conformance.py::test_rfbc_002_current_sign_positive_into_dut`; `tests/conformance/test_y_params_conformance.py::test_rfy_002_two_port_columns_match_analytic_y_block`; `tests/conformance/test_z_params_conformance.py::test_rfz_002_two_port_columns_match_analytic_z_block` | covered |
| Y/Z/S formulas + well-posedness gates | `tests/conformance/test_y_params_conformance.py::test_rfy_001_one_port_voltage_excitation_matches_analytic_y11`; `tests/conformance/test_z_params_conformance.py::test_rfz_001_one_port_current_excitation_matches_analytic_z11`; `tests/conformance/test_s_params_conformance.py::test_rfs_001_z_formula_matches_section_3_1`; `tests/conformance/test_s_params_conformance.py::test_rfs_002_y_formula_matches_section_3_1`; `tests/conformance/test_z_params_conformance.py::test_rfz_003_y_to_z_singular_gate_emits_explicit_code`; `tests/conformance/test_z_params_conformance.py::test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code` | covered |
| Z0 validation + singular conversion diagnostics | `tests/conformance/test_s_params_conformance.py::test_rfs_003_complex_z0_rejected_with_explicit_model_code`; `tests/conformance/test_s_params_conformance.py::test_rfs_004_nonpositive_z0_rejected_with_explicit_model_code`; `tests/conformance/test_s_params_conformance.py::test_rfs_005_singular_conversion_emits_explicit_code`; `tests/conformance/test_z_params_conformance.py::test_rfz_003_y_to_z_singular_gate_emits_explicit_code`; `tests/conformance/test_z_params_conformance.py::test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code` | covered |
| Deterministic RF output/diagnostic ordering | `tests/conformance/test_y_params_conformance.py::test_rfy_002_two_port_columns_match_analytic_y_block`; `tests/conformance/test_z_params_conformance.py::test_rfz_002_two_port_columns_match_analytic_z_block`; `tests/conformance/test_s_params_conformance.py::test_rfs_006_diagnostics_ordering_and_witnesses_are_deterministic`; `tests/conformance/test_z_params_conformance.py::test_rfz_005_diagnostics_are_sorted_and_witness_stable`; `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic`; `tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_diagnostics_sort_severity_before_stage` | covered |
| Include pass/degraded/fail fixtures | `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment`; `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic`; `tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence`; `tests/conformance/test_cli_exit_conformance.py::test_run_exit_mapping_and_preflight_gate_and_fail_visibility` | covered |
| Regression fixture proving no Phase 0 frozen-artifact drift | `tests/conformance/test_phase0_conformance_matrix.py::test_phase0_matrix_declares_all_required_artifacts`; `tests/conformance/test_phase0_conformance_matrix.py::test_phase0_matrix_entries_resolve_to_existing_tests` | covered |

## Normative clause ID to tests

| Conformance ID | Normative clause | Tests (`test_file::test_name`) |
| --- | --- | --- |
| `RFBC-001` | `port_wave_conventions_v4_0_0.md` Section 1 (`V_p = V(p+) - V(p-)`) | `tests/conformance/test_rf_boundary_conformance.py::test_rfbc_001_voltage_orientation_rows_follow_vp_definition` |
| `RFBC-002` | `port_wave_conventions_v4_0_0.md` Section 1 (`I_p` positive into DUT) | `tests/conformance/test_rf_boundary_conformance.py::test_rfbc_002_current_sign_positive_into_dut` |
| `RFS-001` | `port_wave_conventions_v4_0_0.md` Section 3.1 (`S=(Z-Z0)(Z+Z0)^-1`) | `tests/conformance/test_s_params_conformance.py::test_rfs_001_z_formula_matches_section_3_1` |
| `RFS-002` | `port_wave_conventions_v4_0_0.md` Section 3.1 (`S=(I-Z0Y)(I+Z0Y)^-1`) | `tests/conformance/test_s_params_conformance.py::test_rfs_002_y_formula_matches_section_3_1` |
| `RFS-003` | `port_wave_conventions_v4_0_0.md` Section 3 (complex `Z0` rejected) | `tests/conformance/test_s_params_conformance.py::test_rfs_003_complex_z0_rejected_with_explicit_model_code` |
| `RFS-004` | `port_wave_conventions_v4_0_0.md` Section 3 (non-positive `Z0` rejected) | `tests/conformance/test_s_params_conformance.py::test_rfs_004_nonpositive_z0_rejected_with_explicit_model_code` |
| `RFS-005` | `port_wave_conventions_v4_0_0.md` Section 3.1 (singular conversion fails explicitly) | `tests/conformance/test_s_params_conformance.py::test_rfs_005_singular_conversion_emits_explicit_code` |
| `RFS-006` | diagnostics taxonomy deterministic ordering/witness stability | `tests/conformance/test_s_params_conformance.py::test_rfs_006_diagnostics_ordering_and_witnesses_are_deterministic` |
| `RFY-001` | `port_wave_conventions_v4_0_0.md` Section 2.1 (`I=YV`) | `tests/conformance/test_y_params_conformance.py::test_rfy_001_one_port_voltage_excitation_matches_analytic_y11` |
| `RFY-002` | `port_wave_conventions_v4_0_0.md` Sections 1 + 2.1 (orientation + deterministic columns) | `tests/conformance/test_y_params_conformance.py::test_rfy_002_two_port_columns_match_analytic_y_block` |
| `RFZ-001` | `port_wave_conventions_v4_0_0.md` Section 2.2 (`V=ZI`) | `tests/conformance/test_z_params_conformance.py::test_rfz_001_one_port_current_excitation_matches_analytic_z11` |
| `RFZ-002` | `port_wave_conventions_v4_0_0.md` Sections 1 + 2.2 (orientation + deterministic columns) | `tests/conformance/test_z_params_conformance.py::test_rfz_002_two_port_columns_match_analytic_z_block` |
| `RFZ-003` | `stamp_appendix_v4_0_0.md` Section 9 singular Y->Z gate | `tests/conformance/test_z_params_conformance.py::test_rfz_003_y_to_z_singular_gate_emits_explicit_code` |
| `RFZ-004` | `stamp_appendix_v4_0_0.md` Section 9 ill-conditioned Y->Z gate | `tests/conformance/test_z_params_conformance.py::test_rfz_004_y_to_z_ill_conditioned_gate_emits_explicit_code` |
| `RFZ-005` | diagnostics taxonomy deterministic ordering/witness stability | `tests/conformance/test_z_params_conformance.py::test_rfz_005_diagnostics_are_sorted_and_witness_stable` |
| `STAMP_APPENDIX_5_VCCS_KCL` | `stamp_appendix_v4_0_0.md` Section 5 | `tests/conformance/test_controlled_stamps_conformance.py::test_vccs_kcl_equations_match_stamp_appendix_section_5` |
| `STAMP_APPENDIX_5_VCCS_INVALID_PARAM` | `stamp_appendix_v4_0_0.md` Section 5 validation | `tests/conformance/test_controlled_stamps_conformance.py::test_invalid_controlled_source_parameters_emit_required_codes` |
| `STAMP_APPENDIX_5_6_ORIENTATION_MATRIX` | `stamp_appendix_v4_0_0.md` Sections 5-6 orientation/sign matrix | `tests/conformance/test_controlled_stamps_conformance.py::test_full_polarity_orientation_matrix_matches_sign_conventions` |
| `STAMP_APPENDIX_6_VCVS_AUX_REQUIRED` | `stamp_appendix_v4_0_0.md` Section 6 aux-current requirement | `tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_requires_allocated_aux_unknown` |
| `STAMP_APPENDIX_6_VCVS_INVALID_PARAM` | `stamp_appendix_v4_0_0.md` Section 6 validation | `tests/conformance/test_controlled_stamps_conformance.py::test_invalid_controlled_source_parameters_emit_required_codes` |
| `STAMP_APPENDIX_6_VCVS_STAMP` | `stamp_appendix_v4_0_0.md` Section 6 equation stamp | `tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_stamp_matches_appendix_section_6_with_aux_row` |

## Auditability hooks

- `tests/conformance/test_phase1_conformance_matrix.py` enforces deterministic acceptance-area mapping and verifies each mapped node ID resolves to an existing test.

## RF fixture artifacts (`tests/fixtures/rf_*`) and consumers

| Fixture file | Consuming tests (`test_file::test_name`) |
| --- | --- |
| `tests/fixtures/rf_pass_degraded_fail_sweep_v1.json` | `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_pass_degraded_fail_enforces_sentinel_and_index_alignment` |
| `tests/fixtures/rf_payload_sentinel_ordering_v1.json` | `tests/conformance/test_rf_fixtures_conformance.py::test_rf_fixture_payload_sentinel_and_metric_ordering_are_deterministic` |

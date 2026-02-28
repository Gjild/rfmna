# CLI RF Composition/Dependency Matrix (P1-13)

This snippet defines the deterministic `run --rf` composition behavior implemented in `src/rfmna/cli/main.py`.

## Request normalization

- `--rf` is repeatable.
- Values are normalized by trimming whitespace and lower-casing.
- Invalid values are rejected with `DIAG ... code=E_CLI_RF_METRIC_INVALID` and exit code `2`.
- Repeated valid metrics are de-duplicated.
- Canonical emission order is fixed: `y`, `z`, `s`, `zin`, `zout`.

## Dependency/resolution matrix

| Requested metric | Internal dependency source | Resolution order rule | Failure propagation |
|---|---|---|---|
| `y` | direct Y extraction | metric emitted in canonical order | per-point sentinel/status from Y extractor |
| `z` | direct Z extraction | metric emitted in canonical order | per-point sentinel/status from Z extractor |
| `s` | `from_z` conversion path | resolves after `z` in canonical order | if dependency point fails, S point is fail-sentinel |
| `zin` | shared Zin/Zout extraction | emitted after matrix metrics | per-point sentinel/status from impedance extractor |
| `zout` | shared Zin/Zout extraction | emitted after matrix metrics | per-point sentinel/status from impedance extractor |

## Invalid option combinations

- Any `--rf ...` request with missing design-loader RF port mapping fails with
  `DIAG ... code=E_CLI_RF_OPTIONS_INVALID` and exit code `2`.

## Output grammar compatibility

- Existing `POINT` and `DIAG` lines are unchanged.
- RF output is additive and uses deterministic `RF ...` lines only.

## Executable evidence

- `tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_rf_composition_matrix_rows_use_explicit_dependency_paths`
- `tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_rf_composition_canonical_execution_order_is_request_order_invariant`
- `tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_zin_and_zout_share_single_impedance_extraction_call`
- `tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_s_from_z_dependency_failure_propagates_to_sentinel_and_diagnostics`
- `tests/unit/test_cli_rf_options.py::test_rf_repeat_and_composition_are_canonical_and_deterministic`
- `tests/unit/test_cli_rf_options.py::test_point_and_diag_lines_remain_backward_compatible_with_additive_rf_lines`

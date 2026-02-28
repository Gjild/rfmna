# Phase 1 Usage (Implemented Surface)

This document only describes behavior that is currently implemented and covered by tests in this repository.

## 1) Executable Commands

```bash
# CLI surface (always executable)
uv run rfmna --help
uv run rfmna check --help
uv run rfmna run --help

# run the RF CLI integration tests that exercise --rf behavior
uv run pytest tests/unit/test_cli_rf_options.py -q
```

Notes:

- `rfmna run ...` and `rfmna check ...` require a project-specific design-loader integration.
- Without design-loader wiring, CLI returns a typed parameter error for `rfmna run ...` from `src/rfmna/cli/main.py`.
- Additionally, `rfmna check ...` returns a typed `DIAG` error (`E_CLI_CHECK_LOADER_FAILED`).

## 2) Executable API Example (RF Sweep Payloads)

```bash
uv run python - <<'PY'
import numpy as np
from scipy.sparse import csc_matrix

from rfmna.rf_metrics import PortBoundary
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep


def assemble_point(point_index: int, frequency_hz: float):
    del point_index, frequency_hz
    y = np.asarray(
        [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]],
        dtype=np.complex128,
    )
    return csc_matrix(y), np.zeros(2, dtype=np.complex128)

frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
layout = SweepLayout(n_nodes=2, n_aux=0)
request = SweepRFRequest(
    ports=(
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    ),
    metrics=("y", "z", "s", "zin", "zout"),
)
result = run_sweep(frequencies, layout, assemble_point, rf_request=request)
print(result.status.tolist())
print(result.rf_payloads.metric_names if result.rf_payloads is not None else ())
PY
```

Expected output shape and order are deterministic:

- `result.status` aligns 1:1 with input frequency index order.
- `result.rf_payloads.metric_names` canonical order is `("y", "z", "s", "zin", "zout")`.

Evidence:

- `tests/unit/test_sweep_engine_rf_payloads.py::test_rf_payloads_attach_only_requested_metrics_with_canonical_order`
- `tests/unit/test_sweep_engine_rf_payloads.py::test_rf_payload_ordering_and_values_are_deterministic_under_request_permutations`

## 3) CLI RF Composition/Dependency Matrix (Implemented)

For `rfmna run ... --rf ...`:

- `--rf` is repeatable.
- Values are normalized to lowercase and deduplicated.
- Canonical metric order is `y`, `z`, `s`, `zin`, `zout`.
- Existing `POINT`/`DIAG` lines remain; `RF ...` lines are additive.

Resolution/dependency behavior:

| Requested metric | Dependency path used by implementation | Failure propagation |
|---|---|---|
| `y` | direct Y extraction | failed points remain and carry sentinels/diagnostics |
| `z` | direct Z extraction | failed points remain and carry sentinels/diagnostics |
| `s` | default CLI path is `from_z` | failed upstream points propagate to failed S points |
| `zin` | shared Zin/Zout extraction | failed points remain and carry scalar sentinels/diagnostics |
| `zout` | shared Zin/Zout extraction | failed points remain and carry scalar sentinels/diagnostics |

Invalid combinations and values:

- Unsupported `--rf` value -> `E_CLI_RF_METRIC_INVALID`.
- `--rf` without design-loader RF ports -> `E_CLI_RF_OPTIONS_INVALID`.

Evidence:

- `tests/unit/test_cli_rf_options.py::test_run_accepts_each_rf_metric_and_emits_additive_rf_lines`
- `tests/unit/test_cli_rf_options.py::test_rf_repeat_and_composition_are_canonical_and_deterministic`
- `tests/unit/test_cli_rf_options.py::test_invalid_rf_metric_fails_with_structured_deterministic_diagnostic`
- `tests/unit/test_cli_rf_options.py::test_missing_rf_ports_is_invalid_combination_with_machine_mappable_diagnostic`
- `tests/unit/test_cli_rf_options.py::test_point_and_diag_lines_remain_backward_compatible_with_additive_rf_lines`

## 4) RF Sentinel Contract (Applied)

The implemented RF sentinel behavior follows the shared contract:

1. Failed sweep points are retained; indices are not dropped or reordered.
2. Failed complex scalar payloads use complex-NaN sentinel values.
3. Failed complex matrix payloads use full-matrix complex-NaN fill.
4. Sentinel representation/fill policy is consistent across `y`, `z`, `s`, `zin`, `zout` payloads.
5. Failed points include deterministic diagnostics.

Evidence:

- `tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence`
- `tests/unit/test_sweep_engine_rf_payloads.py::test_rf_payloads_align_indices_and_apply_sentinel_policy_for_failed_points`
- `tests/unit/test_y_params.py::test_failed_point_is_retained_with_full_matrix_nan_sentinel_and_frequency_alignment`
- `tests/unit/test_z_params.py::test_failed_point_is_retained_with_matrix_nan_sentinel_and_diagnostic`
- `tests/unit/test_impedance.py::test_open_fixture_uses_gmin_retry_and_retains_point_ordering`

## 5) Determinism, Failure Modes, and Limits

Determinism notes:

- `DiagnosticEvent` ordering uses the canonical sort policy (`diagnostics.sort.diagnostic_sort_key`).
- Sweep/CLI `SweepDiagnostic` ordering uses `sweep_engine.sweep_diagnostic_sort_key`, which maps
  to `DiagnosticEvent` and reuses the canonical sort key.
- RF metric request ordering is canonicalized before execution.
- RF output ordering is deterministic by metric, point index, and matrix/scalar position.

Failure-mode notes:

- Fail points keep index position and emit diagnostic payloads instead of being removed.
- Invalid CLI RF option inputs fail deterministically with cataloged machine-mappable codes.

Evidence:

- `tests/unit/test_sweep_engine_run.py::test_sweep_diagnostic_sort_key_matches_canonical_diagnostics_sorting`
- `tests/unit/test_sweep_engine_run.py::test_sweep_diagnostic_sort_key_is_permutation_stable`
- `tests/unit/test_cli_semantics.py::test_run_prints_sweep_diagnostics_with_permutation_stable_canonical_order`

Current Phase 1 limits:

- RF Y/Z/S extraction supports 1-port and 2-port only.
- CLI does not expose an `s` conversion-source switch; `run --rf s` uses the default `from_z` path.
- CLI run/check require design-loader integration in the embedding project.

## 6) New Phase 1 Diagnostic Codes (Cataloged)

The following are the cataloged new Phase 1 codes in `src/rfmna/diagnostics/catalog.py`.

| Code | Stage | Behavior documented here | Evidence |
|---|---|---|---|
| `E_MODEL_VCCS_INVALID` | `assemble` | invalid VCCS model validation | `tests/unit/test_elements_controlled.py::test_vccs_reference_and_validation_behavior` |
| `E_MODEL_VCVS_INVALID` | `assemble` | invalid VCVS model validation | `tests/unit/test_elements_controlled.py::test_vcvs_reference_variants_and_validation_behavior` |
| `E_IR_KIND_UNKNOWN` | `assemble` | unknown normalized element kind | `tests/unit/test_elements_factory.py::test_unknown_kind_failure_has_deterministic_code_and_witness` |
| `E_MODEL_PORT_Z0_COMPLEX` | `assemble` | invalid complex Z0 in S conversion | `tests/unit/test_s_params.py::test_complex_z0_emits_explicit_model_code_and_retains_all_points` |
| `E_MODEL_PORT_Z0_NONPOSITIVE` | `assemble` | non-positive/non-finite Z0 in S conversion | `tests/unit/test_s_params.py::test_nonpositive_z0_vector_emits_explicit_model_code` |
| `E_TOPO_RF_BOUNDARY_INCONSISTENT` | `assemble` | inconsistent/unknown RF boundary declarations | `tests/unit/test_rf_boundary.py::test_inconsistent_boundary_values_emit_topology_diagnostic` |
| `E_NUM_RF_BOUNDARY_SINGULAR` | `assemble` | redundant singular RF boundary constraints | `tests/unit/test_rf_boundary.py::test_singular_redundant_voltage_boundaries_emit_numeric_diagnostic` |
| `E_NUM_ZBLOCK_SINGULAR` | `postprocess` | singular Y->Z conversion gate | `tests/unit/test_z_params.py::test_y_to_z_singular_gate_emits_explicit_code_and_no_regularization` |
| `E_NUM_ZBLOCK_ILL_CONDITIONED` | `postprocess` | ill-conditioned Y->Z conversion gate | `tests/unit/test_z_params.py::test_y_to_z_ill_conditioned_gate_emits_explicit_code` |
| `E_NUM_S_CONVERSION_SINGULAR` | `postprocess` | singular conversion matrix in S conversion | `tests/unit/test_s_params.py::test_singular_conversion_emits_explicit_code_without_regularization` |
| `E_NUM_IMPEDANCE_UNDEFINED` | `postprocess` | undefined/non-finite extracted Zin/Zout | `tests/unit/test_impedance.py::test_undefined_impedance_emits_explicit_diagnostic_code` |
| `E_CLI_RF_METRIC_INVALID` | `parse` | unsupported CLI `--rf` metric value | `tests/unit/test_cli_rf_options.py::test_invalid_rf_metric_fails_with_structured_deterministic_diagnostic` |
| `E_CLI_RF_OPTIONS_INVALID` | `parse` | invalid CLI RF option combination | `tests/unit/test_cli_rf_options.py::test_missing_rf_ports_is_invalid_combination_with_machine_mappable_diagnostic` |

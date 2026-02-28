# rfmna

Deterministic RF Modified Nodal Analysis (MNA) solver under the v4 contract.

## Status

This repository currently ships the **Phase 2 robustness** surface with
`unit`, `conformance`, `property`, `regression`, and `cross_check` coverage.

Implemented and test-covered surface:

- Complex sparse unsymmetric AC sweep core with fallback-control execution evidence
- RF extraction paths (`y`, `z`, `s`, `zin`, `zout`) with deterministic warning propagation
- Hardened `rfmna check` contract (`text` + canonical `json` schema mode)
- Runtime diagnostics catalog closure with deterministic Track A/Track B governance guards
- CI-enforced Phase 2 verification lanes for all required categories

Evidence anchors:

- `tests/conformance/test_solver_backend_controls_conformance.py::test_controlled_fixture_recovers_on_alt_stage_with_deterministic_metadata`
- `tests/conformance/test_phase2_rf_fallback_execution_conformance.py::test_sweep_rf_payload_paths_keep_mna_gmin_eligible_and_conversion_no_gmin`
- `tests/conformance/test_rf_warning_propagation_conformance.py::test_rf_warning_propagation_parity_and_context_across_metrics`
- `tests/conformance/test_check_command_contract_conformance.py::test_check_json_output_validates_against_canonical_schema`
- `tests/conformance/test_phase2_diagnostics_taxonomy_conformance.py::test_track_a_runtime_inventory_guard_passes_baseline`
- `tests/conformance/test_phase2_ci_gate_conformance.py::test_ci_workflow_runs_all_phase2_category_lanes_with_thread_controls_guard`

## Usage

```bash
# from repository root
uv python install 3.14
uv sync --all-groups

# executable now
uv run rfmna --help
uv run rfmna run --help
uv run rfmna check --help
```

Detailed implemented usage/limits notes:

- `docs/dev/phase2_usage.md`
- `docs/dev/phase1_usage.md` (historical Phase 1 scope reference)

## Validation

```bash
uv run ruff check .
uv run mypy src
uv run pytest -m unit
uv run pytest -m conformance
uv run pytest -m property
uv run pytest -m regression
uv run pytest -m cross_check
```

## Contract References

- `AGENTS.md`
- `docs/spec/v4_contract.md`
- `docs/spec/*`

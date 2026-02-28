# rfmna

Deterministic RF Modified Nodal Analysis (MNA) solver under the v4 contract.

## Status

This repository currently ships the **Phase 1 RF Utility** surface with unit and conformance coverage.

Implemented and test-covered surface:

- Complex sparse unsymmetric AC sweep core
- RF metric extraction utilities: `y`, `z`, `s`, `zin`, `zout`
- Sweep-engine RF payload integration with deterministic ordering and fail-point sentinels
- CLI commands `check` and `run`, including repeatable `run --rf y|z|s|zin|zout`
- Canonical diagnostics sorting (including sweep/CLI via canonical-equivalent adapter) and
  cataloged machine-mappable diagnostic codes

Evidence anchors:

- `tests/conformance/test_sweep_fail_sentinel_conformance.py::test_fail_point_sentinel_policy_and_full_point_presence`
- `tests/unit/test_sweep_engine_rf_composition_dependencies.py::test_rf_composition_matrix_rows_use_explicit_dependency_paths`
- `tests/unit/test_cli_rf_options.py::test_rf_repeat_and_composition_are_canonical_and_deterministic`

## Usage

```bash
# from repository root
uv python install 3.14
uv sync --all-groups

# executable now
uv run rfmna --help
uv run rfmna run --help
```

Detailed Phase 1 usage and behavior notes:

- `docs/dev/phase1_usage.md`

## Validation

```bash
uv run ruff check .
uv run mypy src
uv run pytest -m "unit or conformance"
```

## Contract References

- `AGENTS.md`
- `docs/spec/v4_contract.md`
- `docs/spec/*`

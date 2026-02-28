# Regression Fixture/Schema Convention (`P2-00` + `P2-06`)

Structured regression assets live under:

- golden fixtures: `tests/fixtures/regression/`
- fixture hash lock: `tests/fixtures/regression/approved_hashes_v1.json`
- schemas: `tests/regression/schemas/`
- regression tests: `tests/regression/test_*.py`

## Naming convention

- Fixture files: `<topic>_vN.json`
- Hash lock files: `approved_hashes_vN.json`
- Schema files: `<topic>_vN.schema.json`

## Deterministic fixture schema policy

- Canonical schema for P2-06 golden fixtures: `tests/regression/schemas/rf_regression_fixture_v1.schema.json`.
- Each fixture must include:
  - `schema_version`
  - `fixture_id`
  - `scenario`
  - `tolerance_profile`
  - `frequencies_hz`
  - `ports`
  - `expected` (metric matrices, statuses, diagnostic-code snapshots)
- Complex-value sentinel points use `[null, null]` representation in fixtures and must map to runtime `nan + 1j*nan`.

## Hash policy and approval workflow

- Approved fixture hashes use canonical JSON serialization (`sort_keys=True`, separators `(",", ":")`, ASCII-safe) and SHA-256.
- Regression tests are read-only and fail on hash mismatch.
- Golden hash updates require explicit approval command:
  - `uv run python scripts/regression/approve_fixture_hashes.py --approve`
- Running approval script without `--approve` must fail to prevent silent rewrites.

## Determinism policy

- Regression fixtures must use canonical ordering for arrays/keys used by assertions.
- Regression selectors must remain deterministic and addressable via `pytest tests/regression -m regression`.
- CI runs the structured selector directly; ad-hoc regression fixture placement outside `tests/fixtures/regression` is not accepted for P2-06 golden coverage.

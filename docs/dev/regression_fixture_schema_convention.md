# Regression Fixture/Schema Convention (`P2-00` bootstrap)

Structured regression assets live under:

- fixtures: `tests/regression/fixtures/`
- schemas: `tests/regression/schemas/`
- tests: `tests/regression/test_*.py`

## Naming convention

- Fixture files: `<topic>_vN.json`
- Schema files: `<topic>_vN.schema.json`

## Determinism policy

- Regression fixtures must use canonical ordering for arrays/keys used by assertions.
- Regression smoke tests must be deterministic and selector-addressable via `pytest tests/regression -m regression`.
- CI runs the structured selector directly; ad-hoc placement outside `tests/regression` is not acceptable for regression bootstrap coverage.

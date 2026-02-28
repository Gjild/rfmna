# Change Scope Declaration Policy (`P2-00`)

`docs/dev/change_scope.yaml` is a mandatory machine-readable declaration consumed by the Phase 2 governance gate.
It is parsed as YAML (JSON subset supported).

## Rules

- `declared_frozen_ids` must be `none` or a sorted unique subset of `1..12`.
- If detected frozen IDs are non-empty, all evidence fields in `change_scope.yaml` must be fully populated:
  - semver bump (`from_version` -> `to_version`),
  - decision record path(s),
  - conformance update path(s),
  - migration note path(s),
  - reproducibility impact statement path.
- Evidence paths must be repository-relative and use required prefixes:
  - `decision_records`: `docs/spec/decision_records/`
  - `conformance_updates`: `tests/conformance/`
  - `migration_notes`: `docs/spec/migration_notes/`
  - `reproducibility_impact_statement_path`: `docs/`
- CI blocks merges when declaration and touched-path-derived scope differ.
- CI blocks merges when required frozen-change evidence is incomplete.

## Schema

- Contract schema: `docs/dev/change_scope_schema_v1.json`
- Enforcement entry point: `python -m rfmna.governance.phase2_gate --sub-gate governance`

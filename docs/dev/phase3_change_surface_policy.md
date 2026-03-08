# Phase 3 Change-Surface Declaration Policy (`P3-00`)

`docs/dev/phase3_change_surface.yaml` is the mandatory machine-readable declaration for non-frozen Phase 3 contract surfaces.
It is parsed as YAML (JSON subset supported).

## Rules

- `declared_surface_ids` must be `none` or a sorted unique subset of surface IDs defined in `docs/dev/phase3_contract_surface_governance_rules.yaml`.
- Touched non-frozen Phase 3 contract surfaces are derived deterministically from the rule table and must exactly match `declared_surface_ids`.
- Exact declaration matching is enforced when the diff touches at least one Phase 3 contract surface or updates `docs/dev/phase3_change_surface.yaml`; unrelated diffs do not re-open stale declaration scope by themselves.
- Each touched surface must name the exact canonical evidence artifact paths required by its rule-table entry; unrelated files in the same evidence bucket do not satisfy the declaration.
- Evidence is machine-checkable and must use repository-relative paths with required prefixes:
  - `policy_docs`: `docs/dev/`
  - `schema_artifacts`: `docs/dev/` or `docs/spec/schemas/` and `.json`
  - `conformance_updates`: `tests/conformance/`
  - `ci_enforcement`: `.github/workflows/`
  - `process_traceability`: `docs/dev/`
- Required evidence keys are determined by the touched surface rule(s); missing required evidence or missing required canonical artifact paths is merge-blocking.
- `docs/dev/change_scope.yaml` remains the only authority for frozen-artifact declaration. Phase 3 change-surface rules must stay disjoint from frozen detection rules.

## Schema

- Contract schema: `docs/dev/phase3_change_surface_schema_v1.json`
- Enforcement entry point: `python -m rfmna.governance.phase3_gate --sub-gate contract-surface`

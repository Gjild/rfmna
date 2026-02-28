<!-- docs/spec/decision_records/2026-02-28-p2-02-fallback-ladder-rf-hardening.md -->
# DR: P2-02 fallback ladder and RF conversion hardening governance record

Status: `accepted`  
Date: `2026-02-28`

## Context

P2-02 introduces deterministic fallback execution hardening across sweep and RF API paths:

- MNA-system solves propagate `node_voltage_count` for eligible paths.
- Conversion-math solves (`Y->S`, `Z->S`, `Y->Z` conversion internals) enforce no gmin regularization.
- `solver_config_snapshot` payload defaults and attempt-trace summary fields are locked by schema and conformance.
- RF extraction/conversion warning propagation is normalized.

These changes touch files mapped by governance rule-table frozen IDs `2`, `3`, `9`, and `11`.

Frozen artifact `#10` (canonical API data shapes/ordering) was explicitly assessed as
`non-impact`: P2-02 does not modify canonical API shape/order contract files
(`src/rfmna/sweep_engine/types.py`, `src/rfmna/viz_io/rf_export.py`,
`src/rfmna/viz_io/manifest.py`).

## Decision

1. Accept P2-02 hardening as a governed frozen-scope change set with full evidence.
2. Keep `rfmna run` exit semantics unchanged (`0/1/2`) while expanding fallback/repro metadata and RF warning parity.
3. Bump package version from `0.1.0` to `0.1.1`.
4. Publish the companion migration note and reproducibility impact statement:
   - `docs/spec/migration_notes/2026-02-28-v0-1-1-p2-02-fallback-ladder-rf-hardening.md`
   - `docs/dev/reproducibility_impact_p2_02.md`

## Consequences

- Conversion singularities are no longer silently rescued by gmin-enabled solve overrides in default RF API wiring.
- Eligible sweep/RF extraction paths now provide explicit node-voltage row counts to fallback/gmin logic.
- Run manifests include deterministic solver snapshot defaults and attempt-trace summary fields on every run.
- Existing CLI exit behavior is unchanged.

## Conformance impact

Conformance coverage updated for:

- run exit semantics under fallback-controlled behavior,
- RF warning propagation parity/context,
- solver repro snapshot schema/default/empty semantics.

## Reproducibility impact

Snapshot payload presence/shape is now explicit and deterministic for all runs, including empty/default attempt-trace summaries when retries are not exercised.

## Migration note

See:
`docs/spec/migration_notes/2026-02-28-v0-1-1-p2-02-fallback-ladder-rf-hardening.md`

## Semver impact

Version bump: `0.1.0` -> `0.1.1`.

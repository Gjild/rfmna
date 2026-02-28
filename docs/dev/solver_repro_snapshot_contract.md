# Solver Repro Snapshot Contract (`v1`)

This document defines the `solver_config_snapshot` payload embedded in run manifests for
Phase 2 fallback/reproducibility hardening (`P2-02`).

Schema:
- `docs/spec/schemas/solver_repro_snapshot_v1.json`

## Required shape

`solver_config_snapshot` is always present on `rfmna run` manifests and must include:

- `schema`: fixed string `solver_repro_snapshot_v1`
- `retry_controls`
- `conversion_math_controls`
- `attempt_trace_summary`

`analysis` is optional and additive (CLI currently sets it to the run analysis mode).

## Default and empty semantics

The snapshot is emitted on every run, including runs where no retries occur.

Default/empty behavior:

- `retry_controls` uses explicit booleans and ordered arrays (no omitted keys).
- `conversion_math_controls.enable_gmin_regularization` is always `false`.
- `attempt_trace_summary` keys are always present.
- If no solve attempts are captured, all count fields are `0`, stage maps are zero-filled, and
  `skip_reason_counts` is `{}`.

## Determinism rules

- Stage maps are fixed-key objects in canonical fallback order:
  `baseline`, `alt_pivot`, `scaling`, `gmin`, `final_fail`.
- `skip_reason_counts` is key-sorted before serialization.
- No run may omit required keys based on control activity.

## Versioning policy

- Additive fields require introducing a new schema version when compatibility cannot be guaranteed.
- Changing required fields, canonical ordering, or object shape must be explicitly classified
  against frozen artifacts:
  - `#9` CLI semantics/output behavior
  - `#10` canonical API data shapes/ordering
- If classified frozen-impacting, full frozen governance evidence is mandatory:
  semver bump, decision record, conformance updates, migration note, and reproducibility impact statement.

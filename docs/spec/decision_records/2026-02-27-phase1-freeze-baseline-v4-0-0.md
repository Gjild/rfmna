<!-- docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md -->
# DR: Establish v4.0.0 Frozen-Artifact Baseline for Phase 1

Status: `accepted`  
Date: `2026-02-27`

## Context

Phase 1 review identified working-tree edits under `docs/spec/*` with no decision-record
evidence, creating frozen-artifact governance drift relative to `AGENTS.md` and
`docs/dev/codex_backlog.md` global constraints.

The following frozen-artifact sources are intentionally present for the Phase 1 baseline:

- `docs/spec/v4_contract.md`
- `docs/spec/frozen_artifacts_v4_0_0.md`
- `docs/spec/stamp_appendix_v4_0_0.md`
- `docs/spec/port_wave_conventions_v4_0_0.md`
- `docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md`
- `docs/spec/thresholds_v4_0_0.yaml`
- `docs/spec/diagnostics_taxonomy_v4_0_0.md`

## Decision

1. Treat the files above as the intentional normative frozen baseline for contract version
   `4.0.0` used by Phase 1 implementation and conformance references.
2. Record this baseline as governed by this decision record; future edits to any frozen item
   require semver change-control workflow per `AGENTS.md`.
3. Publish a migration note for this baseline establishment:
   `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`.

## Consequences

- Frozen-artifact/spec drift is now explicitly governed and auditable.
- Reviewers can distinguish intentional baseline publication from unauthorized spec churn.
- Future frozen-artifact changes remain blocked without DR + conformance + migration updates.

## Conformance Impact

No new formulas are introduced by this governance step. Existing Phase 1 conformance tests
already exercise the referenced frozen behavior (stamps, frequency rules, solver thresholds,
CLI semantics, and fail-point policy).

## Reproducibility Impact

No numerical or ordering behavior change is introduced by this governance action. This is a
documentation/process control update only.

## Migration Note

See:
`docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`

## Semver Impact

Contract baseline is established at version `4.0.0` (initial publication in this branch).
No additional semantic bump beyond `4.0.0` is introduced by this DR.

# Phase 1 Process Traceability Record

This record keeps Phase 1 assumptions/scope/governance evidence in-repo so verification does
not depend on PR-only metadata.

## Assumptions

- `phase_contract`: Phase 1 work in this branch is evaluated against the v4.0.0 frozen-artifact baseline.
- `design_loader_boundary`: CLI `check`/`run` behavior assumes embedding-project design-loader wiring.
- `test_evidence_scope`: compliance evidence is taken from in-repo docs and unit/conformance tests.

## Scope Boundaries

- `in_scope`: governance docs in `docs/dev/*`, deterministic behavior and failure contracts
  documented in `docs/dev/phase1_usage.md`, and tests under `tests/unit` + `tests/conformance`.
- `out_of_scope`: unapproved normative/spec edits under `docs/spec/*`, and claims not backed by
  current repository tests.

## Governance Links

- `authority_backlog`: `docs/dev/codex_backlog.md`
- `authority_agents`: `AGENTS.md`
- `baseline_dr`: `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`
- `baseline_migration_note`: `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`
- `phase_gate`: `docs/dev/phase1_gate.md`

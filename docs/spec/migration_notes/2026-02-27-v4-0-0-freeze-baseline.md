<!-- docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md -->
# Migration Note: v4.0.0 Frozen-Artifact Baseline

Date: `2026-02-27`  
Related DR: `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`

## Summary

This change formalizes the `v4.0.0` frozen-artifact baseline used by Phase 1 work and
records required governance evidence for the existing `docs/spec/*` artifact set.

## Consumer impact

- Implementations should continue to treat `docs/spec/v4_contract.md` and
  versioned appendices/thresholds as normative.
- No runtime migration steps are required for users already consuming the current branch.

## Required action

None for runtime behavior. For future frozen-artifact edits, follow DR + semver +
conformance + migration workflow before merging.

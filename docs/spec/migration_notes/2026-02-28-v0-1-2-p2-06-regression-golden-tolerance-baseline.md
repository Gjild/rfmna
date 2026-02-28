<!-- docs/spec/migration_notes/2026-02-28-v0-1-2-p2-06-regression-golden-tolerance-baseline.md -->
# Migration Note: v0.1.2 P2-06 regression golden/tolerance baseline

Date: `2026-02-28`  
Related DR: `docs/spec/decision_records/2026-02-28-p2-06-regression-golden-tolerance-baseline-v0-1-2.md`

## Summary

This release expands regression coverage with deterministic golden fixtures, tolerance-table-driven assertions, and approved fixture-hash locking.

## Consumer impact

- Regression pass/fail tolerances now originate from `docs/dev/tolerances/regression_baseline_v1.yaml`.
- Approved fixture hashes are locked in `tests/fixtures/regression/approved_hashes_v1.json`.
- Golden hash updates require explicit approval (`--approve`) and are no longer silently writable in test flows.

## Required action

1. When intentionally updating regression fixtures, run:
   - `uv run python scripts/regression/approve_fixture_hashes.py --approve`
2. Keep merge-gating tolerance changes governance-compliant with DR + migration + conformance + reproducibility evidence.

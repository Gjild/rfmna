<!-- docs/spec/decision_records/2026-02-28-p2-06-regression-golden-tolerance-baseline-v0-1-2.md -->
# DR: P2-06 regression golden suite and tolerance baseline governance record

Status: `accepted`  
Date: `2026-02-28`

## Context

P2-06 expands the structured regression scaffold into a dedicated golden suite with
tolerance-table-driven pass/fail checks and approved fixture-hash locking.

This task introduces `docs/dev/tolerances/regression_baseline_v1.yaml` and classifies it as
`normative_gating` and merge-gating in `docs/dev/threshold_tolerance_classification.yaml`.

No frozen artifact IDs (`1..12`) are touched by this change set.

## Decision

1. Accept `regression_baseline_v1` as a governed merge-gating tolerance source for regression pass/fail assertions.
2. Require explicit golden hash approval (`--approve`) for updates to approved regression fixture hashes.
3. Bump package version from `0.1.1` to `0.1.2`.
4. Publish task-specific migration and reproducibility records:
   - `docs/spec/migration_notes/2026-02-28-v0-1-2-p2-06-regression-golden-tolerance-baseline.md`
   - `docs/dev/reproducibility_impact_p2_06.md`

## Consequences

- Regression tolerance updates in the normative baseline are governance-blocking without full evidence.
- Regression goldens are deterministic and hash-locked against silent rewrites.
- Sentinel policy assertions are explicit in golden fixtures (`[null, null]` -> `nan + 1j*nan`).

## Conformance impact

Conformance/governance coverage updated for classification and merge-gating expectations:

- `tests/conformance/test_phase2_governance_conformance.py`

## Reproducibility impact

Golden fixture lock checks use canonical JSON + SHA-256 with deterministic fixture-set parity and explicit approval workflow.

## Migration note

See:
`docs/spec/migration_notes/2026-02-28-v0-1-2-p2-06-regression-golden-tolerance-baseline.md`

## Semver impact

Version bump: `0.1.1` -> `0.1.2`.

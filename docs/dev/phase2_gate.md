# Phase 2 Gate + Freeze-Boundary Verification (`P2-00`)

This checklist is the authoritative Phase 2 governance gate.

## Baseline gate checks

- [ ] `docs/dev/change_scope.yaml` is present and machine-valid.
- [ ] Frozen-ID detection from `docs/dev/frozen_change_governance_rules.yaml` is deterministic and matches declaration.
- [ ] Full frozen-evidence bundle is present whenever any frozen ID is touched.
- [ ] Threshold/tolerance classification artifact is present and machine-valid.
- [ ] Merge-gating tolerance sources are classified `normative_gating` only.
- [ ] `cross_check` marker + strict-marker enforcement + non-empty lane guard are active.
- [ ] Regression scaffold selector executes from `tests/regression`.

## Frozen artifact checklist (12)

1. Canonical element stamp equations (`docs/spec/stamp_appendix_v4_0_0.md`)
2. 2-port Y/Z block equations and stamping policy (`docs/spec/stamp_appendix_v4_0_0.md`)
3. Port current/voltage and wave conventions (`docs/spec/port_wave_conventions_v4_0_0.md`)
4. Residual formula and condition-indicator definition (`docs/spec/v4_contract.md`, `docs/spec/thresholds_v4_0_0.yaml`)
5. Threshold table values and status bands (`docs/spec/thresholds_v4_0_0.yaml`)
6. Retry ladder order/defaults (`docs/spec/v4_contract.md`, `docs/spec/thresholds_v4_0_0.yaml`)
7. IR serialization/hash rules (`src/rfmna/ir/serialize.py`, `src/rfmna/ir/models.py`)
8. Frequency grammar and grid generation rules (`docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md`, `src/rfmna/sweep_engine/frequency_grid.py`)
9. CLI exit semantics and partial-sweep behavior (`docs/spec/v4_contract.md`, `src/rfmna/cli/main.py`)
10. Canonical API data shapes and ordering (`src/rfmna/sweep_engine/types.py`, `src/rfmna/viz_io/*`)
11. Fail-point sentinel policy (`src/rfmna/sweep_engine/run.py`, `src/rfmna/rf_metrics/*`)
12. Deterministic thread-control defaults (`.envrc`, `.github/workflows/ci.yml` thread-env lines)

## Required evidence for frozen-artifact changes

Any touched frozen ID requires all of:

- Semantic version bump.
- Decision record in `docs/spec/decision_records/`.
- Conformance updates.
- Migration note.
- Reproducibility-impact statement.

## Rule and classification artifacts

- Frozen-ID rule table + path detection rules: `docs/dev/frozen_change_governance_rules.yaml`
- Rule-table evidence mapping: `required_evidence_by_frozen_id` in `docs/dev/frozen_change_governance_rules.yaml`
- Threshold/tolerance classification table: `docs/dev/threshold_tolerance_classification.yaml`
- Change-scope declaration artifact: `docs/dev/change_scope.yaml`
- Change-scope schema: `docs/dev/change_scope_schema_v1.json`
- Change-scope policy note: `docs/dev/change_scope_policy.md`
- Governance tamper-resistance is enforced by evaluating detection/classification against baseline (`base-ref`) artifacts in CI, not mutable head-branch edits.

## CI enforcement anchors

- Informational checklist artifact: `.github/workflows/ci.yml` step `Phase 2 gate status (informational)`
- Blocking governance sub-gate: `.github/workflows/ci.yml` step `Phase 2 governance sub-gate (blocking)`
- Blocking category-bootstrap sub-gate: `.github/workflows/ci.yml` step `Phase 2 category bootstrap sub-gate (blocking)`
- Executable governance checker: `python -m rfmna.governance.phase2_gate`

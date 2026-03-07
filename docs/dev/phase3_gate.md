# Phase 3 Gate and Boundary Bootstrap (`P3-00`)

This checklist is the authoritative Phase 3 governance gate.

## Baseline gate checks

- [ ] Phase 2 inherited blockers remain active and blocking:
  - `python -m rfmna.governance.phase2_gate --sub-gate governance`
  - `python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap`
- [ ] `docs/dev/phase3_change_surface.yaml` is present and machine-valid.
- [ ] Phase 3 contract-surface detection from `docs/dev/phase3_contract_surface_governance_rules.yaml` is deterministic and matches declaration.
- [ ] Contract-surface declaration matching is diff-scoped: unrelated diffs do not fail solely because the checked-in declaration artifact records prior bootstrap scope.
- [ ] Required non-frozen evidence is present whenever a Phase 3 contract surface is touched, including the exact canonical artifact paths required by the touched surface rule.
- [ ] Phase 3 anti-tamper checks evaluate governance rules against `base-ref` data when available; bootstrap is allowed only for first-introduction paths.
- [ ] Missing/unresolvable base-ref context is blocking for Phase 3 anti-tamper and active optional-track checks.
- [ ] Optional-track activation is deterministic, path-aware, and freshness-checked against the base-ref commit UTC date.
- [ ] Active optional-track records cannot be mutated by head-only edits; anti-tamper compares activated baseline records against current content.
- [ ] CI always emits Phase 3 gate status logs and uploads a Phase 3 gate status artifact.

## Inherited Phase 2 blockers

- Frozen-ID declaration artifact: `docs/dev/change_scope.yaml`
- Frozen-ID detection/rule table: `docs/dev/frozen_change_governance_rules.yaml`
- Threshold/tolerance classification table: `docs/dev/threshold_tolerance_classification.yaml`
- Mandatory category selectors and guards: `docs/dev/phase2_ci_category_enforcement.md`
- Executable inherited blockers:
  - `python -m rfmna.governance.phase2_gate --sub-gate governance`
  - `python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap`

## Phase 3 governance artifacts

- Gate checklist: `docs/dev/phase3_gate.md`
- Process traceability: `docs/dev/phase3_process_traceability.md`
- Contract-surface rules: `docs/dev/phase3_contract_surface_governance_rules.yaml`
- Phase 3 change-surface declaration: `docs/dev/phase3_change_surface.yaml`
- Phase 3 change-surface schema: `docs/dev/phase3_change_surface_schema_v1.json`
- Phase 3 change-surface policy: `docs/dev/phase3_change_surface_policy.md`
- Optional-track activation declaration: `docs/dev/optional_track_activation.yaml`
- Optional-track activation schema: `docs/dev/optional_track_activation_schema_v1.json`
- Optional-track activation policy: `docs/dev/optional_track_activation_policy.md`
- Executable governance checker: `python -m rfmna.governance.phase3_gate`

## Independently blocking sub-gates

1. Inherited Phase 2 blockers:
   - `python -m rfmna.governance.phase2_gate --sub-gate governance`
   - `python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap`
2. Phase 3 contract-surface governance:
   - `python -m rfmna.governance.phase3_gate --sub-gate contract-surface`
3. Phase 3 anti-tamper:
   - `python -m rfmna.governance.phase3_gate --sub-gate anti-tamper`
4. Optional-track activation (conditional/path-aware):
   - `python -m rfmna.governance.phase3_gate --sub-gate optional-track`

## Evidence requirements for Phase 3 non-frozen contract surfaces

Any touched Phase 3 contract surface must provide the machine-checkable evidence required by `docs/dev/phase3_contract_surface_governance_rules.yaml` through `docs/dev/phase3_change_surface.yaml`:

- policy docs in `docs/dev/`
- schema artifacts in `docs/dev/`
- conformance updates in `tests/conformance/`
- CI enforcement in `.github/workflows/`
- process-traceability records in `docs/dev/`

## Optional-track policy anchors

- Tracks are deferred by default in `docs/dev/optional_track_activation.yaml`.
- Activation is gateable only with explicit approval, fresh usage evidence, and an activation-PR frozen-ID declaration in `docs/dev/change_scope.yaml`.
- Freshness uses `usage_evidence_date` (`YYYY-MM-DD`) compared against the `base-ref` commit UTC date.
- If `docs/dev/optional_track_activation.yaml` itself is edited, malformed activation content is merge-blocking even when no implementation track is otherwise active.
- Touching optional-track implementation scope without valid activation evidence is blocking, including token-matched shared integration files reserved in `docs/dev/phase3_contract_surface_governance_rules.yaml`.

## CI enforcement anchors

- Blocking contract-surface sub-gate: `.github/workflows/ci.yml` step `Phase 3 contract-surface governance sub-gate (blocking)`
- Blocking anti-tamper sub-gate: `.github/workflows/ci.yml` step `Phase 3 anti-tamper sub-gate (blocking)`
- Blocking optional-track sub-gate: `.github/workflows/ci.yml` step `Phase 3 optional-track activation sub-gate (blocking)`
- Informational checklist artifact: `.github/workflows/ci.yml` step `Phase 3 gate status (informational)`

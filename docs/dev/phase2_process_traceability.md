# Phase 2 Process Traceability Record

This record anchors Phase 2 governance assumptions and enforcement links in-repo.

## Assumptions

- `phase_contract`: Phase 2 robustness work remains under the v4.0.0 frozen-artifact contract unless explicitly changed through formal governance evidence.
- `change_scope_declaration`: every change declares frozen impact in `docs/dev/change_scope.yaml`.
- `classification_policy`: merge-gating tolerance sources must be classified `normative_gating`.

## Scope Boundaries

- `in_scope`: governance gate enforcement, threshold/tolerance classification policy, `cross_check` bootstrap lane, regression scaffold bootstrap.
- `out_of_scope`: frozen semantic changes without full governance evidence and non-Phase-2 feature expansion.

## Governance Links

- `authority_backlog`: `docs/dev/phase2_backlog.md`
- `authority_agents`: `AGENTS.md`
- `phase_gate`: `docs/dev/phase2_gate.md`
- `change_scope_policy`: `docs/dev/change_scope_policy.md`
- `frozen_rule_table`: `docs/dev/frozen_change_governance_rules.yaml`
- `threshold_classification`: `docs/dev/threshold_tolerance_classification.yaml`
- `governance_checker`: `src/rfmna/governance/phase2_gate.py`

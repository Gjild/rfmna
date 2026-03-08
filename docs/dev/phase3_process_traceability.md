# Phase 3 Process Traceability Record

This record anchors Phase 3 governance assumptions, evidence artifacts, and CI links in-repo.

## Assumptions

- `phase_contract`: Phase 3 productivity work remains under the v4.0.0 contract and frozen-artifact governance defined by `docs/spec/v4_contract.md`, `AGENTS.md`, and `docs/dev/change_scope.yaml`.
- `base_ref_contract`: Phase 3 anti-tamper evaluation requires an explicit resolvable `base-ref`; no permissive head-only fallback is allowed after bootstrap.
- `phase3_change_surface_declaration`: non-frozen Phase 3 contract surfaces are declared in `docs/dev/phase3_change_surface.yaml`.
- `optional_track_default_state`: optional tracks P3-10 and P3-11 are deferred by default until activated by schema-valid evidence in `docs/dev/optional_track_activation.yaml`.

## Scope Boundaries

- `in_scope`: Phase 3 gate/bootstrap artifacts, contract-surface governance, anti-tamper enforcement, path-aware optional-track activation policy, CI informational/blocking links, and the governed P3-01 design-bundle contract surface.
- `out_of_scope`: hierarchy, model-card, template, and optional-track implementation work beyond the current design-bundle loader scope.

## Governance Links

- `authority_backlog`: `docs/dev/phase3_backlog.md`
- `authority_agents`: `AGENTS.md`
- `phase_gate`: `docs/dev/phase3_gate.md`
- `frozen_scope_declaration`: `docs/dev/change_scope.yaml`
- `phase3_change_surface_policy`: `docs/dev/phase3_change_surface_policy.md`
- `phase3_change_surface_schema`: `docs/dev/phase3_change_surface_schema_v1.json`
- `phase3_contract_surface_rule_table`: `docs/dev/phase3_contract_surface_governance_rules.yaml`
- `design_bundle_contract`: `docs/dev/design_bundle_contract.md`
- `design_bundle_schema`: `docs/spec/schemas/design_bundle_v1.json`
- `design_bundle_interim_exclusions`: `docs/dev/p3_loader_temporary_exclusions.yaml`
- `optional_track_policy`: `docs/dev/optional_track_activation_policy.md`
- `optional_track_schema`: `docs/dev/optional_track_activation_schema_v1.json`
- `governance_checker`: `src/rfmna/governance/phase3_gate.py`
- `ci_workflow`: `.github/workflows/ci.yml`

You are reviewing a newly authored Phase 3 backlog document for this repository.

## Objective

Perform a rigorous review of:

- `/home/miba/rfmna/docs/dev/phase3_backlog.md`

The goal is to determine whether it is consistent with repository governance, technically sound, complete enough to execute, and stylistically aligned with existing phase backlog standards.

## Repository Context

- Repo root: `/home/miba/rfmna`
- Project: deterministic RF MNA solver under strict v4 contract
- Current package version: `0.1.2`
- Phase 2 is complete; this new document proposes the Phase 3 implementation backlog.

## What changed

A new file was added:

- `/home/miba/rfmna/docs/dev/phase3_backlog.md`

No other files were changed.

## Mandatory authority order for review

Use this precedence while reviewing:

1. `docs/spec/v4_contract.md`
2. Normative appendices and threshold artifacts in `docs/spec/*`
3. `AGENTS.md`
4. Existing tests/fixtures and established Phase 2 governance patterns
5. Authoring convenience/style

## Required comparison artifacts

At minimum, read these before concluding:

- `/home/miba/rfmna/AGENTS.md`
- `/home/miba/rfmna/docs/spec/v4_contract.md`
- `/home/miba/rfmna/docs/spec/frozen_artifacts_v4_0_0.md`
- `/home/miba/rfmna/docs/spec/diagnostics_taxonomy_v4_0_0.md`
- `/home/miba/rfmna/docs/dev/phase2_backlog.md`
- `/home/miba/rfmna/docs/dev/phase2_gate.md`
- `/home/miba/rfmna/docs/dev/phase2_conformance_coverage.md`
- `/home/miba/rfmna/docs/initial_project_description.md`
- `/home/miba/rfmna/docs/dev/phase3_backlog.md`

## Review focus (must cover all)

1. Governance correctness:
- No accidental conflict with frozen artifact policy.
- Correct handling of semver/DR/migration/conformance/reproducibility evidence requirements.
- No suggestions that would bypass Phase 2 governance gates.

2. Contract alignment:
- No hidden drift from v4 contract constraints (sparse unsymmetric class, deterministic ordering, sentinel policy, no silent regularization).
- No ungoverned exit-semantics/API-shape assumptions.

3. Backlog quality:
- Tasks are actionable and testable (Goal/Scope/Deliverables/Acceptance).
- Execution order is coherent.
- Exit criteria are measurable.
- Optional tracks are clearly gated.

4. Completeness for Phase 3 intent:
- Covers subcircuits/macros, model cards, productivity/reporting/templates, and example workflows.
- Addresses current known gap that CLI loader is still stubbed.
- Preserves architecture boundaries.

5. Internal consistency:
- No contradictory statements between sections.
- No ambiguous acceptance bullets that could permit weak implementation.

## Severity model

Classify findings as:

- `high`: policy/contract violation risk or likely to cause incorrect implementation/governance drift
- `medium`: missing/ambiguous acceptance criteria, incomplete scope, or traceability weakness
- `low`: wording/structure clarity issues that do not materially alter implementation risk

## Output format (strict)

1. Findings first, ordered by severity (`high` -> `medium` -> `low`).
2. Each finding must include:
- severity
- concise title
- file reference with line number(s)
- why it is a problem
- precise recommended fix
3. Then include:
- open questions/assumptions
- brief overall assessment

If no findings are identified, state that explicitly and list any residual risks/testing gaps.

## Constraints for this review session

- Do **not** edit files.
- Do **not** propose speculative features outside documented Phase 3 scope unless framed as optional follow-up.
- Keep recommendations compatible with existing AGENTS and v4 governance policy.
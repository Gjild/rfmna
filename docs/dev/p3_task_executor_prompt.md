# Phase 3 Task Executor Prompt Template

Use this template to execute any single Phase 3 task (`P3-XX`) from `docs/dev/phase3_backlog.md`.
Replace all `{{...}}` placeholders before use.

## Template

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: {{TASK_ID}}   # e.g., P3-03
- Task title: {{TASK_TITLE}}
- Execution-order context: {{ORDER_CONTEXT}}   # e.g., core order #4/14 or conditional track

Repository root:
- {{REPO_ROOT}}   # e.g., /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for {{TASK_ID}})

Execution contract:
1) Implement only the selected task scope and acceptance criteria. Do not pull in unrelated refactors.
2) If any conflict appears between backlog text and higher-priority artifacts, stop and align to higher-priority policy.
3) Preserve deterministic behavior, fail-point sentinel policy, diagnostics schema rules, and no-silent-regularization rules.
4) Preserve strict two-stage assembly for topology-stable sweeps: compile sparse pattern/index maps once, numeric fill per point only.
5) Treat frozen artifacts as governance-controlled:
   - If impacted, require semver bump + decision record + migration note + conformance updates + reproducibility impact statement.
6) Keep module boundaries per AGENTS.md; do not blend parser/model logic into solver internals.
7) No TODO/pseudo-implementation in production paths.

Required workflow:
1) Read {{TASK_ID}} Goal/Scope/Deliverables/Acceptance from docs/dev/phase3_backlog.md.
2) Copy exact acceptance bullets into a short execution plan and map each bullet to intended file/test changes.
3) Implement code/docs/tests needed for acceptance closure.
4) Update governance artifacts when required by scope:
   - docs/dev/change_scope.yaml
   - task-required schema/contract/governance artifacts (for example Phase 3 gate/surface rules, optional-track artifacts, matrix/coverage artifacts)
5) Run validation commands and report results.
6) Report file-by-file changes and explicit acceptance mapping.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Additional required selectors by change type:
- Bug fix: run relevant regression selectors now (no deferral).
- Determinism/ordering/hash/sentinel/partial-sweep changes: run conformance + regression selectors.
- Cross-check/tolerance changes: run cross_check + regression selectors.
- Property-domain logic changes: run property selectors.
- New orientation logic: add/update sign-convention tests.
- New topology checks: add deterministic witness tests.
- Schema/output-surface changes: add compatibility-policy checks and schema validation tests.
- New failure codes: update Track A/Track B governance artifacts and run CI-negative guards.

Phase 3-specific governance checks:
- Preserve inherited Phase 2 blockers:
  - python -m rfmna.governance.phase2_gate --sub-gate governance
  - python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap
- Preserve Phase 3 gate semantics when applicable:
  - python -m rfmna.governance.phase3_gate --sub-gate {{P3_SUB_GATE_OR_ALL}}
- Optional-track gate is path-aware:
  - If optional-track scope is touched OR optional track is activated, activation evidence checks are blocking.

Output format (strict):
1) Implementation summary (what changed and why)
2) Acceptance checklist for {{TASK_ID}} with pass/fail evidence per bullet
3) Files changed (path + purpose)
4) Test commands executed + pass/fail
5) Governance impact statement:
   - Frozen artifact IDs touched (or "none")
   - Required evidence added (or reason not required)
   - Track A/Track B updates (or "none")
6) Residual risks / follow-ups (must be empty for completed scope, otherwise explicit blockers)

If blocked:
- Stop and report the exact blocking conflict, affected files, and the minimum policy-compliant resolution path.
```

## Adaptation Notes

- Keep one prompt per task/PR. Do not batch multiple `P3-XX` tasks in one execution prompt.
- For conditional tasks (`P3-10`, `P3-11`), include activation/defer state and enforce path-aware optional-track gating.
- For loader/composition tasks, explicitly map tests to frozen non-regression obligations (`#3`, `#9`, `#11`) where applicable.
- For conformance/closure tasks (`P3-12`, `P3-13`), require matrix/docs claims to resolve to executable tests.

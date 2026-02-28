# Phase 2 Task Executor Prompt Template

Use this template to execute any single Phase 2 task (`P2-XX`) from `docs/dev/phase2_backlog.md`.
Replace all `{{...}}` placeholders before use.

## Template

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: {{TASK_ID}}   # e.g., P2-03
- Task title: {{TASK_TITLE}}
- Phase ordering context: {{PRE_OR_POST_P2_09}}   # pre-P2-09 or post-P2-09

Repository root:
- {{REPO_ROOT}}   # e.g., /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for {{TASK_ID}})

Execution contract:
1) Implement only the selected task scope and acceptance criteria. Do not pull in unrelated refactors.
2) If any conflict appears between backlog text and higher-priority artifacts, stop and align to higher-priority policy.
3) Preserve deterministic behavior, sentinel policy, diagnostics schema rules, and no-silent-regularization rules.
4) Treat frozen artifacts as governance-controlled:
   - If impacted, require semver bump + DR + migration note + conformance updates + reproducibility impact statement.
5) Keep module boundaries per AGENTS.md; do not blend parser/model logic into solver internals.
6) No TODO/pseudo-implementation in production paths.

Required workflow:
1) Read {{TASK_ID}} Goal/Scope/Deliverables/Acceptance from docs/dev/phase2_backlog.md.
2) Produce a short execution plan mapped to each acceptance bullet.
3) Implement code/docs/tests needed for acceptance closure.
4) Update governance artifacts when required by scope:
   - docs/dev/change_scope.yaml
   - any task-required schema/contract artifacts
5) Run validation commands and report results.
6) Report file-by-file changes and explicit acceptance mapping.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- If pre-P2-09:
  - uv run pytest -m "unit or conformance"
  - uv run pytest -m cross_check   # non-empty required for tasks after P2-00
- If post-P2-09:
  - uv run pytest -m unit
  - uv run pytest -m conformance
  - uv run pytest -m property
  - uv run pytest -m regression
  - uv run pytest -m cross_check

Additional required selectors by change type:
- Bug fix: run relevant regression selectors now (no deferral).
- Determinism/ordering/sentinel/partial-sweep changes: run conformance + regression selectors.
- Cross-check/tolerance changes: run cross_check + regression selectors.
- Property-domain logic changes: run property selectors.
- New orientation logic: add/update sign-convention tests.
- New topology checks: add deterministic witness tests.

Output format (strict):
1) Implementation summary (what changed and why)
2) Acceptance checklist for {{TASK_ID}} with pass/fail evidence per bullet
3) Files changed (path + purpose)
4) Test commands executed + pass/fail
5) Governance impact statement:
   - Frozen artifact IDs touched (or "none")
   - Required evidence added (or reason not required)
6) Residual risks / follow-ups (must be empty for completed scope, otherwise explicit blockers)

If blocked:
- Stop and report the exact blocking conflict, affected files, and the minimum policy-compliant resolution path.
```

## Adaptation Notes

- Keep one prompt per task/PR. Do not batch multiple `P2-XX` tasks in one execution prompt.
- Copy the exact acceptance bullets for `{{TASK_ID}}` into the execution plan before edits.
- Prefer selector-targeted test runs that prove the task’s acceptance criteria, then run the mandatory baseline commands.

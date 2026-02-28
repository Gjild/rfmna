# Phase 2 Task Executor Prompts (Per Task)

This document provides one ready-to-use executor prompt for each Phase 2 task (`P2-00` to `P2-11`).
Each section is cleanly separated and pre-filled with task ID/title/phase context.

Use these prompts as-is or with minimal edits to branch/task metadata.

## P2-00 — Phase 2 gate + freeze-boundary verification

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-00
- Task title: Phase 2 gate + freeze-boundary verification
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-00)

Task-specific focus:
- Implement governance gate, change-scope machine artifact checks, and threshold/tolerance classification enforcement.
- Bootstrap mandatory cross_check marker/strictness/lane with non-empty collection.
- Bootstrap regression scaffold and smoke selector.

Execution contract:
1) Implement only P2-00 scope/acceptance criteria.
2) If conflicts appear against higher-priority artifacts, stop and align.
3) Preserve deterministic behavior and all frozen-contract constraints.
4) For frozen impacts, require semver bump + DR + migration note + conformance updates + reproducibility impact statement.

Required workflow:
1) Read P2-00 Goal/Scope/Deliverables/Acceptance.
2) Produce a short execution plan mapped to acceptance bullets.
3) Implement code/docs/tests for acceptance closure.
4) Update governance artifacts required by P2-00.
5) Run validation commands and report results.
6) Report file-by-file changes and explicit acceptance mapping.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check

Output format (strict):
1) Implementation summary
2) Acceptance checklist for P2-00 with pass/fail evidence per bullet
3) Files changed (path + purpose)
4) Test commands executed + pass/fail
5) Governance impact statement (frozen IDs touched or none; evidence added)
6) Residual risks / blockers
```

## P2-01 — Backend conditioning controls: real scaling and pivot behavior

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-01
- Task title: Backend conditioning controls: real scaling and pivot behavior
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-01)

Task-specific focus:
- Make scaling/pivot controls materially active in backend solve path.
- Preserve sparse unsymmetric solve class, deterministic attempt metadata, and sentinel/fail semantics.
- Add conformance + regression evidence for corrected fallback-stage control flow.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-01 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- plus relevant regression selectors for behavior changes

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-02 — Fallback ladder execution hardening in sweep and RF API contexts

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-02
- Task title: Fallback ladder execution hardening in sweep and RF API contexts
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-02)

Task-specific focus:
- Ensure node_voltage_count propagation for eligible MNA-system solves.
- Enforce no gmin regularization in algebraic conversion-math solves.
- Lock solver_config_snapshot + attempt-trace schema/default behavior and warning propagation parity in RF paths.
- Keep run exit semantics intact unless full frozen governance evidence is provided.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-02 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- plus relevant regression selectors for determinism/sentinel/partial-sweep/fallback behavior

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-03 — Diagnostics taxonomy closure and canonical catalog completion

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-03
- Task title: Diagnostics taxonomy closure and canonical catalog completion
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-03)

Task-specific focus:
- Complete Track A runtime diagnostic catalog/inventory CI enforcement.
- Implement Track B non-diagnostic typed error-code registry + mandatory mapping matrix + CI guard.
- Preserve deterministic diagnostics ordering and required metadata fields.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-03 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- plus relevant regression selectors for bug-fix paths

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-04 — Diagnostic emission normalization across modules

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-04
- Task title: Diagnostic emission normalization across modules
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-04)

Task-specific focus:
- Refactor emission plumbing to canonical builders and deterministic ordering.
- Preserve behavior defined by P2-02/P2-03; do not alter semantic warning propagation targets.
- Enforce schema-complete diagnostics payload fields across modules.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-04 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- plus regression selectors where bug behavior is fixed

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-05 — Hardened check command contract

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-05
- Task title: Hardened check command contract
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-05)

Task-specific focus:
- Implement deterministic machine-readable check JSON output while preserving existing text grammar mode.
- Lock check exit-code mapping and deterministic ordering semantics.
- Classify any run/check exit-semantics change against frozen artifact #9 with full governance evidence when applicable.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-05 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- plus conformance/regression selectors for CLI semantics and output grammar

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-06 — Regression suite expansion (golden + tolerance-aware)

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-06
- Task title: Regression suite expansion (golden + tolerance-aware)
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-06)

Task-specific focus:
- Expand structured regression suite from P2-00 scaffold.
- Introduce/lock tolerance baseline with explicit governance classification for normative_gating vs calibration_only.
- Ensure gating tolerances are normative_gating and frozen-governed when required.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-06 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- uv run pytest -m regression

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-07 — Property-based robustness suite

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-07
- Task title: Property-based robustness suite
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-07)

Task-specific focus:
- Add deterministic property tests for solver/diagnostics/sweep invariants.
- Keep witnesses/order/hash behavior stable and reproducible.
- Ensure non-vacuous property lane coverage before merge.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-07 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- uv run pytest -m property

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-08 — Threshold/tolerance calibration + cross-check harness

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-08
- Task title: Threshold/tolerance calibration + cross-check harness
- Phase ordering context: pre-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-08)

Task-specific focus:
- Expand cross_check harness with documented references and deterministic tolerances.
- Reuse P2-00 marker/strictness ownership and verify no drift.
- Apply threshold/tolerance classification policy; prevent calibration_only usage in CI gating decisions.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-08 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- uv run pytest -m regression

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-09 — CI enforcement for unit/conformance/property/regression/cross-check

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-09
- Task title: CI enforcement for unit/conformance/property/regression/cross-check
- Phase ordering context: transition task (builds post-P2-09 enforcement)

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-09)

Task-specific focus:
- Make all five category lanes mandatory and non-empty where required.
- Enforce thread-control deterministic defaults in CI/conformance guards.
- Keep AGENTS governance alignment audit separate from incidental policy edits.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-09 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m "unit or conformance"
- uv run pytest -m cross_check
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-10 — Phase 2 conformance bundle + matrix

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-10
- Task title: Phase 2 conformance bundle + matrix
- Phase ordering context: post-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-10)

Task-specific focus:
- Complete Phase 2 conformance matrix mapping area -> conformance_id/test_id.
- Ensure deterministic evidence bundle and freeze-boundary coverage are explicit.
- Verify matrix traceability and CI pass status.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-10 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```

## P2-11 — Documentation sync (implemented behavior only)

```text
You are Codex executing one authoritative Phase 2 backlog task.

Task selector:
- Task ID: P2-11
- Task title: Documentation sync (implemented behavior only)
- Phase ordering context: post-P2-09

Repository root:
- /home/miba/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase2_backlog.md (task scope and acceptance for P2-11)

Task-specific focus:
- Update docs to implemented behavior only, with no aspirational claims.
- Preserve governance wording precision and frozen-boundary references.
- Keep traceability to conformance/test evidence.

Execution contract and workflow:
- Follow docs/dev/p2_task_executor_prompt.md exactly, scoped to P2-11 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p2_task_executor_prompt.md.
```


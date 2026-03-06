# Phase 3 Task Executor Prompts (Per Task)

This document provides one ready-to-use executor prompt for each Phase 3 task (`P3-00` to `P3-13`).
Each section is cleanly separated and pre-filled with task ID/title/order context.

Use these prompts as-is or with minimal edits to branch/task metadata.

## P3-00 — Phase 3 gate and boundary bootstrap

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-00
- Task title: Phase 3 gate and boundary bootstrap
- Execution-order context: core order #1/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-00)

Task-specific focus:
- Implement Phase 3 gate artifacts and CI integration with inherited Phase 2 blockers intact.
- Implement anti-tamper base-ref governance checks with blocking behavior on missing/unresolvable base-ref.
- Implement path-aware optional-track governance triggers and deterministic freshness policy.
- Keep scope/evidence machine-checkable for both frozen and non-frozen change surfaces.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-00 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus governance/conformance tests for all Phase 3 sub-gates (positive + negative)

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-01 — Canonical design bundle contract and real CLI loader integration

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-01
- Task title: Canonical design bundle contract and real CLI loader integration
- Execution-order context: core order #2/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-01)

Task-specific focus:
- Replace loader stub with deterministic in-repo loader and schema contract.
- Enforce interim-vs-closure criteria distinction for temporary exclusions.
- Lock loader-path non-regression for frozen semantics: port/wave conventions (#3), exit semantics (#9), sentinel/partial-sweep (#11), and frequency grammar impacts (#8) when applicable.
- Ensure taxonomy-complete diagnostics and Track A/Track B governance updates for new failure codes.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-01 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus targeted conformance/regression for loader-driven run/check exit and sentinel invariants

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-02 — Hierarchical grammar support for subcircuits and macros

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-02
- Task title: Hierarchical grammar support for subcircuits and macros
- Execution-order context: core order #3/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-02)

Task-specific focus:
- Add deterministic hierarchy grammar/schema support with explicit diagnostics.
- Enforce explicit schema-evolution path and compatibility checks (v2 path or strict additive-v1 path).
- Add permutation-invariance and deterministic parse-product conformance evidence.
- Keep Track A/Track B failure-code governance complete.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-02 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-03 — Deterministic hierarchy elaboration engine

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-03
- Task title: Deterministic hierarchy elaboration engine
- Execution-order context: core order #4/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-03)

Task-specific focus:
- Implement deterministic hierarchy elaboration to canonical IR.
- Lock canonical instance-path naming/collision behavior and witness stability.
- Add explicit non-regression evidence for strict two-stage assembly pattern reuse under hierarchy-enabled topology-stable sweeps.
- Preserve frozen IR serialization/hash governance boundary (#7).

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-03 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus conformance/regression proving no per-point pattern recompilation in topology-stable sweeps

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-05 — Model-card contract, schema, and registry

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-05
- Task title: Model-card contract, schema, and registry
- Execution-order context: core order #5/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-05)

Task-specific focus:
- Define model-card schema/contract and deterministic registry resolution behavior.
- Bind model-card provenance into manifest/hash paths.
- Enforce diagnostics completeness and Track A/Track B governance for new failures.
- Classify impacts against frozen artifacts #7/#8/#9/#10 where applicable.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-05 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-04 — Hierarchical parameter scoping and override precedence lock

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-04
- Task title: Hierarchical parameter scoping and override precedence lock
- Execution-order context: core order #6/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-04)

Task-specific focus:
- Finalize precedence chain and ensure legacy file-vs-CLI/API compatibility.
- Enforce conflict/cycle diagnostics with taxonomy-complete fields and deterministic ordering/witness policy.
- Update Track A/Track B governance artifacts when new failure codes are introduced.
- Preserve frozen-boundary governance if precedence changes imply frozen impacts.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-04 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-06 — Practical RF model-card implementations and numeric policy

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-06
- Task title: Practical RF model-card implementations and numeric policy
- Execution-order context: core order #7/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-06)

Task-specific focus:
- Implement required model-card families with deterministic numeric policies.
- Enforce out-of-domain diagnostics and no hidden regularization.
- Add regression/cross_check/conformance evidence per required card family.
- Preserve two-stage assembly pattern reuse for topology-stable model-card sweeps.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-06 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-07 — CLI/API productivity surfaces for composed designs

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-07
- Task title: CLI/API productivity surfaces for composed designs
- Execution-order context: core order #8/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-07)

Task-specific focus:
- Implement deterministic CLI/API surfaces for load/check/run/inspect composition workflows.
- Produce required surface matrix artifact with schema and conformance mapping.
- Preserve existing run/check contracts unless explicitly frozen-governed.
- Classify machine-readable surface changes against frozen #9/#10 (and #8 if applicable).

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-07 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus conformance tests that each surface-matrix row resolves to executable tests

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-08 — Notebook/report template system (deterministic)

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-08
- Task title: Notebook/report template system (deterministic)
- Execution-order context: core order #9/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-08)

Task-specific focus:
- Implement deterministic template generation paths with fixed encoding/newline/ordering rules.
- Add template render contract and hash-locked fixtures.
- Ensure template command/API/output surfaces are classified against frozen #9/#10 (and #8 when applicable).
- Keep dependencies minimal and policy-compliant.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-08 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-09 — Canonical example-project library and end-to-end workflow harness

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-09
- Task title: Canonical example-project library and end-to-end workflow harness
- Execution-order context: core order #10/14

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-09)

Task-specific focus:
- Implement canonical end-to-end examples aligned to v4 deliverables.
- Add deterministic harness + hash-locked outputs.
- Include at least one intentional failing example (or deterministic negative mode) to exercise diagnostics conformance.
- Ensure failure-path diagnostics are taxonomy-complete and deterministic.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-09 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus conformance checks for intentional failing example path

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-10 — Optional extension track A: CCCS/CCVS (usage-driven)

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-10
- Task title: Optional extension track A: CCCS/CCVS (usage-driven)
- Execution-order context: conditional task #11/14 (usage-driven)

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-10)

Task-specific focus:
- Only execute if optional-track activation is valid and governance checks are satisfied.
- If activated: implement deterministic CCCS/CCVS semantics with sign/orientation conformance.
- If not activated: implement explicit defer evidence path only (no stealth scope expansion).
- Enforce optional-track path-aware gate behavior and activation evidence freshness/approval rules.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-10 only.
- Respect conditional-gate logic: touching optional-track scope requires valid activation evidence.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus sign/orientation conformance selectors

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-11 — Optional extension track B: mutual inductance/coupled inductor support (usage-driven)

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-11
- Task title: Optional extension track B: mutual inductance/coupled inductor support (usage-driven)
- Execution-order context: conditional task #12/14 (usage-driven)

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-11)

Task-specific focus:
- Only execute if optional-track activation is valid and governance checks are satisfied.
- If activated: implement deterministic mutual-inductance semantics with domain/sign conformance and cross-check evidence.
- If not activated: implement explicit defer evidence path only (no stealth scope expansion).
- Enforce optional-track path-aware gate behavior and activation evidence freshness/approval rules.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-11 only.
- Respect conditional-gate logic: touching optional-track scope requires valid activation evidence.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus sign/domain conformance + cross_check selectors

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-12 — Phase 3 conformance bundle + matrix

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-12
- Task title: Phase 3 conformance bundle + matrix
- Execution-order context: core order #13/14 (closure-critical)

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-12)

Task-specific focus:
- Build closure-grade conformance matrix with deterministic ordering and executable nodeid mapping.
- Enforce status-domain policy (`covered` only).
- Include required non-regression areas: loader, hierarchy, assembly-pattern reuse, run exit semantics, sentinel/partial-sweep invariants, diagnostics tracks, CLI/API outputs, templates, examples, governance gates, optional-track evidence.
- Ensure run_exit/sentinel coverage is exercised through loader + hierarchy + model-card example workflows.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-12 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus matrix integrity conformance selectors

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

## P3-13 — Documentation sync and Phase 3 closure package

```text
You are Codex executing one authoritative Phase 3 backlog task.

Task selector:
- Task ID: P3-13
- Task title: Documentation sync and Phase 3 closure package
- Execution-order context: core order #14/14 (closure)

Repository root:
- /home/anon/rfmna

Authoritative sources (in priority order):
1) docs/spec/v4_contract.md
2) docs/spec/* normative appendices and threshold tables
3) AGENTS.md
4) Existing tests/fixtures
5) docs/dev/phase3_backlog.md (task scope and acceptance for P3-13)

Task-specific focus:
- Update docs to implemented behavior only (no aspirational drift).
- Build docs-claim map with schema + CI validation.
- Provide closure evidence including explicit proof that loader temporary exclusions are fully closed.
- Ensure optional-track implemented/deferred status is explicit and evidence-backed.

Execution contract and workflow:
- Follow docs/dev/p3_task_executor_prompt.md exactly, scoped to P3-13 only.

Mandatory validation commands:
- uv run ruff check .
- uv run mypy src
- uv run pytest -m unit
- uv run pytest -m conformance
- uv run pytest -m property
- uv run pytest -m regression
- uv run pytest -m cross_check
- plus docs-claim-map validator/conformance selectors

Output format:
- Use strict output format from docs/dev/p3_task_executor_prompt.md.
```

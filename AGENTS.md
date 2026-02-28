# AGENTS.md — Codex Operating Rules for `rfmna`

This repository implements a deterministic RF MNA solver under a strict v4 contract.
Treat this file as binding implementation policy.

## 1) Priority Order (must follow)

1. `docs/spec/v4_contract.md`
2. Normative appendices and threshold tables in `docs/spec/*`
3. This `AGENTS.md`
4. Existing tests and fixtures
5. Local implementation convenience

If any conflict appears, stop and align with the higher-priority artifact.

---

## 2) Non-negotiable technical constraints

- Core linear system class is **general complex sparse unsymmetric**.
- No symmetry/Hermitian assumptions in solve logic.
- No dense matrix path in solver core.
- No silent regularization in conversions (Y/Z/S or other). Use explicit diagnostics.
- Deterministic behavior is mandatory for indexing, assembly, diagnostics, outputs, and hashes.
- Stamping code must be pure, deterministic, side-effect free with respect to inputs.
- Failed-point sentinel policy is mandatory:
  - complex outputs: `nan + 1j*nan`
  - real outputs: `nan`
  - status: `fail`
  - diagnostics entry required
- Never omit requested sweep points from outputs.

---

## 3) Frozen artifacts that cannot change silently

Any change to the following requires:
(a) semantic version bump,
(b) decision record in `docs/spec/decision_records/`,
(c) conformance updates,
(d) migration note.

Frozen list:
1. Canonical element stamp equations
2. 2-port Y/Z block equations and stamping policy
3. Port voltage/current and wave conventions
4. Residual formula and condition-indicator definition
5. Threshold table values and status bands
6. Retry ladder order/defaults
7. IR serialization/hash rules
8. Frequency grammar and grid generation rules
9. CLI exit semantics and partial-sweep behavior
10. Canonical API data shapes and ordering
11. Fail-point sentinel policy
12. Deterministic thread-control defaults

---

## 4) Required architecture boundaries

Keep module boundaries explicit:

- `parser`: input formats, units, expressions, canonical parse products
- `ir`: immutable canonical model structures, IDs, serialization
- `elements`: stamping + per-element validation
- `assembler`: pattern compile + numeric fill only
- `solver`: backend abstraction, retries, scaling/gmin, residual/cond diagnostics
- `rf_metrics`: port extraction, Y/Z/S, derived RF metrics
- `sweep_engine`: frequency/parameter orchestration, cache policy
- `viz_io`: export/plots/reporting
- `diagnostics`: taxonomy, payload schema, stable ordering

Do not blend parser/model concerns into solver internals.

---

## 5) Determinism rules (strict)

- Stable ordering for:
  - nodes
  - auxiliary unknowns
  - ports
  - sweeps and points
  - traces/output columns
  - diagnostics and witness payloads
- No reliance on unordered iteration for user-visible behavior.
- Locale-independent numeric parsing.
- Deterministic expression evaluation and override precedence (`file < CLI/API override`).
- Canonical serialization for hash generation.
- Threading defaults for reproducible CI/runtime should remain fixed unless intentionally changed.

---

## 6) Diagnostics contract (strict)

All diagnostics must include:

- `code`, `severity`, `message`
- context (`element_id` and/or node/port)
- sweep/frequency context when applicable
- `suggested_action`
- `solver_stage` in `{parse, preflight, assemble, solve, postprocess}`
- deterministic `witness` payload where applicable

Severity semantics:
- Errors block affected path.
- Warnings annotate only; warnings do not alter numerical results.

No ad-hoc string-only diagnostics.

---

## 7) Test policy for every change

Minimum expectations:

- Every behavior change includes tests.
- Math-sensitive changes require conformance tests.
- Determinism-impacting changes require explicit reproducibility tests.
- Bug fixes require regression tests.
- New element/source/control orientation logic requires sign-convention tests.
- New topology checks require deterministic witness tests.

Test categories:
- `unit`
- `property`
- `regression`
- `conformance`
- `cross_check` (mandatory for Phase 2 robustness work; reference-comparison tests with documented tolerances)

Never merge math/core changes with missing tests.

---

## 8) Performance and implementation policy

- Preserve sparse pattern reuse model: compile pattern once, numeric fill per point.
- Avoid premature micro-optimizations that reduce clarity unless benchmark-backed.
- If performance changes behavior/risk, include benchmark evidence and diagnostics impact notes.
- Avoid adding heavy dependencies without explicit request.

---

## 9) Dependency and toolchain policy

- Python baseline: 3.14 (project contract).
- Keep dependency drift minimal.
- Do not replace established numeric backend behavior without explicit request.
- Do not add `scikit-rf` dependency.

---

## 10) Code change style for Codex tasks

When implementing tasks:
- Keep diffs scoped to task boundaries.
- Do not refactor unrelated files.
- Preserve public API unless task explicitly authorizes API changes.
- If assumptions are needed, encode them in tests and document briefly in commit notes.
- Prefer explicit types and small pure functions.

---

## 11) Forbidden shortcuts

- No TODO-based pseudo-implementation in production paths for accepted tasks.
- No hidden fallback that masks failure classification.
- No auto-clamping or silent threshold edits.
- No changing default ladder/threshold order without formal change control.

---

## 12) Commit message convention (recommended)

Use concise conventional style:

- `feat: ...`
- `fix: ...`
- `refactor: ...`
- `test: ...`
- `docs: ...`
- `chore: ...`

Math/normative-impact commits should mention:
`[spec-impact]` and link decision record path.

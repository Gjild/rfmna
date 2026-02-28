# Codex Backlog — Phase 2 (Robustness)

This document is the authoritative implementation backlog for **Phase 2**.
It supersedes ad-hoc task lists for robustness work and is grounded in the current repository state.

Phase 1 is complete and reviewed. Baseline verification snapshot at backlog authoring time (historical, not a live invariant):

- `uv run pytest -q` passed (2026-02-27 historical snapshot)
- backlog-authoring commit anchor: `f0236ef08bf50023da890c1990b04ec41e62c014`
- baseline CI/artifact anchor: workflow `.github/workflows/ci.yml`, artifact `phase1-gate-status` for the authoring commit (or local-equivalent verification record when CI URL is unavailable)
- frozen-artifact governance baseline remains:
  - `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`
  - `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`

---

## 1) Priority and Authority

Global precedence is defined by `AGENTS.md` and must not be redefined here.
For Phase 2 execution within that existing precedence, apply this backlog after `docs/spec/*`, `AGENTS.md`, and existing tests/fixtures.
If any conflict appears, stop and align to the higher-priority artifact set.

---

## 2) Phase 2 Objectives (Authoritative)

Phase 2 scope is derived from the Phase 2 roadmap plus global verification requirements in
`docs/initial_project_description.md` (Phase 2 robustness bullets and Section 8 verification contract, including cross-check harness):

1. Conditioning controls (`gmin`/scaling/retry ladder)
2. Diagnostics taxonomy finalized
3. Regression expansion + tolerance baseline definition
4. Hardened `check` command
5. Full conformance suite enforced in CI
6. Cross-check harness with documented references and tolerance-bounded comparisons

Source anchors:

- Phase 2 robustness bullets: `docs/initial_project_description.md` (Phase 2 section)
- Cross-check harness requirement: `docs/initial_project_description.md` Section 8.4
- Tolerance-table requirement: `docs/initial_project_description.md` Section 8.5

Out-of-scope for Phase 2 (unless an explicit DR changes this):

- CCCS/CCVS and mutual inductance feature expansion
- subcircuits/macros/model-card productivity features (Phase 3)
- nonlinear/transient engines

---

## 3) Current-State Snapshot (Ground Truth)

Implemented and working now:

- fallback ladder artifact load + ordering checks: `src/rfmna/solver/fallback.py`, `tests/unit/test_solver_fallback.py`, `tests/conformance/test_solver_fallback_conformance.py`
- deterministic preflight diagnostics for reference/floating/port/vsource-loop/hard-constraint checks: `src/rfmna/parser/preflight.py`
- deterministic run/check CLI semantics and RF options:
  `src/rfmna/cli/main.py`, `tests/unit/test_cli_semantics.py`, `tests/unit/test_cli_rf_options.py`
- full unit+conformance baseline green in CI:
  `.github/workflows/ci.yml`

Phase-2-critical gaps (must be closed in this backlog):

1. `BackendSolveOptions.scaling_enabled` and `pivot_profile` are currently metadata-level toggles; backend numeric solve path does not materially branch on them (`src/rfmna/solver/backend.py`).
2. MNA-system solve paths (`run_sweep()` and standalone RF extraction APIs) currently default to `solve_linear_system(A, b)` without `node_voltage_count`, so gmin can be skipped (`node_voltage_count_unavailable`) in base sweep and RF extraction computations.
   Conversion-math solves (`convert_y_to_s`, `convert_z_to_s`, and conversion internals) are explicitly out of scope for node-voltage/gmin propagation and must retain no-regularization singularity diagnostics.
3. Diagnostics taxonomy closure is broader than warning/preflight gaps: repository code emits additional parse/assembler/manifest/solver-config codes that must be inventoried and governed to avoid uncataloged diagnostic behavior (`docs/spec/diagnostics_taxonomy_v4_0_0.md`, `src/rfmna/diagnostics/catalog.py`, parser/assembler/viz/solver modules).
4. `rfmna check` is loader-boundary-stubbed (`_load_design_bundle` raises by default), so hardened check behavior is not yet a complete in-repo pipeline (`src/rfmna/cli/main.py`).
5. Test categories currently emphasize `unit` and `conformance`; required `property` and `regression` suites are not yet first-class with CI enforcement.
6. Dedicated cross-check/reference comparison harness and CI lane are not yet implemented as first-class verification artifacts.
7. RF extraction/conversion paths currently risk dropping backend solve warnings (`W_NUM_*`) unless explicitly normalized into deterministic per-point diagnostics across `y`/`z`/`s`/`zin`/`zout` workflows.

---

## 4) Global Constraints for All Phase 2 Tasks

1. No silent frozen-artifact changes.
2. Any frozen-artifact change requires semver bump + DR + migration note + conformance updates + reproducibility impact statement.
3. No dense-path introduction in solver core.
4. No silent regularization or threshold/clamp edits.
5. Deterministic ordering/witness/hash behavior remains strict.
6. No parser/model concerns embedded into solver internals.
7. Every behavior change must include tests in the correct category.
8. Keep diffs task-scoped; avoid unrelated refactors.
9. Preserve CLI exit semantics (`0/1/2`) and additive output grammar.
10. Keep fail-point sentinel policy exact (`nan+1j*nan`, `nan`, `fail`, diagnostic present).
11. Catalog all newly emitted diagnostics before release.
12. Run baseline quality gates for each task:
    - `uv run ruff check .`
    - `uv run mypy src`
    - pre-P2-09 minimum: `uv run pytest -m "unit or conformance"`; for tasks after P2-00, also run mandatory `cross_check` selector (non-empty collection required)
    - post-P2-09 minimum: run the five category selectors explicitly (`unit`, `conformance`, `property`, `regression`, `cross_check`); `uv run pytest -q` is additional coverage, not a substitute
13. `docs/spec/schemas/*` policy:
    - schemas referenced by CLI/API outputs are treated as contract artifacts;
    - adding a new versioned schema file (`*_vN.json`) without modifying active schema selection is additive but requires conformance + compatibility evidence;
    - modifying an existing referenced schema or changing active/default schema version must be classified against frozen artifacts #9/#10 and requires full frozen-artifact governance evidence.

---

## 5) Phase 2 Backlog Tasks

## P2-00 — Phase 2 gate + freeze-boundary verification

**Goal**
Establish a formal Phase 2 gate equivalent to Phase 1 governance.

**Scope**

- Create Phase 2 gate checklist anchored to all 12 frozen artifacts.
- Add CI informational step/artifact for Phase 2 gate status.
- Add blocking governance enforcement check that fails when frozen-artifact-impacting changes lack required governance evidence.
- Define deterministic governance rule table mapping change scopes to required evidence artifacts (semver bump, DR, conformance updates, migration note, reproducibility impact statement), with explicit mapping to frozen-artifact IDs `1..12` from `docs/spec/frozen_artifacts_v4_0_0.md`.
- Require machine-readable change-scope declaration artifact `docs/dev/change_scope.yaml` declaring impacted frozen IDs (`none` or subset of `1..12`) for each change.
- Add deterministic CI validation that fails when the change-scope artifact is missing, schema-invalid, or inconsistent with touched-path detection rules from the governance rule table.
- Define threshold/tolerance governance classification table (artifact-path -> `normative_gating|calibration_only`) covering threshold/status-band/tolerance sources used by regression/cross-check/conformance paths.
- Enforce deterministic CI policy from that table: touching any `normative_gating` threshold/status-band source is merge-blocking unless full frozen-artifact evidence is present.
- Require all CI pass/fail tolerances to originate from `normative_gating` artifacts; `calibration_only` artifacts are non-gating inputs only until explicitly promoted.
- Bootstrap Phase 2 `cross_check` governance early: register marker + strict-marker enforcement + deterministic cross-check lane with non-empty collection, so mandatory category policy is active before later expansion tasks.
- Bootstrap regression scaffolding early so pre-P2-06 regression requirements are deterministic and non-ad-hoc (directory, fixture/schema convention, and at least one deterministic regression smoke test).
- Record Phase 2 process traceability links.

**Deliverables**

- `docs/dev/phase2_gate.md`
- `docs/dev/phase2_process_traceability.md`
- `docs/dev/change_scope.yaml` + schema/policy note
- governance rule-table artifact documenting scope->evidence mapping and detection rules
- threshold/tolerance governance classification artifact (path-level `normative_gating|calibration_only` table + promotion policy note)
- `.github/workflows/ci.yml` informational + blocking governance checks
- conformance/policy test(s) validating blocking governance enforcement behavior for positive and negative cases
- mandatory early `cross_check` artifacts: `pytest.ini` marker/strictness entry, deterministic `cross_check` tests (non-empty collection), CI execution of `cross_check`, and non-empty-lane guard active from P2-00
- regression scaffold artifacts: `tests/regression/` bootstrap structure, fixture/schema note under `docs/dev/`, and deterministic regression smoke test wired into selectors

**Acceptance**

- Gate checklist present in CI logs/artifacts on every run.
- Checklist includes DR/semver/conformance/migration/reproducibility triggers.
- Governance rule-table detection is deterministic for declared change scopes and required evidence artifacts, with direct traceability to frozen-artifact IDs `1..12`.
- Change-scope declaration artifact is mandatory and machine-validated; CI fails on missing/invalid artifact and on mismatch between declared scope and touched-path-derived scope.
- Threshold/tolerance classification table is mandatory; CI fails on missing/invalid classification and on any touch to `normative_gating` threshold/status-band sources without full frozen-artifact evidence.
- CI fails if a merge-gating tolerance source is classified as `calibration_only`.
- CI fails if any frozen threshold/status-band source (including `docs/spec/thresholds_v4_0_0.yaml` and any source mapped to frozen artifact #5) is classified as `calibration_only`.
- Blocking CI/policy check fails when required frozen-artifact governance evidence is missing.
- No frozen-artifact change lands without required governance evidence.
- Governance/policy tests include positive and negative cases that exercise each frozen-artifact ID mapping path.
- `cross_check` marker + strict-marker config are active by end of P2-00, and CI fails if `cross_check` collects zero tests.
- For tasks after P2-00, `cross_check` execution is mandatory (not optional smoke-only coverage), with coverage depth expanded in later tasks.
- Regression scaffold is active by end of P2-00, so P2-01/P2-02 regression requirements run against structured `tests/regression` selectors rather than ad-hoc placement.

**Sub-gates (independently blocking)**

1. Governance gate:
   - change-scope artifact + rule-table + threshold/tolerance classification checks are green,
   - frozen-evidence blocking checks are green for touched frozen paths.
2. Category bootstrap gate:
   - `cross_check` marker/strict-marker configuration is active and audited,
   - CI `cross_check` lane executes with non-empty collection,
   - regression scaffold smoke selector executes from structured `tests/regression`.

Each sub-gate has its own CI pass signal and must pass independently.

---

## P2-01 — Backend conditioning controls: real scaling and pivot behavior

**Goal**
Make scaling and alternative pivot profile real numeric controls, not metadata-only signals.

**Scope**

- Implement deterministic scaling behavior for `scaling_enabled=True` in backend solve path.
- Implement deterministic alternative pivot/permutation profile behavior for `pivot_profile="alt"`.
- Return backend metadata describing effective pivot/scaling mode applied.
- Ensure backend solve metadata exposes deterministic per-attempt control state required for reproducibility snapshots (effective pivot profile, scaling flag, gmin value).
- Preserve sparse unsymmetric solve class and no dense fallback.

**Deliverables**

- `src/rfmna/solver/backend.py` updates
- unit + conformance tests proving controls affect solve path deterministically
- at least one regression test locking stable behavior for corrected fallback-stage control flow

**Acceptance**

- Controlled fixtures include at least one case where baseline fails and a later configured stage succeeds (scaling and/or alt pivot), with deterministic stage ordering.
- Attempt trace and backend metadata prove stage-specific behavior (`pivot_profile`, `scaling_enabled`) is applied as configured.
- Reproducibility-facing metadata needed by manifest snapshots is deterministic and schema-stable under repeated equivalent runs.
- Deterministic results and diagnostics remain stable under permutation-equivalent inputs, using tolerance-bounded numeric assertions where appropriate.
- Backend-control changes do not regress frozen fail semantics when exercised via sweep-facing tests: failed points remain present, sentinel fill is exact, `status=fail`, and diagnostics are emitted.

---

## P2-02 — Fallback ladder execution hardening in sweep and RF API contexts

**Goal**
Ensure ladder stages execute as intended during normal sweeps and standalone RF extraction/S-conversion APIs.

**Scope**

- Pass `node_voltage_count` from sweep layout into default solver invocation.
- Ensure gmin stage is executable (not systematically skipped) for **MNA-system solves**: default sweep runs, sweep-triggered RF extraction payload paths, and standalone RF extraction APIs (`extract_y_parameters`, `extract_z_parameters`, `extract_zin_zout`).
- Treat algebraic conversion solves (`convert_y_to_s`, `convert_z_to_s`, and related conversion internals) as conversion-math paths where gmin regularization is explicitly forbidden; singular/ill-conditioned conversion behavior must remain explicit diagnostics/fail semantics.
- Propagate effective retry/scaling/gmin/pivot controls into run reproducibility artifacts (`solver_config_snapshot`) and include deterministic attempt-trace summary payloads.
- Add explicit tests for stage execution/skip reasons in sweep-driven and standalone RF extraction API solves.
- Propagate `solve_result.warnings` from RF extraction/conversion solves into deterministic structured diagnostics (including frequency/point context) for `extract_y_parameters`, `extract_z_parameters`, `extract_zin_zout`, `convert_y_to_s`, and `convert_z_to_s`.
- Verify degraded/fail retry semantics against contract text; if frozen-artifact semantics change is needed, require semver bump + DR + conformance updates + migration note + reproducibility impact statement.

**Deliverables**

- `src/rfmna/sweep_engine/run.py` updates
- RF solve-path updates in `src/rfmna/rf_metrics/y_params.py`, `src/rfmna/rf_metrics/z_params.py`, `src/rfmna/rf_metrics/s_params.py`, `src/rfmna/rf_metrics/impedance.py`, and/or shared solve wrappers where needed
- targeted updates in `src/rfmna/solver/solve.py` / `src/rfmna/solver/fallback.py` as needed
- run artifact integration updates in `src/rfmna/cli/main.py` and `src/rfmna/viz_io/manifest.py`
- `docs/spec/schemas/solver_repro_snapshot_v1.json` + `docs/dev/solver_repro_snapshot_contract.md` defining `solver_config_snapshot` + attempt-trace summary fields (including default/empty semantics and versioning policy)
- regression/conformance tests for stage execution evidence
- conformance tests locking reproducibility payload key presence, deterministic ordering, and explicit default/empty behavior on runs with and without active fallback controls
- unit + conformance tests for RF warning propagation parity and deterministic ordering

**Acceptance**

- gmin attempts run when applicable for base sweep solves, sweep-triggered RF extraction payload solves, and eligible standalone RF extraction API solves.
- For eligible sweep/RF/standalone extraction paths, tests show no unintended `node_voltage_count_unavailable` skip reason.
- Conversion paths (`convert_y_to_s`, `convert_z_to_s`, conversion internals) do not use gmin regularization; conversion singularity/ill-conditioning remains explicit diagnostic/fail behavior with no silent rescue.
- Manifest `solver_config_snapshot` is schema-stable and present on every run, with effective retry/scaling/gmin/pivot settings serialized via explicit defaults when controls are inactive.
- Deterministic attempt-trace summary fields are schema-stable and present on every run (including explicit empty/default values when no retries/stages are exercised).
- Any change to `solver_config_snapshot` / attempt-trace summary required fields, ordering, or canonical shape is explicitly classified against frozen artifacts #9 (CLI semantics/output behavior) and #10 (canonical API data shapes/ordering); when classified frozen-impacting, full governance evidence is mandatory (semver bump + DR + conformance updates + migration note + reproducibility impact statement).
- attempt traces show deterministic stage transitions with no hidden retries.
- Under retry/failure scenarios, base sweep outputs and RF payloads preserve frozen sentinel/partial-sweep guarantees: no point omission, exact sentinel fill, `status=fail`, and required diagnostics.
- RF extraction/conversion paths preserve solver warning diagnostics (for example `W_NUM_*`) with deterministic ordering and point/frequency context parity to base sweep solve behavior.
- Conformance coverage demonstrates frozen `rfmna run` exit semantics remain intact (`0` all-pass, `1` degraded-only, `2` any-fail/preflight-error) when fallback-controlled solve behavior is exercised.
- Any frozen-artifact contract-interpretation adjustment follows full governance: semver bump + DR + conformance updates + migration note + reproducibility impact statement.

---

## P2-03 — Diagnostics taxonomy closure and canonical catalog completion

**Goal**
Finalize diagnostics taxonomy implementation coverage.

**Scope**

- Add missing required minimum-set taxonomy codes to canonical catalog for runtime-emitted diagnostic paths.
- Include warning-family codes and phase-appropriate runtime parse/preflight/assemble/solve/postprocess diagnostic codes.
- Build repository-wide emitted-code inventory spanning parser, assembler, solver, sweep, `rf_metrics`, CLI, and viz/manifest paths.
  Inventory scope is strictly **structured diagnostics emitted at runtime** (`DiagnosticEvent` and sweep-equivalent typed diagnostic payloads), not internal exception/error constants that are never emitted as diagnostics.
- Build a separate typed non-diagnostic error-code registry for internal exception paths (for example parse/manifest/assembler/config errors) with mandatory mapping matrix to diagnostics taxonomy behavior.
- Define mandatory Track B mapping matrix `docs/dev/typed_error_mapping_matrix.yaml` with explicit families and required mapping mode:
  - `E_PARSE_*` -> `typed_error_only`
  - `E_ASSEMBLER_*` / `E_INDEX_*` -> `typed_error_only` unless explicitly promoted to runtime diagnostic paths
  - `E_MANIFEST_*` -> `typed_error_only`
  - `E_SOLVER_CONFIG_*` -> `typed_error_only` unless explicitly surfaced as runtime diagnostics
  - any family declared `diagnostic_equivalent_required` must include mapped runtime diagnostic code(s)
- Enforce uniqueness and required metadata fields for catalog entries.
- Add CI verification guard that fails on any uncataloged emitted diagnostic code from the inventory.
- Add Track B CI verification guard that fails on uncataloged/duplicate non-diagnostic typed error codes and any registry/matrix violation.

**Deliverables**

- `src/rfmna/diagnostics/catalog.py` updates
- `docs/dev/diagnostic_runtime_code_inventory.yaml`
- `docs/dev/typed_error_code_registry.yaml` + `docs/dev/typed_error_code_mapping_policy.md`
- `docs/dev/typed_error_mapping_matrix.yaml`
- catalog validation tests
- CI/policy guard test(s) failing on uncataloged emitted codes
- CI/policy guard test(s) failing on Track B registry/matrix drift (uncataloged/duplicate/matrix-nonconformant/unmapped-required entries)

**Acceptance**

- Track A (runtime diagnostics): repository-wide inventory of emitted structured diagnostics is complete/deterministic and CI fails on uncataloged emitted diagnostic codes.
- Track B (non-diagnostic typed errors): parse/manifest/assembler/solver-config and related non-diagnostic typed error codes are registry-tracked with deterministic mapping policy and uniqueness checks.
- Track B CI guard is enforced and fails on uncataloged/duplicate typed error codes, matrix-family coverage violations, and any unmapped entry for families marked `diagnostic_equivalent_required`.
- For the minimum taxonomy set in `docs/spec/diagnostics_taxonomy_v4_0_0.md`, all codes expected on runtime diagnostic paths are represented in the diagnostic catalog; non-diagnostic-only codes are covered by the typed error-code registry track.

---

## P2-04 — Diagnostic emission normalization across modules

**Goal**
Eliminate ad-hoc diagnostic construction drift across parser/preflight/solve/postprocess/CLI via catalog-backed construction and shared adapters, without introducing new behavioral semantics.

**Scope**

- Centralize catalog-backed diagnostic creation helpers.
- Ensure required fields (`code`, `severity`, `message`, context, `suggested_action`, `solver_stage`) are consistently populated.
- Ensure frequency/sweep context is included whenever applicable.
- Preserve deterministic witness canonicalization and sort ordering.

**Deliverables**

- `src/rfmna/diagnostics/adapters.py` and call-site updates
- module-level migration from local ad-hoc strings to structured diagnostics
- unit tests for deterministic schema compliance and stable ordering

**Acceptance**

- No string-only ad-hoc diagnostics in parser, preflight, solver, sweep_engine, rf_metrics, and CLI emission paths.
- Behavioral warning-propagation semantics remain as established by P2-02; this task validates structural consistency only.
- Deterministic ordering remains invariant under input permutations.

---

## P2-05 — Hardened `check` command contract

**Goal**
Make `rfmna check` robust, deterministic, and machine-consumable for structural validation workflows.

**Scope**

- Define explicit loader-boundary behavior and typed diagnostics for loader/preflight failures.
- Add machine-readable check output mode (deterministic JSON) while preserving current line grammar mode.
- Define a versioned JSON schema for check output, including required fields and compatibility policy.
- Use a canonical schema location/name for check output (initial target: `docs/spec/schemas/check_output_v1.json`) and validate implementation outputs against it in CI.
- Classify check JSON-contract changes at implementation time as either additive non-frozen surface or frozen-artifact-impacting surface (`CLI`/canonical API shape-order) before merge.
- Lock check exit mapping contract: exit `0` when no error diagnostics are present (warnings-only allowed), and exit `2` on any error diagnostic path including structural/topology violations, loader-boundary failures, and internal failures.
- Any proposed change to `run/check` exit semantics is explicitly classified against frozen artifact #9 and requires full frozen-artifact governance evidence before merge (semver bump + DR + conformance updates + migration note + reproducibility impact statement).
- Add conformance tests for deterministic ordering and witness payload stability in check outputs.
- Add explicit non-regression tests that `rfmna check` text output grammar and ordering remain unchanged when JSON mode is not selected.

**Deliverables**

- `src/rfmna/cli/main.py` updates
- `docs/dev/check_command_contract.md` (new)
- `docs/spec/schemas/check_output_v1.json` (canonical versioned schema path; next versions follow `check_output_vN.json`)
- unit + conformance CLI tests
- CI schema-validation step/policy test for `check` JSON output against canonical schema

**Acceptance**

- `check` outputs are deterministic and machine-parseable.
- JSON schema versioning and required-field validation are test-enforced.
- Canonical schema location/version naming is enforced and CI validates `check` JSON output against the declared schema.
- key ordering and witness determinism are stable in machine-readable output.
- When JSON mode is not selected, `rfmna check` text output grammar and deterministic ordering remain backward-compatible with pre-JSON behavior.
- Check exit mapping is conformance-locked: exit `0` for no-error/warnings-only outcomes; exit `2` for structural/topology errors, loader-boundary failures, and internal failures (no alternate `check` exit codes).
- loader-boundary failures emit typed diagnostics, not opaque failures.
- Change classification is explicit in task evidence: additive non-frozen changes are documented as such; if canonical CLI/API shape-order semantics are altered, full frozen-artifact governance evidence is required (semver bump + DR + conformance updates + migration note + reproducibility impact statement).
- Any `run/check` exit-semantics change is treated as frozen-artifact #9 evaluation path and cannot merge without full governance evidence.

---

## P2-06 — Regression suite expansion (golden + tolerance-aware)

**Goal**
Expand the regression scaffold established in P2-00 into a dedicated regression suite for numeric behavior stability.

**Scope**

- Add `tests/regression` with golden fixtures for core/RF scenarios:
  - RC/RLC sanity
  - controlled-source cases
  - 2-port Y/Z/S consistency where valid
- Define deterministic fixture schema and hash policy.
- Define initial regression tolerance-table baseline used by regression assertions before calibration refinement.
- Ensure failed points still honor sentinel policy in regression fixtures.

**Deliverables**

- `tests/regression/*`
- `tests/fixtures/regression/*`
- regression fixture schema note in `docs/dev/`
- initial regression tolerance-table artifact (versioned and referenced by regression tests)

**Acceptance**

- Regression tests fail on explicit tolerance-table exceedance and/or approved fixture-hash mismatch.
- Golden updates require explicit approval workflow, not silent rewrites.
- Baseline tolerance table is present and consumed by regression tests before P2-08 calibration refinement.
- Threshold/tolerance artifact updates follow the P2-00 classification table: `normative_gating` threshold/status-band sources always require full frozen-artifact governance before merge (semver bump + DR + conformance updates + migration note + reproducibility impact statement); `calibration_only` artifacts are experimental/non-gating only, must not drive CI pass/fail decisions, and may change under standard Phase 2 test/traceability controls only while they remain non-gating.

---

## P2-07 — Property-based robustness suite

**Goal**
Add property testing for invariants that are hard to exhaust via example-based tests.

**Scope**

- Add `tests/property` using Hypothesis for:
  - node relabeling invariance
  - ordering/permutation invariance
  - passive-domain reciprocity sanity (where assumptions hold)
  - conditioning/well-posedness filter correctness
- Fix deterministic execution profile for CI reproducibility.

**Deliverables**

- `tests/property/*`
- pytest/Hypothesis deterministic profile config

**Acceptance**

- Property tests are non-flaky in CI.
- Counterexamples produce deterministic minimal repro artifacts.

---

## P2-08 — Threshold/tolerance calibration + cross-check harness

**Goal**
Refine baseline tolerances with evidence-based calibration and add cross-check validation against reference sources.

**Scope**

- Define calibration datasets and scripts.
- Refine and ratify explicit regression/conversion tolerance tables with provenance (starting from P2-06 baseline).
- Implement `tests/cross_check` harness with documented analytical/external references and deterministic fixtures.
- Reuse marker/strictness ownership from P2-00 bootstrap and verify no drift while expanding cross-check harness coverage.
- Apply P2-00 threshold/tolerance classification policy for all touched tolerance artifacts; any `normative_gating` threshold/default/status-band source requires semver bump + DR + conformance updates + migration note + reproducibility impact statement, and `calibration_only` artifacts remain non-gating until explicit promotion.

**Deliverables**

- calibration scripts under `tools/` or `scripts/`
- calibration report under `docs/dev/`
- `tests/cross_check/*` and `tests/fixtures/cross_check/*`
- cross-check reference-source and tolerance policy doc in `docs/dev/`
- marker/strictness drift-check evidence confirming P2-00 bootstrap contract remains intact
- thresholds/spec governance artifacts when applicable

**Acceptance**

- Tolerance values are justified by reproducible evidence.
- Cross-check harness defines per-metric pass/fail tolerances (magnitude/phase or scalar equivalents) and deterministic fixture references.
- Cross-check failures are triggered by explicit tolerance exceedance, not ad-hoc judgment.
- Platform/backend variability policy is documented and test-enforced.
- `cross_check` marker/strictness settings introduced by P2-00 remain active with no drift while harness coverage is expanded.

---

## P2-09 — CI enforcement for unit/conformance/property/regression/cross-check

**Goal**
Make full verification categories mandatory in CI, not optional.

**Scope**

- Add explicit CI lanes for `unit`, `conformance`, `property`, `regression`, and `cross_check`.
- Define Phase 2 policy note that `cross_check` is an additional mandatory category for this phase (supplementing the base category set).
- Enforce in CI that marker declarations and strict-marker behavior introduced by P2-00 remain active and audited.
- Add non-empty-lane guards so each category lane fails if it collects zero tests.
- Keep deterministic thread/env controls fixed.
- Add CI artifact uploads for calibration/regression/cross-check failure diagnostics.

**Deliverables**

- `.github/workflows/ci.yml` updates
- `pytest.ini` alignment update only if drift from P2-00 marker/strictness bootstrap contract is detected
- marker/selection documentation in `docs/dev/`
- CI selection docs including `cross_check` marker/lane contract
- governance alignment check confirming `AGENTS.md` already states `cross_check` as mandatory for Phase 2; if policy drift is detected, open a separate governance-labeled task/PR (do not bundle AGENTS edits into P2-09 implementation changes)

**Acceptance**

- CI fails on any category failure.
- CI fails on unknown/undeclared markers.
- Each category lane (`unit`, `conformance`, `property`, `regression`, `cross_check`) fails when zero tests are collected.
- Category coverage (`unit`, `conformance`, `property`, `regression`, `cross_check`) is explicit and auditable in workflow logs.
- CI explicitly asserts frozen thread-control defaults (artifact #12) and runs thread-controls conformance guard as part of mandatory verification lanes.
- `AGENTS.md` alignment is audited; if drift is found, a separate governance-labeled task/PR is required and P2-09 records the dependency.

---

## P2-10 — Phase 2 conformance bundle + matrix

**Goal**
Freeze and verify Phase 2 robustness semantics with explicit conformance mapping.

**Scope**

- Add conformance tests for:
  - real scaling/pivot behavior
  - sweep-integrated and standalone-RF fallback ladder execution
  - `rfmna run` exit-semantic preservation (`0/1/2`) under fallback-sensitive `pass`/`degraded`/`fail` outcomes
  - diagnostics taxonomy closure and ordering
  - frozen sentinel/partial-sweep invariants under retry/failure (base sweep outputs + RF payloads)
  - hardened check command contract
  - CI category enforcement expectations
- Add Phase 2 conformance matrix mapping areas -> `conformance_id` and `test_id`.
- Define matrix columns explicitly (`area`, `conformance_id`, `test_id`, `status`, `notes`).
- Add coverage report doc.

**Deliverables**

- `tests/conformance/test_phase2_conformance_matrix.py`
- `docs/dev/phase2_conformance_coverage.md`

**Acceptance**

- All matrix entries resolve to executable tests.
- Normative and governance-critical behavior is explicitly auditable.

---

## P2-11 — Documentation sync (implemented behavior only)

**Goal**
Update documentation to reflect shipped Phase 2 behavior without aspirational claims.

**Scope**

- Update README status and validation commands.
- Add Phase 2 usage/limits documentation.
- Document diagnostics catalog changes, check contract, regression/property workflow, and CI lanes.

**Deliverables**

- `README.md` updates
- `docs/dev/phase2_usage.md`

**Acceptance**

- Every documented behavior has executable evidence in repository tests.
- No “planned but not implemented” content in implemented-surface docs.

---

## 6) Execution Order

1. P2-00
2. P2-01
3. P2-02
4. P2-03
5. P2-04
6. P2-05
7. P2-06
8. P2-07
9. P2-08
10. P2-09
11. P2-10
12. P2-11

---

## 7) Phase 2 Exit Criteria

Phase 2 is complete only when all are true:

1. Conditioning controls are implemented and conformance-covered (not wiring-only).
2. Reproducibility artifacts include effective retry/scaling/gmin/pivot controls and deterministic attempt-trace summary fields where applicable.
3. Diagnostics taxonomy minimum set is cataloged and runtime-emission-validated.
4. `check` command is hardened with deterministic machine-consumable output.
5. Regression, property, and cross-check suites are implemented and enforced in CI.
6. Phase 2 conformance matrix and coverage report are present and passing.
7. All governance requirements are met for any frozen-artifact changes.

---

## 8) Per-Task Runbook (mandatory)

For each P2 task:

1. Modify only scoped files.
2. Add/adjust tests for acceptance criteria.
3. Do not modify normative docs unless task explicitly includes semver bump + DR + conformance updates + migration note + reproducibility impact statement.
4. Record assumptions in task/PR notes.
5. Run:
   - `uv run ruff check .`
   - `uv run mypy src`
   - pre-P2-09 minimum: `uv run pytest -m "unit or conformance"`; for tasks after P2-00, also run mandatory `cross_check` selector (non-empty collection required)
   - post-P2-09 minimum: run all five category selectors (`unit`, `conformance`, `property`, `regression`, `cross_check`); `uv run pytest -q` is additional, not a substitute
6. Pre-P2-09 task-type requirements (in addition to pre-P2-09 minimum):
   - bug fix: run relevant `regression` selectors now (do not defer to P2-09).
   - determinism/ordering/sentinel/partial-sweep behavior change: run relevant `conformance` and `regression` selectors.
   - cross-check/tolerance change: run relevant `cross_check` and `regression` selectors.
   - property-domain logic change: run relevant `property` selectors.
7. Non-vacuous selector rule (pre-P2-09): when a task requires `property`, `regression`, or `cross_check` selectors, that category must collect >=1 test in the task branch (existing or newly added).
8. No deferral for mandatory categories on governed changes: for bug fixes, math/core behavior changes, determinism/sentinel/ordering changes, and cross-check/tolerance changes, missing required selector evidence is merge-blocking.
9. Determinism-impacting task requirement: add or update explicit reproducibility assertions (repeat-run stability for ordering/hash/manifest fields as applicable), not only selector execution.
10. Orientation-logic trigger: tasks introducing new source/control orientation/sign behavior must add or update explicit sign-convention tests.
11. Topology-check trigger: tasks introducing new topology/structural checks must add deterministic witness tests that lock witness ordering/content stability.

# Codex Backlog — Phase 3 (Personal Productivity)

This document is the authoritative implementation backlog for **Phase 3**.
It supersedes ad-hoc task lists for Phase 3 productivity work and is grounded in the current repository state.

Phase 2 is complete and governance-closed. Baseline snapshot at backlog authoring time (historical evidence, not a live invariant):

- package version: `0.1.2` (`pyproject.toml`, `src/rfmna/__init__.py`)
- backlog-authoring commit anchor: `00dc0647ad775d696ac1d34341b5baf449665c02`
- baseline CI/artifact anchor: workflow `.github/workflows/ci.yml`, artifact `phase2-gate-status` for the authoring commit (or local-equivalent verification record when CI URL is unavailable)
- historical pre-closure baseline verification reference only: `uv run pytest -q` passed (`2026-02-27` snapshot recorded in `docs/dev/phase2_backlog.md`)
- Phase 2 closure evidence pointers (authoritative for closure status): `docs/dev/phase2_gate.md`, `docs/dev/phase2_conformance_coverage.md` (evaluated at the backlog-authoring commit anchor)
- Date clarity: `2026-02-27` evidence above is historical pre-closure context only; closure-authoritative Phase 2 evidence is the gate/conformance state at backlog anchor commit `00dc0647ad775d696ac1d34341b5baf449665c02`.
- current status docs:
  - `docs/dev/phase2_usage.md`
  - `docs/dev/phase2_conformance_coverage.md`
- baseline governance anchors:
  - `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`
  - `docs/spec/decision_records/2026-02-28-p2-02-fallback-ladder-rf-hardening.md`
  - `docs/spec/decision_records/2026-02-28-p2-06-regression-golden-tolerance-baseline-v0-1-2.md`
  - `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`
  - `docs/spec/migration_notes/2026-02-28-v0-1-1-p2-02-fallback-ladder-rf-hardening.md`
  - `docs/spec/migration_notes/2026-02-28-v0-1-2-p2-06-regression-golden-tolerance-baseline.md`

---

## 1) Priority and Authority

Global precedence is defined by `AGENTS.md` and is not redefined here.
For Phase 3 execution within that precedence, apply this backlog after `docs/spec/*`, `AGENTS.md`, and existing tests/fixtures.
If conflict appears, stop and align to the higher-priority artifact set.

---

## 2) Phase 3 Objectives (Authoritative)

Phase 3 scope is derived from `docs/initial_project_description.md`:

1. Personal productivity workflow completion (Section 13: Phase 3)
2. Subcircuits/macros as first-class deterministic model-composition tools
3. Practical RF model cards for reusable linear AC blocks
4. Notebook/report templates for fast, repeatable analysis workflows
5. Optional usage-driven expansion track: CCCS/CCVS and/or mutual inductance
6. End-to-end example workflows aligned with documented v4 deliverables

Source anchors:

- Phase 3 roadmap bullets: `docs/initial_project_description.md` (Section 13)
- v4 deliverables and examples: `docs/initial_project_description.md` (Section 14)
- v4 acceptance criteria and change control: `docs/initial_project_description.md` (Sections 16, 19)

Out-of-scope for Phase 3 unless separately approved via governance:

- nonlinear operating-point/transient engines
- full SPICE-compatibility goals
- cloud/collaboration/auth/billing concerns

---

## 3) Current-State Snapshot (Ground Truth)

Implemented and stable now:

- deterministic sparse unsymmetric AC core with fallback/repro diagnostics
- RF extraction/conversion (`y`, `z`, `s`, `zin`, `zout`) with deterministic warning propagation
- hardened `rfmna check` text/json contract with schema lock
- full CI category enforcement (`unit`, `conformance`, `property`, `regression`, `cross_check`)
- frozen-artifact governance gate and threshold/tolerance classification controls

Phase-3-critical gaps to close:

1. CLI design loading remains loader-boundary stubbed (`src/rfmna/cli/main.py::_load_design_bundle`), so no in-repo end-to-end design execution path exists.
2. There is no canonical in-repo design schema/grammar for hierarchical composition (subcircuits/macros).
3. There is no deterministic hierarchy elaboration layer that flattens composed designs into canonical IR with stable IDs/hashes.
4. Practical RF model-card infrastructure (schema, registry, deterministic resolution and binding) is not implemented.
5. Notebook/report template workflow and canonical example project set from the v4 deliverables are not implemented as executable in-repo assets.
6. Optional CCCS/CCVS and mutual inductance tracks remain explicitly deferred from Phase 2 and require governed handling if activated.

---

## 4) Global Constraints for All Phase 3 Tasks

1. No silent frozen-artifact changes.
2. Any frozen-artifact change requires semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
3. No dense-path introduction in solver core.
4. No silent regularization, auto-clamping, or threshold edits.
5. Deterministic ordering/witness/hash behavior remains strict.
6. Keep architecture boundaries explicit (`parser`, `ir`, `elements`, `assembler`, `solver`, `rf_metrics`, `sweep_engine`, `viz_io`, `diagnostics`).
7. Every behavior change must include tests in the correct category.
8. Keep diffs task-scoped; avoid unrelated refactors.
9. Preserve run/check exit semantics unless full frozen governance evidence is provided.
10. Keep fail-point sentinel policy exact and complete (no point omission).
11. Continue Track A/Track B diagnostic governance: no uncataloged runtime diagnostic codes, no uncataloged typed internal codes.
12. Preserve deterministic thread-control defaults and associated conformance guards.
13. Hierarchical elaboration must be deterministic and pure: canonical instance path naming, stable expansion order, no order dependence on unordered iteration.
14. Parameter-resolution precedence for composed designs must be fixed, documented, and conformance-locked.
15. Model-card resolution/interpolation/extrapolation behavior must be deterministic and explicitly diagnostic on invalid domains.
16. Example/report/template outputs used for gating must be reproducible and hash-stable under identical inputs/configs.
17. Continue mandatory verification categories as repository-wide CI lanes for all Phase 3 work:
   - `unit`
   - `conformance`
   - `property`
   - `regression`
   - `cross_check`
   Per-task test additions are required for affected behavior domains per the runbook (not every task must add tests in every category).
18. Every new failure code introduced by Phase 3 must be explicitly classified as Track A runtime diagnostic (catalog + runtime inventory) or Track B typed internal error (registry + mapping matrix), with CI guards for uncataloged and mis-mapped failures.
19. Preserve strict two-stage assembly behavior for topology-stable sweeps: compile sparsity pattern/index maps once and reuse across points, with numeric fill per point only (no per-point pattern recompilation).

---

## 5) Phase 3 Backlog Tasks

## P3-00 — Phase 3 gate and boundary bootstrap

**Goal**
Establish a formal Phase 3 gate equivalent in rigor to the Phase 2 governance gate.

**Scope**

- Create Phase 3 gate checklist with explicit traceability to frozen artifacts and Phase 3 productivity scope.
- Keep existing Phase 2 governance checks active and non-regressed, with explicit blocking parity requirements.
- Add Phase 3 process traceability record and CI informational artifact.
- Add blocking Phase 3 policy checks for mandatory evidence on any new contract/schema surfaces introduced by Phase 3.
- Require anti-tamper governance evaluation: Phase 3 rule/classification checks must evaluate against baseline refs (base-ref), not mutable head-branch edits.
- Define deterministic scope-to-evidence mapping for new non-frozen Phase 3 contract surfaces using machine-checkable artifacts and CI enforcement (no prose-only interpretation paths).
- Define deterministic optional-track activation criteria for usage-driven scope toggles (minimum evidence type, freshness window, approval rule, and CI-verifiable pass/fail logic).
- Define deterministic optional-track freshness contract: `usage_evidence_date` format, UTC basis, and canonical comparison timestamp source for CI decisions (no locale/timezone-dependent logic).
- Define deterministic optional-track scope detection rules: path-pattern matching against base-ref diff determines when optional-track governance checks are mandatory even if activation is not declared.

**Deliverables**

- `docs/dev/phase3_gate.md`
- `docs/dev/phase3_process_traceability.md`
- CI updates in `.github/workflows/ci.yml` for Phase 3 informational + blocking gate signals
- executable governance checker interface: `python -m rfmna.governance.phase3_gate` with deterministic sub-gate selection via `--sub-gate`
- updates to `docs/dev/frozen_change_governance_rules.yaml` path mappings when Phase 3 introduces authoritative files/schemas that affect frozen IDs
- `docs/dev/phase3_contract_surface_governance_rules.yaml` (scope->evidence mapping + path-detection rules for new non-frozen contract surfaces)
- `docs/dev/phase3_change_surface.yaml` + `docs/dev/phase3_change_surface_schema_v1.json` + `docs/dev/phase3_change_surface_policy.md` (machine-checkable declaration + policy)
- `docs/dev/optional_track_activation.yaml` + `docs/dev/optional_track_activation_schema_v1.json` (fixed-field activation evidence for usage-driven optional tracks)
- `docs/dev/optional_track_activation_policy.md` (objective activation criteria, freshness window, approval rule, CI decision logic, deterministic time contract: `usage_evidence_date` in `YYYY-MM-DD` UTC compared against base-ref commit timestamp projected to UTC date, and deterministic path-scope trigger rules)
- conformance tests for Phase 3 gate expectations, including non-regression pass/fail cases for inherited Phase 2 blocking gates
- conformance/policy tests proving head-branch edits to Phase 3 governance rule artifacts cannot self-approve without corresponding baseline-ref evidence
- conformance/policy tests (positive + negative) for new Phase 3 surface-governance declarations and scope/evidence mapping enforcement

**Acceptance**

- Phase 3 gate status appears on every CI run.
- A canonical executable Phase 3 gate checker is defined and used by CI: `python -m rfmna.governance.phase3_gate` (with deterministic `--sub-gate` behavior and pass/fail semantics).
- Phase 2 blocking gates remain mandatory and non-weakened:
  - `python -m rfmna.governance.phase2_gate --sub-gate governance`
  - `python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap`
  - `docs/dev/change_scope.yaml` presence/schema and frozen-ID detection enforcement
  - threshold/tolerance classification enforcement from `docs/dev/threshold_tolerance_classification.yaml`
  - mandatory category lanes (`unit`, `conformance`, `property`, `regression`, `cross_check`) with strict markers and non-empty guards
- New Phase 3 contract/surface additions are blocked when required evidence is missing.
- Conformance tests prove inherited Phase 2 blockers still fail in negative cases (missing evidence, invalid scope declaration, missing/empty category lane conditions).
- Phase 3 governance checks are anti-tamper by construction: rule/classification evaluation uses baseline refs, and negative tests prove mutable head-only edits cannot relax blocking outcomes.
- Base-ref resolution is a blocking prerequisite for governance checks: missing/ambiguous/unresolvable base-ref context fails CI (no permissive fallback to head-only evaluation).
- Conformance tests cover the canonical Phase 3 gate command interface with positive/negative fixtures for each independently blocking sub-gate.
- When Phase 3 adds authoritative files/schemas that map to frozen artifacts, `docs/dev/frozen_change_governance_rules.yaml` detection paths are updated and conformance-tested so frozen-scope detection remains complete.
- New non-frozen Phase 3 contract-surface changes are declared through machine-checkable artifacts and blocked in CI when declaration/schema/path-detection/scope-evidence checks fail.
- Cross-consistency is blocking and conformance-tested:
  - `docs/dev/change_scope.yaml` remains authoritative for frozen-ID declaration,
  - `docs/dev/phase3_change_surface.yaml` declares only non-frozen Phase 3 contract surfaces,
  - overlap or mismatch between the two declaration paths fails CI.
- When either optional track (P3-10/P3-11) is activated, or optional-track scope is touched, `docs/dev/optional_track_activation.yaml` presence/schema validity is blocking in CI via the conditional optional-track sub-gate.
- Optional-track activation decisions are objectively gateable and conformance-tested against `docs/dev/optional_track_activation_policy.md`:
  - minimum evidence type requirements are satisfied,
  - evidence freshness window is satisfied using deterministic UTC date logic (`usage_evidence_date` `YYYY-MM-DD` UTC compared to base-ref commit UTC date),
  - explicit approval rule is satisfied,
  - positive and negative decision fixtures (including boundary-date cases, optional-track-scope-touched-without-activation cases, and missing/unresolvable base-ref cases) produce deterministic CI pass/fail outcomes.

**Sub-gates (independently blocking)**

1. Inherited governance non-regression gate:
   - `python -m rfmna.governance.phase2_gate --sub-gate governance` and `python -m rfmna.governance.phase2_gate --sub-gate category-bootstrap` remain enforced and blocking.
2. Phase 3 contract-surface governance gate:
   - `phase3_contract_surface_governance_rules.yaml` + `phase3_change_surface.yaml` declaration/schema/path-detection checks are green.
3. Anti-tamper gate:
   - baseline-ref evaluation is active, and head-branch edits cannot self-approve governance relaxations.
   - base-ref missing/unresolvable/ambiguous states are independently blocking.
4. Optional-track activation gate (conditional):
   - activation predicate is deterministic and path-aware: the gate is active iff either optional track is declared `activated` in `docs/dev/optional_track_activation.yaml` OR base-ref diff path matching indicates optional-track scope is touched;
   - when active, optional-track schema/policy/decision checks are independently blocking, including freshness-window evaluation against base-ref commit UTC date per `docs/dev/optional_track_activation_policy.md`;
   - touching optional-track scope without valid activation evidence is blocking (CI fail).

---

## P3-01 — Canonical design bundle contract and real CLI loader integration

**Goal**
Replace loader-boundary stub behavior with an in-repo deterministic design bundle loading contract.

**Scope**

- Define versioned design-bundle schema(s) for `rfmna check`/`rfmna run` input.
- Implement deterministic loader from design path to `CliDesignBundle`.
- Add structured diagnostics for schema/loader/validation failures.
- Preserve existing run/check exit mappings and deterministic output ordering.

**Deliverables**

- loader implementation in `src/rfmna/cli/main.py` and dedicated parser/loader module(s)
- canonical schema artifact `docs/spec/schemas/design_bundle_v1.json` and explicit active/default version-selection policy for future `design_bundle_vN.json` additions
- `docs/dev/design_bundle_contract.md`
- optional governed temporary exclusion list artifact `docs/dev/p3_loader_temporary_exclusions.yaml` + schema `docs/dev/p3_loader_temporary_exclusions_schema_v1.json` (required only if any v4 in-scope loader capability is deferred)
- unit + conformance tests for valid/invalid loader paths and deterministic diagnostics

**Acceptance**

- Interim P3-01 merge criteria (temporary exclusions allowed): `rfmna check <design>` and `rfmna run <design> --analysis ac` execute through in-repo loader for supported schema inputs, with conformance IDs and regression fixtures for implemented capability paths.
- Temporary deferral of currently in-scope v4 loader capabilities is allowed only during interim implementation and must be explicitly listed in `docs/dev/p3_loader_temporary_exclusions.yaml` with deterministic check/run diagnostics per exclusion, conformance evidence, and governance classification; omission is merge-blocking while exclusions are present.
- Phase 3 closure criteria for P3-01 (no exclusions): full contract-parity with currently in-scope v4 inputs is mandatory and conformance-covered for both `check` and `run` paths: `R/L/C/G`, independent `I/V` sources, controlled sources `VCCS/VCVS`, frequency-dependent compact linear forms, 1-port/2-port `Y` blocks, 1-port/2-port `Z` blocks, RF port declarations, linear/log frequency sweep grammar, and parameter sweep support.
- Phase 3 closure criteria for P3-01 (no exclusions): `docs/dev/p3_loader_temporary_exclusions.yaml` must be absent or schema-valid empty before closure approval.
- Loader failures emit typed deterministic diagnostics (no opaque boundaries).
- Existing CLI exit semantics and check JSON schema behavior are preserved unless separately governed.
- Any change to design-bundle schema version selection, active/default schema behavior, or ordering semantics is explicitly classified against frozen artifacts `#9/#10`; if the schema touches frequency grammar or sweep-generation semantics, classification against frozen artifact `#8` is also mandatory; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
- Loader handling of RF port declarations includes explicit frozen artifact `#3` (port/wave conventions) impact determination with conformance non-regression tests through loader-driven `check/run` paths; if impacted, full frozen governance evidence is mandatory.
- Loader-driven run paths preserve frozen fail-point semantics (artifact `#11`): no requested point omission, exact sentinel fill (`nan + 1j*nan` for complex outputs, `nan` for real outputs), `status=fail`, and mandatory diagnostics entries; conformance tests lock these invariants through loader-backed execution.
- Loader/schema failure codes are Track-classified (`Track A` or `Track B`) with CI negative tests for uncataloged runtime diagnostics and mis-mapped typed codes.
- Loader diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.
- P3-01 explicitly records frozen artifact `#8` impact determination with non-regression conformance evidence, including explicit “no semantic change” evidence when applicable.

---

## P3-02 — Hierarchical grammar support for subcircuits and macros

**Goal**
Introduce deterministic parser support for hierarchical composition constructs.

**Scope**

- Add schema/grammar constructs for subcircuit definitions, macro/template definitions, and instantiation.
- Enforce deterministic identifier normalization and duplicate/conflict handling.
- Add explicit diagnostics for undefined references, duplicate definitions, and illegal recursion declarations.

**Deliverables**

- parser/schema updates for hierarchical declarations
- explicit schema-evolution decision artifact for hierarchy support: `docs/dev/p3_02_design_bundle_schema_evolution.md` choosing one path:
  - (a) introduce `docs/spec/schemas/design_bundle_v2.json`, or
  - (b) additive extension of `design_bundle_v1.json`
- schema-evolution compatibility checker/tests for the selected path (v2 compatibility/default-selection/migration checks, or strict additive-v1 diff-policy checks)
- diagnostics catalog and inventory updates for new runtime emission codes
- Track B governance artifact updates (`docs/dev/typed_error_code_registry.yaml` and `docs/dev/typed_error_mapping_matrix.yaml`) whenever new typed internal failures are introduced
- unit + property tests for permutation invariance and deterministic parse products
- conformance tests with named IDs locking canonical parse-product determinism (ordering/hash stability) and deterministic diagnostics ordering/witness stability for hierarchy grammar paths

**Acceptance**

- Equivalent declaration permutations produce identical canonical parse products.
- Undefined/duplicate/illegal declarations fail with deterministic structured diagnostics.
- No parser-path behavior depends on unordered map/set iteration.
- Any hierarchical grammar/schema default-selection or ordering-surface change is explicitly classified against frozen artifacts `#9/#10`; if grammar/schema updates touch frequency grammar or sweep-generation semantics, classification against frozen artifact `#8` is also mandatory; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
- New hierarchy parse/elaboration failure codes are Track-classified (`Track A` or `Track B`) with CI negative tests for uncataloged and mis-mapped codes.
- Hierarchy grammar diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.
- Named conformance IDs for P3-02 canonical parse-product determinism and diagnostics ordering/witness stability are present and mapped to executable tests.
- Schema-evolution path is explicit and test-enforced:
  - if path (a): `design_bundle_v2.json` compatibility/default-selection/migration tests are present;
  - if path (b): strict additive-only v1 evolution is enforced by conformance/policy tests (no removed fields, no tightened validation constraints for existing fields, no changed defaults/implicit semantics for existing fields), additive-v1 backward-compatibility tests are present, and frozen-impact classification is explicit.

---

## P3-03 — Deterministic hierarchy elaboration engine

**Goal**
Compile hierarchical designs into canonical flattened IR deterministically.

**Scope**

- Implement subcircuit/macro elaboration into immutable canonical IR suitable for existing assembler/solver pipeline.
- Define canonical hierarchical instance-path naming rules and collision policy.
- Add deterministic witness payloads for elaboration failures (cycles, unresolved references, illegal port bindings).
- Ensure aux-unknown allocation policy remains deterministic after elaboration.

**Deliverables**

- elaboration implementation with primary ownership in `src/rfmna/ir/`; parser integration limited to deterministic parse-to-elaboration handoff adapters in `src/rfmna/parser/`
- conformance tests for canonical elaboration ordering and witness stability
- regression fixtures proving canonical IR hash stability for equivalent hierarchical inputs

**Acceptance**

- Equivalent hierarchical designs elaborate to identical canonical IR ordering/hashes.
- Elaboration failures include deterministic structured diagnostics and witnesses.
- Downstream assembly/solve behavior remains contract-compliant and deterministic.
- Hierarchy-enabled topology-stable sweeps preserve strict two-stage assembly behavior: compiled sparsity pattern/index maps are reused across points, and only numeric fill varies per point (no per-point pattern recompilation), enforced by conformance/regression evidence.
- Elaboration diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.
- New elaboration failure codes are Track-classified (`Track A` or `Track B`) with CI negative tests for uncataloged and mis-mapped codes.
- Any change affecting canonical IR serialization or hash semantics is explicitly classified against frozen artifact `#7`; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.

---

## P3-04 — Hierarchical parameter scoping and override precedence lock

**Goal**
Finalize deterministic parameter binding semantics across hierarchy and runtime overrides.

**Scope**

- Define and implement one normative precedence chain (locked): `model_card_defaults < design_file_defaults < subcircuit_definition_defaults < subcircuit_instance_overrides < CLI_API_overrides`.
- Keep locale-independent numeric parsing and deterministic expression evaluation.
- Add cycle/conflict detection for parameter dependency graphs in hierarchical contexts.

**Deliverables**

- parameter-resolution policy doc: `docs/dev/parameter_resolution_policy.md`
- parser/IR binding updates
- unit + property + conformance tests for precedence invariants and cycle diagnostics

**Acceptance**

- Parameter resolution is deterministic and reproducible under declaration permutations.
- Conflicts/cycles emit explicit `E_MODEL_*` diagnostics with deterministic witness payloads.
- P3-04 diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.
- New P3-04 failure codes are Track-classified (`Track A` or `Track B`) with required catalog/inventory or typed-registry/mapping updates and CI negative tests for uncataloged/mis-mapped codes.
- Override precedence is documented and test-locked with explicit conformance IDs and one-to-one tests for each precedence edge.
- The locked precedence chain is a refinement of existing `file < CLI/API override` semantics: legacy file-vs-CLI/API outcomes remain unchanged unless governed as frozen-impacting.
- Conformance tests explicitly prove legacy precedence compatibility; if any legacy outcome changes, frozen-artifact governance evidence is mandatory before merge.
- `docs/dev/parameter_resolution_policy.md` is referenced by `docs/dev/phase3_gate.md` during P3-04 implementation; linkage into `docs/dev/phase3_conformance_coverage.md` is finalized at/after P3-12 (interim traceability in P3-04 notes or gate docs is acceptable).

---

## P3-05 — Model-card contract, schema, and registry

**Goal**
Create deterministic practical RF model-card infrastructure.

**Scope**

- Define versioned model-card schema(s) with allowed linear AC model classes.
- Implement model-card registry loading, validation, normalization, and canonical resolution by ID/version.
- Enforce explicit diagnostics for invalid units/domains/parameter omissions.
- Bind model-card artifacts into reproducibility manifests/hashes.

**Deliverables**

- canonical schema artifact `docs/spec/schemas/model_card_v1.json` and explicit active/default version-selection policy for future `model_card_vN.json` additions
- `docs/dev/model_card_contract.md`
- registry/loader code in parser/IR modules
- tests for schema conformance, deterministic ordering, and hash stability

**Acceptance**

- Model-card load/resolve is deterministic for equivalent inputs.
- Invalid cards fail with structured diagnostics and actionable suggestions.
- Manifest/hash metadata includes stable model-card provenance fields.
- Any model-card schema version/default-selection/order-surface change is explicitly classified against frozen artifacts `#9/#10`; if model-card schema updates touch frequency grammar or sweep-generation semantics, classification against frozen artifact `#8` is also mandatory; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
- Model-card validation/resolve failure codes are Track-classified (`Track A` or `Track B`) with CI negative tests for uncataloged and mis-mapped codes.
- Any change that alters canonical IR serialization/hash behavior through model-card integration is explicitly classified against frozen artifact `#7` with full frozen governance evidence when impacted.
- Model-card registry diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.

---

## P3-06 — Practical RF model-card implementations and numeric policy

**Goal**
Implement first-party practical linear RF card families with explicit numeric behavior.

**Scope**

- Implement a minimum required card set:
  - `mc_one_port_shunt_rc` (analytic reference eligible),
  - `mc_one_port_series_rlc` (analytic reference eligible),
  - `mc_two_port_passive_y_table` (tabulated frequency-domain matrix card with deterministic interpolation policy).
- Define deterministic interpolation/extrapolation and validity-domain diagnostics for table-backed cards where applicable.
- Add cross-check/regression fixtures with tolerance-table-driven assertions.
- Prohibit hidden regularization in model-card conversion/extraction paths.

**Deliverables**

- model-card implementation in `src/rfmna/parser/`, `src/rfmna/ir/`, and `src/rfmna/elements/` with deterministic assembler handoff integration; `rf_metrics` remains a consumer of solved outputs and does not own model-card definition/resolution logic
- regression and cross-check fixture sets for model-card scenarios
- tolerance source updates/classification evidence if gating thresholds are touched

**Acceptance**

- Card outputs are deterministic and tolerance-bounded against references.
- Out-of-domain/invalid card usage produces explicit diagnostics (no silent fallback).
- Gating tolerances remain classification-compliant (`normative_gating` vs `calibration_only`).
- Minimum evidence per required card family is present:
  - at least one regression golden fixture,
  - at least one cross-check reference fixture,
  - at least one invalid-domain diagnostic fixture.
- Conformance coverage is mandatory for each required card family, with clause-mapped tests for equation semantics, interpolation/extrapolation policy, and diagnostics behavior.
- Model-card-driven sweeps preserve strict two-stage assembly behavior for topology-stable cases: compiled sparsity pattern/index maps are reused across points, with per-point numeric fill only.
- Model-card runtime diagnostics are taxonomy-complete and deterministic: required fields (`code`, `severity`, `message`, context, sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, deterministic `witness`) and canonical ordering are enforced by conformance tests.

---

## P3-07 — CLI/API productivity surfaces for composed designs

**Goal**
Expose hierarchy/model-card workflows via stable CLI/API entry points.

**Scope**

- Add deterministic CLI/API entry points for loading, checking, running, and inspecting composed designs.
- Add optional deterministic introspection surfaces (for example elaborated IR export) as additive outputs only.
- Preserve backward compatibility and existing output contracts unless explicitly governed.
- Define a required CLI/API surface matrix (surface name, arguments, output schema ID, conformance ID, test ID) with at least one deterministic surface row for each operation class: `load`, `check`, `run`, and `inspect`.

**Deliverables**

- CLI/API updates in `src/rfmna/cli/` and relevant API modules
- contract note for any new machine-readable outputs/schemas
- required surface matrix artifact: `docs/dev/p3_07_surface_matrix.yaml` + schema `docs/dev/p3_07_surface_matrix_schema_v1.json`
- unit + conformance tests for ordering, schema validity, and non-regression

**Acceptance**

- New productivity commands are deterministic and machine-consumable with versioned schema IDs for any machine-readable outputs.
- Every newly introduced machine-readable CLI/API surface is explicitly classified at introduction against frozen artifact `#10` (and `#9` where CLI output/exit semantics are implicated), with either full frozen-governance evidence or documented non-frozen additive rationale plus conformance coverage.
- Existing run/check contracts remain intact unless governed as frozen-impacting.
- Added output surfaces are schema-versioned and test-enforced.
- Machine-readable outputs lock deterministic key ordering and witness canonicalization under permutation tests.
- Output fixtures used for contract validation are hash-locked with explicit approval workflow for updates.
- Any output-schema active/default version or ordering change is explicitly classified against frozen artifacts `#9/#10`; if output-schema changes alter frequency grammar or sweep-generation semantics, classification against frozen artifact `#8` is also mandatory; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
- The required surface matrix is complete and non-vacuous: each matrix row resolves to at least one executable conformance test, and the matrix covers `load`, `check`, `run`, and `inspect` operation classes.

---

## P3-08 — Notebook/report template system (deterministic)

**Goal**
Provide reusable, deterministic analysis templates for personal workflow acceleration.

**Scope**

- Add report/notebook templates for common RF workflows (sweep summary, pass/fail diagnostics, Y/Z/S views).
- Add deterministic template rendering workflow (inputs, file naming, metadata ordering, encoding policy).
- Keep dependency footprint minimal and aligned with project policy.

**Deliverables**

- template/rendering implementation with primary ownership in `src/rfmna/viz_io/`; static template assets in `docs/`; invocation wrappers in `scripts/`
- `docs/dev/template_render_contract.md` defining deterministic encoding/newline/order/metadata rules and fixture/hash policy
- template rendering command(s) or API helper(s)
- regression tests for deterministic rendered outputs

**Acceptance**

- Template generation is reproducible and deterministic across repeated runs with canonical output ordering and fixed encoding/newline policy.
- Generated artifacts include manifest references and deterministic metadata keys.
- No aspirational template claims without executable evidence.
- Template output fixtures are hash-locked; updates require explicit approval workflow (no silent rewrites).
- CI category selectors include non-vacuous `regression` coverage for template outputs and `conformance` checks for any template schema/output contract behavior.
- Template determinism assertions are validated against `docs/dev/template_render_contract.md` in conformance tests.
- Every newly introduced template command/API/machine-readable output surface is explicitly classified at introduction against frozen artifact `#10` (and `#9` where CLI output/exit semantics are implicated); if template output/schema changes touch frequency grammar or sweep-generation semantics, classification against frozen artifact `#8` is also mandatory; frozen-impacting changes require semver bump + decision record + conformance updates + migration note + reproducibility impact statement.

---

## P3-09 — Canonical example-project library and end-to-end workflow harness

**Goal**
Ship executable Phase 3 examples that exercise full personal productivity workflows.

**Scope**

- Add canonical examples aligned with v4 deliverables:
  - L-match network
  - narrowband RLC filter
  - passive 2-port with S export
- Add deterministic harness/scripts for running examples and validating outputs.
- Add regression fixtures/hash locks for example outputs.

**Deliverables**

- `examples/` project assets (schema-valid designs and expected output manifests)
- test harness under `tests/regression/`, with contract-level assertions in `tests/conformance/` when output-schema/governance behavior is exercised
- at least one deterministic failing example path (or explicit negative harness mode) for failure-diagnostics contract validation
- docs note for example workflow execution and expected artifacts

**Acceptance**

- All canonical examples execute end-to-end via in-repo CLI/API workflows.
- Output ordering/hashes are deterministic under identical inputs/configs.
- Example failures provide structured diagnostics fully compliant with `docs/spec/diagnostics_taxonomy_v4_0_0.md`: required `code`, `severity`, `message`, context (`element_id` and/or node/port), sweep/frequency context when applicable, `suggested_action`, valid `solver_stage`, and deterministic witness payload assertions.
- Example output fixtures/manifests are hash-locked; updates require explicit approval workflow and CI enforcement (no silent fixture/hash rewrites).
- Failure-path coverage is non-vacuous: at least one intentional failing example (or deterministic negative harness mode) is executed in CI with conformance assertions for taxonomy-complete diagnostics and fail-path contract behavior.

---

## P3-10 — Optional extension track A: CCCS/CCVS (usage-driven)

**Goal**
Provide optional controlled-source expansion for current-controlled forms when justified by usage.

**Scope**

- Add CCCS/CCVS modeling/stamping/validation paths.
- Add deterministic orientation/sign convention tests.
- Update normative stamp artifacts and conformance mapping if this track is activated.

**Deliverables**

- element/parser/factory updates for CCCS/CCVS
- conformance tests with clause mapping for sign/orientation
- required governance artifacts when frozen surfaces are impacted
- optional-track activation evidence update in `docs/dev/optional_track_activation.yaml` when this track is enabled

**Acceptance**

- CCCS/CCVS behavior is deterministic and mathematically clause-mapped.
- No frozen-artifact updates merge without full governance evidence.
- If not activated, this task is explicitly deferred with rationale in Phase 3 closure notes.
- If activated, the activation PR must declare impacted frozen IDs in `docs/dev/change_scope.yaml`, provide schema-valid `docs/dev/optional_track_activation.yaml` with fixed fields (`usage_evidence_source`, `usage_evidence_date`, `activation_rationale`, `impacted_frozen_ids`, `approval_record`), and pass all active blocking gates from P3-00 (Phase 2 inherited governance/category gates, Phase 3 contract-surface governance gate, and anti-tamper gate).
- If activated, `usage_evidence_date` must be `YYYY-MM-DD` UTC, and freshness-window evaluation must use the deterministic base-ref commit UTC date contract from `docs/dev/optional_track_activation_policy.md`.

---

## P3-11 — Optional extension track B: mutual inductance/coupled inductor support (usage-driven)

**Goal**
Provide optional mutual-inductance support under deterministic and governed behavior.

**Scope**

- Add mutual-inductance representation, stamping, and validation policy for linear AC domain.
- Add explicit diagnostics for invalid coupling coefficients/domains.
- Add cross-check/regression scenarios for coupled inductor behavior.

**Deliverables**

- element/parser updates for mutual inductance
- conformance and cross-check coverage for coupling equations and sign conventions
- required governance artifacts when frozen surfaces are impacted
- optional-track activation evidence update in `docs/dev/optional_track_activation.yaml` when this track is enabled

**Acceptance**

- Coupled-inductor behavior is deterministic and test-covered for sign/domain correctness.
- No silent regularization or hidden stabilization paths are introduced.
- If not activated, this task is explicitly deferred with rationale in Phase 3 closure notes.
- If activated, the activation PR must declare impacted frozen IDs in `docs/dev/change_scope.yaml`, provide schema-valid `docs/dev/optional_track_activation.yaml` with fixed fields (`usage_evidence_source`, `usage_evidence_date`, `activation_rationale`, `impacted_frozen_ids`, `approval_record`), and pass all active blocking gates from P3-00 (Phase 2 inherited governance/category gates, Phase 3 contract-surface governance gate, and anti-tamper gate).
- If activated, `usage_evidence_date` must be `YYYY-MM-DD` UTC, and freshness-window evaluation must use the deterministic base-ref commit UTC date contract from `docs/dev/optional_track_activation_policy.md`.

---

## P3-12 — Phase 3 conformance bundle + matrix

**Goal**
Freeze and audit Phase 3 productivity semantics with explicit conformance traceability.

**Scope**

- Add conformance matrix mapping Phase 3 areas -> conformance IDs -> executable tests.
- Cover loader contract, hierarchy elaboration/scoping, assembly pattern-reuse non-regression, model-card registry/behavior, template determinism, and example workflows.
- Include governance-critical checks for any new schema/version contracts.
- Require explicit matrix areas for:
  - `loader_contract`
  - `hierarchy_grammar`
  - `hierarchy_elaboration`
  - `assembly_pattern_reuse_non_regression`
  - `run_exit_semantics_non_regression`
  - `sentinel_partial_sweep_non_regression`
  - `parameter_precedence_compatibility`
  - `model_card_registry_contract`
  - `model_card_numeric_policy`
  - `diagnostics_track_a_inventory_non_regression`
  - `diagnostics_track_b_mapping_non_regression`
  - `cli_api_productivity_outputs`
  - `template_determinism`
  - `example_e2e_workflows`
  - `phase3_governance_gate_non_regression`
  - `optional_track_activation_or_defer_evidence`
- Lock matrix contract columns and ordering policy to Phase 2 standard:
  - required columns: `area`, `conformance_id`, `test_id`, `status`, `notes`
  - deterministic row ordering and executable nodeid resolution are mandatory and test-enforced
- Lock matrix status policy to Phase 2 closure strictness:
  - allowed `status` value is `covered` only
  - conformance tests enforce status-domain validity (`covered` only) and fail on `planned|pending|partial|deferred` or any other value

**Deliverables**

- `tests/conformance/test_phase3_conformance_matrix.py`
- `docs/dev/phase3_conformance_coverage.md`

**Acceptance**

- All matrix entries resolve to executable tests.
- Normative/governance-critical Phase 3 behavior is auditable.
- Deterministic ordering and witness stability are explicitly locked by tests.
- Matrix includes explicit evidence entries for:
  - each required core area key (`loader_contract`, `hierarchy_grammar`, `hierarchy_elaboration`, `assembly_pattern_reuse_non_regression`, `run_exit_semantics_non_regression`, `sentinel_partial_sweep_non_regression`, `parameter_precedence_compatibility`, `model_card_registry_contract`, `model_card_numeric_policy`, `diagnostics_track_a_inventory_non_regression`, `diagnostics_track_b_mapping_non_regression`, `cli_api_productivity_outputs`, `template_determinism`, `example_e2e_workflows`),
  - inherited Phase 2 blocking-governance non-regression in Phase 3 CI,
  - optional-track state handling (activated tracks must show full evidence; deferred tracks must show explicit defer record checks).
- Conformance tests enforce required matrix columns, deterministic row ordering, and executable nodeid resolution for all listed `test_id` entries.
- Conformance tests enforce `status == covered` for every matrix row.
- Conformance coverage for `run_exit_semantics_non_regression` and `sentinel_partial_sweep_non_regression` is exercised through loader + hierarchy + model-card example workflows (not isolated unit-only paths).

---

## P3-13 — Documentation sync and Phase 3 closure package

**Goal**
Publish implemented-surface documentation for Phase 3 without aspirational drift.

**Scope**

- Update README status and executable commands.
- Add `docs/dev/phase3_usage.md` with implemented limits and evidence links.
- Add closure summary for optional-track disposition (implemented vs deferred).

**Deliverables**

- `README.md` updates
- `docs/dev/phase3_usage.md`
- closure notes in `docs/dev/` tying implemented claims to tests
- `docs/dev/phase3_docs_claim_map.yaml` (claim->test nodeid map)
- `docs/dev/phase3_docs_claim_map_schema_v1.json` + validator test/conformance check

**Acceptance**

- Every documented implemented-surface claim is listed in `docs/dev/phase3_docs_claim_map.yaml` (nodeid-resolved), schema-valid against `docs/dev/phase3_docs_claim_map_schema_v1.json`, and validated by CI.
- No “planned but not implemented” statements in implemented-surface docs.
- Optional-track status is explicit and evidence-backed.
- Phase 3 closure evidence includes CI proof that loader temporary exclusions are fully closed (`docs/dev/p3_loader_temporary_exclusions.yaml` absent or schema-valid empty), matching Section 7 exit criterion #9.

---

## 6) Execution Order

1. P3-00
2. P3-01
3. P3-02
4. P3-03
5. P3-05
6. P3-04
7. P3-06
8. P3-07
9. P3-08
10. P3-09
11. P3-10 (conditional, usage-driven)
12. P3-11 (conditional, usage-driven)
13. P3-12
14. P3-13

Notes:

- P3-10/P3-11 are conditional tracks; core Phase 3 closure does not require both unless explicitly activated.
- Default state is deferred for both conditional tracks; activation requires explicit `docs/dev/change_scope.yaml` declaration, schema-valid `docs/dev/optional_track_activation.yaml`, and policy compliance from `docs/dev/optional_track_activation_policy.md`; touching optional-track implementation scope without valid activation evidence is blocking via the optional-track sub-gate.
- P3-03 intentionally precedes P3-05: hierarchy elaboration and assembly-pattern reuse invariants are locked first, so later model-card integration can be validated as non-regressive against the same compile-once/numeric-fill-per-point contract.
- P3-05 intentionally precedes P3-04 because parameter-precedence lock in P3-04 depends on model-card default semantics established in P3-05.
- P3-03 precedes P3-04 so hierarchy elaboration semantics (canonical instance paths, flattening, collision policy, structural diagnostics) are implemented before hierarchy-aware parameter binding lock and precedence conformance.
- P3-04 finalizes precedence on top of elaborated hierarchy outputs and must explicitly classify any resulting canonical IR/hash semantic changes against frozen artifact `#7`.
- If either conditional track is activated, its full governance and conformance obligations become mandatory before closure.

---

## 7) Phase 3 Exit Criteria

Phase 3 is complete only when all core criteria are true:

1. In-repo deterministic design loader has full Phase 3 closure parity with currently in-scope v4 inputs from P3-01 (`R/L/C/G`, independent `I/V` sources, controlled sources `VCCS/VCVS`, frequency-dependent compact linear forms, 1-port/2-port `Y` blocks, 1-port/2-port `Z` blocks, RF port declarations, linear/log frequency sweep grammar, parameter sweep support), with conformance-matrix evidence rows mapped to executable tests for each listed capability path.
2. Hierarchical composition (subcircuits/macros) is implemented with deterministic elaboration and canonical IR output.
3. Hierarchical parameter scoping/override precedence is documented and conformance-locked.
4. Model-card registry + practical model-card support is implemented with deterministic diagnostics and provenance.
5. Notebook/report templates and canonical example workflows are executable and reproducible.
6. Phase 3 conformance matrix and coverage report are present and passing.
7. Existing Phase 2 governance and CI category enforcement remain intact.
8. Any frozen-artifact changes introduced in Phase 3 are fully governed.
9. Loader temporary exclusions are fully closed at Phase 3 closure: `docs/dev/p3_loader_temporary_exclusions.yaml` is absent or schema-valid empty; no remaining in-scope v4 loader exclusions are permitted at closure.

Conditional criteria (only if activated):

10. CCCS/CCVS track is fully implemented and governed.
11. Mutual inductance track is fully implemented and governed.

---

## 8) Per-Task Runbook (mandatory)

For each P3 task:

1. Modify only scoped files.
2. Add/adjust tests for acceptance criteria.
3. Do not modify normative docs unless task explicitly includes semver bump + decision record + conformance updates + migration note + reproducibility impact statement.
4. Record assumptions in task/PR notes.
5. Run baseline quality gates:
   - `uv run ruff check .`
   - `uv run mypy src`
   - `uv run pytest -m unit`
   - `uv run pytest -m conformance`
   - `uv run pytest -m property`
   - `uv run pytest -m regression`
   - `uv run pytest -m cross_check`
6. Non-vacuous selector rule: all five category selectors (`unit`, `conformance`, `property`, `regression`, `cross_check`) must remain non-vacuous in repository CI, and tasks that change behavior in a category domain must add or adjust tests in that category (not rely only on unrelated existing tests).
7. Determinism-impacting changes must include explicit reproducibility assertions (ordering/hash/manifest stability) beyond simple execution.
8. New orientation/sign behavior must include explicit sign-convention tests.
9. New topology/structural checks must include deterministic witness tests.
10. Schema/output-surface changes must include compatibility-policy checks and schema validation tests.
11. For schema/output-surface changes, explicitly classify impact against frozen artifacts `#7/#8/#9/#10` as applicable (`#7` for canonical IR serialization/hash semantics, `#8` when frequency grammar/sweep semantics are touched, `#9` for CLI/exit/output behavior, `#10` for canonical API shape/order); if frozen-impacting, add semver bump + decision record + conformance updates + migration note + reproducibility impact statement before merge.

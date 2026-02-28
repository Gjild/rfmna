# Codex Backlog — Phase 1 (RF Utility)

This backlog assumes **Phase 0 is complete and frozen**.
Execution order is strict. Any frozen-artifact-impacting change requires decision record workflow per `AGENTS.md`.

---

## Global constraints for all Phase 1 tasks

1. No unapproved normative/spec edits. Approved normative/spec edits require DR + migration note + conformance evidence.
2. No dense-path introduction in solver/assembly/RF conversion.
3. No silent regularization/clamping.
4. Deterministic ordering for all user-visible outputs, diagnostics, and witness payloads.
5. No IR serialization/hash drift for Phase 0-supported models unless separately approved.
6. Every new diagnostic must include full schema fields and deterministic witness shape.
7. Keep diffs task-scoped; no opportunistic refactors.
8. Reuse existing canonical sort policy (`diagnostics.sort.diagnostic_sort_key`) wherever possible; do not fork sort semantics per module.
9. Preserve CLI exit semantics (`0/1/2`) and existing POINT/DIAG line grammar; RF additions must be additive.
10. All new diagnostic codes must be registered in a canonical diagnostics catalog with uniqueness checks.
11. Reproducibility policy:

    * ordering/witness/metadata determinism is strict equality;
    * numeric result reproducibility is tolerance-based across different platforms/backends unless otherwise specified.
12. For tasks claiming “normative exactness,” tests must map to explicit conformance IDs tied to normative clauses (formula/sign/orientation), not only fixture-level assertions.

---

## P1-00 — Phase gate + freeze verification

**Goal**
Lock Phase 1 baseline and verify frozen-artifact boundaries.

**Scope**

* Re-run existing unit + conformance with no semantic change.
* Add explicit gate checklist referencing all 12 frozen artifacts.
* Add CI informational gate output (non-blocking allowed) that always emits phase/freeze checklist status in logs.

**Deliverables**

* `docs/dev/phase1_gate.md` (new)
* CI update in `.github/workflows/ci.yml` for informational gate output (non-blocking permitted)

**Acceptance**

* `uv run pytest -m "unit or conformance"` passes.
* No unapproved normative/spec edits.
* Approved normative/spec edits require DR + migration note + conformance evidence.
* Approved baseline publication for Phase 1 is tracked by `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md` and `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`.
* Checklist includes DR triggers, semver requirements, and reproducibility-impact statement requirement.
* CI run always surfaces gate checklist status in logs/artifacts.

---

## P1-01 — Controlled source stamps (VCCS, VCVS)

**Goal**
Implement canonical VCCS/VCVS stamps with deterministic ordering and validation.

**Scope**

* Add `VCCSStamp`, `VCVSStamp` in dedicated module.
* VCVS requires aux current unknown.
* Validation emits:

  * `E_MODEL_VCCS_INVALID`
  * `E_MODEL_VCVS_INVALID`
* Deterministic `touched_indices`, footprint, and stamp emission order.
* Preserve pure stamp behavior (side-effect free).
* Add conformance ID mapping for all sign/orientation rules referenced from normative appendix.

**Deliverables**

* `src/rfmna/elements/controlled.py` (new)
* `src/rfmna/elements/__init__.py` export updates
* Unit + conformance tests (equations, orientation matrix, invalid params)
* Conformance ID → test mapping table for controlled source rules

**Acceptance**

* Equations/signs exactly match `stamp_appendix_v4_0_0.md` through clause-mapped conformance tests.
* Full polarity/orientation matrix passes.
* No hidden fallback or dense path.
* No duplicate local sort policy; uses existing canonicalization helpers from `elements.base`.

---

## P1-02 — IR kind normalization + aux allocation policy

**Goal**
Define deterministic normalization from IR declarations to constructor-ready element models.

**Scope**

* Canonical kind mapping: `R,C,G,L,I,V,VCCS,VCVS`.
* Alias/case behavior fixed: either canonicalize deterministically or reject with explicit code.
* Deterministic aux allocation order for `L`, `V`, `VCVS`.
* Unknown kind emits stable structured error + witness.
* **Important:** keep existing `CanonicalIR` serialization stable for Phase 0-supported inputs.
* **Authority rule:** aux allocation policy is owned exclusively by normalization layer; downstream factory must not allocate/reorder aux unknowns.

**Deliverables**

* `src/rfmna/ir/models.py` (additive hooks only)
* `src/rfmna/elements/factory_models.py` (new; normalized internal models)
* Tests for equivalent IR permutations and aux ordering
* Regression tests: unchanged canonical JSON/hash for Phase 0 element set

**Acceptance**

* Equivalent IR permutations produce identical normalized model sequence.
* Unknown kind failure has stable code/message/witness schema.
* No canonical IR serialization/hash drift for Phase 0-supported inputs.
* Aux allocation invariants proven by tests and not duplicated downstream.

---

## P1-03 — Deterministic element factory/registry

**Goal**
Create parser→IR→stamp construction pipeline for Phase 1 element set.

**Scope**

* Registry: normalized kind → constructor.
* Supported kinds: `R,C,G,L,I,V,VCCS,VCVS`.
* Construction order equals canonical IR element order.
* Validation/constructor failures preserved for diagnostics adapter (no lossy stringification).
* Factory remains allocation-free with respect to aux unknown policy (consumes normalized models as-is).

**Deliverables**

* `src/rfmna/elements/factory.py` (new)
* Unit tests for ordering, unknown kind, validation propagation, and no aux-policy duplication

**Acceptance**

* Stamp list order stable and canonical.
* Unknown kind fails with deterministic code/witness.
* Validation payloads are machine-mappable (no string-only errors).
* Factory does not introduce aux allocation/reordering behavior.

---

## P1-04 — ValidationIssue → DiagnosticEvent adapter

**Goal**
Deterministic mapping of element validation failures into project diagnostics schema.

**Scope**

* Adapter utility for `ValidationIssue -> DiagnosticEvent`.
* Preserve `solver_stage`, severity, context, suggested_action.
* Stable witness key ordering and event sorting invariants.
* Ensure adapter produces diagnostics that pass existing strict `DiagnosticEvent` model constraints.
* Register all newly introduced Phase 1 diagnostic codes in canonical diagnostics catalog.

**Deliverables**

* `src/rfmna/diagnostics/adapters.py` (new)
* Diagnostics catalog update (canonical code registry)
* Unit tests for schema compliance, deterministic ordering, and code uniqueness checks

**Acceptance**

* All current and new element validators map to schema-valid diagnostics.
* Mapping deterministic under input permutations.
* Construction path emits no ad-hoc string errors.
* New codes are cataloged uniquely with required schema contract metadata.

---

## P1-05 — Port declaration canonicalization + preflight hardening

**Goal**
Enforce deterministic, strict port preflight.

**Scope**

* Enforce unique `port_id`.
* Enforce orientation uniqueness `(p_plus,p_minus)` (unless future DR changes policy).
* Reject unknown/degenerate declarations deterministically.
* Prepare deterministic witness payloads.
* Z0 model/API checks:

  * complex -> `E_MODEL_PORT_Z0_COMPLEX`
  * non-finite or <=0 -> `E_MODEL_PORT_Z0_NONPOSITIVE`
* Keep hard-invalid ports as errors (never warnings).

**Deliverables**

* `src/rfmna/parser/preflight.py` updates
* Optional additive checks in `src/rfmna/ir/models.py`
* Unit tests for duplicates/degenerate/unknown/permutation stability

**Acceptance**

* `E_TOPO_PORT_INVALID` deterministic across permutations.
* Witness payload key ordering stable.
* Hard-invalid ports are errors, not warnings.

---

## P1-06 — RF boundary-condition engine

**Goal**
Reusable deterministic boundary injection for RF extraction paths.

**Scope**

* Helpers for imposed port voltages/currents and inactive-port boundaries.
* Preserve current sign: positive into DUT.
* Deterministic inconsistent/singular boundary diagnostics (`E_TOPO_*` / `E_NUM_*`).
* Explicit API returns both boundary-augmented system artifacts and deterministic metadata for diagnostics.
* Add conformance ID mapping for boundary/sign conventions from normative references.

**Deliverables**

* `src/rfmna/rf_metrics/boundary.py` (new)
* Unit tests for 1-port/2-port, inconsistent and singular fixtures
* Conformance ID → test mapping for boundary sign/orientation rules

**Acceptance**

* Deterministic independent of dict/set iteration.
* Explicit diagnostic on inconsistent systems.
* Sign/orientation invariants covered by clause-mapped tests.

---

## P1-07 — Y-parameter extraction core

**Goal**
Deterministic Y extraction via voltage excitation.

**Scope**

* Column-wise extraction through boundary engine.
* 1-port and 2-port support in Phase 1.
* Output shape `[n_points, n_ports, n_ports]`.
* Fail-point policy: retain points; complex-NaN sentinels on failed points.
* Deterministic per-column ordering by canonical port order.
* Enforce shared RF sentinel contract (see “RF sentinel contract” below).

**Deliverables**

* `src/rfmna/rf_metrics/y_params.py` (new)
* Unit + conformance fixtures vs analytical references

**Acceptance**

* Conventions match `port_wave_conventions_v4_0_0.md` through clause-mapped tests.
* Failed points retained with sentinels + diagnostics.
* Port ordering stable under declaration permutations.

---

## P1-08 — Z-parameter extraction core + singularity gates

**Goal**
Deterministic Z extraction via current excitation with strict failure behavior.

**Scope**

* Direct column-wise Z extraction.
* Optional Y→Z conversion only when explicitly configured.
* Emit:

  * `E_NUM_ZBLOCK_SINGULAR`
  * `E_NUM_ZBLOCK_ILL_CONDITIONED`
* No silent regularization.
* Enforce shared RF sentinel contract (see “RF sentinel contract” below).

**Deliverables**

* `src/rfmna/rf_metrics/z_params.py` (new)
* Unit + conformance tests for direct and failing paths

**Acceptance**

* Ordering/sign conventions preserved.
* Singular/near-singular conversions fail explicitly.
* No implicit clamp/regularization.

---

## P1-09 — S-parameter conversion core + Z0 policy

**Goal**
Deterministic S conversion from Y or Z with strict Z0 validation and immutability.

**Scope**

* Implement:

  * `S=(Z-Z0)(Z+Z0)^-1`
  * `S=(I-Z0Y)(I+Z0Y)^-1`
* Validate Z0 (scalar or per-port vector; real finite >0):

  * complex -> `E_MODEL_PORT_Z0_COMPLEX`
  * non-positive/non-finite -> `E_MODEL_PORT_Z0_NONPOSITIVE`
* Singular conversion -> `E_NUM_S_CONVERSION_SINGULAR`
* No in-place mutation of Y/Z inputs.
* Enforce shared RF sentinel contract (see “RF sentinel contract” below).

**Deliverables**

* `src/rfmna/rf_metrics/s_params.py` (new)
* Unit + conformance tests (valid, invalid, singular, immutability)

**Acceptance**

* Formula/conventions align with normative docs via clause-mapped tests.
* Diagnostics taxonomy exact and deterministic.
* Input arrays unchanged (no in-place mutation) after call.

---

## P1-10 — Zin/Zout utilities

**Goal**
Deterministic impedance utilities built on shared boundary engine.

**Scope**

* Implement Zin/Zout per fixed excitation/termination conventions.
* Explicit undefined/singular boundary diagnostics.
* Per-point fail sentinels; never drop points.
* Enforce shared RF sentinel contract (see “RF sentinel contract” below).

**Deliverables**

* `src/rfmna/rf_metrics/impedance.py` (new)
* Unit tests for open/short/trivial/singular fixtures

**Acceptance**

* Sign/orientation conventions preserved.
* Undefined/singular points emit explicit diagnostics + sentinels.

---

## P1-11 — Sweep engine RF payload integration

**Goal**
Integrate RF payloads without altering Phase 0 base semantics.

**Scope**

* Optional payloads for Y/Z/S/Zin/Zout.
* Preserve existing base arrays and fail semantics.
* Exact per-point index alignment.
* Deterministic key/trace ordering.
* Preserve current `SweepResult` compatibility for non-RF callers (additive type evolution only).
* Use shared RF sentinel contract consistently across all RF payloads.

**Deliverables**

* `src/rfmna/sweep_engine/run.py` updates
* `src/rfmna/sweep_engine/types.py` (new or optional, if needed for additive typing)
* Unit tests for partial failure and alignment invariants

**Acceptance**

* Requested points always present.
* RF payload indices exactly match base indices.
* Failed points use standardized complex NaN sentinels + diagnostics entries.
* Existing non-RF tests remain green without test rewrite churn.

---

## P1-12 — RF export (NPZ/CSV + metadata)

**Goal**
Deterministic export surface for scripting/regression.

**Scope**

* NPZ export for complex matrices.
* Deterministic flattened CSV traces.
* Stable metadata keys: port order, Z0, convention tag, grid hash.
* Explicit schema version tag for RF export payload.
* Define and enforce CSV formatting contract:

  * deterministic column naming template,
  * deterministic complex encoding (e.g., paired real/imag columns),
  * fixed numeric formatting/precision policy,
  * stable encoding/newline behavior independent of locale.

**Deliverables**

* `src/rfmna/viz_io/rf_export.py` (new)
* Unit tests for deterministic key/column ordering, metadata stability, and CSV format contract

**Acceptance**

* Same input/config yields stable NPZ keys and CSV columns.
* Metadata includes RF convention context.
* Export deterministic under repeated runs and equivalent input permutations.
* CSV formatting contract validated by tests.

---

## P1-13 — CLI RF options/subcommands

**Goal**
Expose RF utilities while preserving v4 exit semantics and existing line grammar.

**Scope**

* Extend `run` options:

  * `--rf y|z|s|zin|zout` with explicit repeat/composition semantics.
* Deterministic summary/diagnostic ordering.
* Invalid option combinations fail with structured deterministic error.
* Maintain additive output: existing `POINT`/`DIAG` lines preserved; RF lines are additional and deterministic.
* Define deterministic composition/dependency matrix (including resolution order and failure propagation).

**Deliverables**

* `src/rfmna/cli/main.py` updates
* CLI integration tests
* CLI composition matrix doc snippet (co-located with tests or developer docs)

**Acceptance**

* Exit codes unchanged (0/1/2 contract).
* Existing grammar backward-compatible; RF additions additive.
* Invalid combos fail deterministically with machine-mappable diagnostics.
* Composition/dependency behavior is fully specified and test-covered.

---

## P1-14 — Phase 1 conformance bundle

**Goal**
Freeze and verify implemented Phase 1 semantics.

**Scope**

* Conformance for:

  * VCCS/VCVS stamps
  * port sign conventions
  * Y/Z/S formulas + well-posedness gates
  * Z0 validation + singular conversion diagnostics
  * deterministic RF output/diagnostic ordering
* Include pass/degraded/fail fixtures.
* Include regression fixture proving no Phase 0 frozen-artifact drift.
* Require explicit conformance ID coverage report mapping normative clauses to tests.

**Deliverables**

* `tests/conformance/rf_*`
* `tests/fixtures/rf_*`
* Conformance coverage report (ID → test cases)

**Acceptance**

* Detects convention drift.
* `unit + conformance` passes.
* No frozen-artifact drift without DR.
* Normative clause coverage is explicit and auditable.

---

## P1-15 — Documentation sync (implemented surface only)

**Goal**
Document only shipped behavior.

**Scope**

* Update README phase/status.
* Add concise usage docs (CLI + API).
* Document deterministic behavior, failure modes, and limits.
* Include diagnostic code table for new Phase 1 codes only.
* Include RF sentinel contract and CLI composition/dependency matrix documentation aligned with tests.

**Deliverables**

* `README.md` updates
* `docs/dev/phase1_usage.md` (new)

**Acceptance**

* No aspirational claims.
* All examples executable now.
* No normative spec edits unless approved via DR.
* Docs match actual tested behavior for sentinel contract and CLI composition rules.

---

## RF sentinel contract (applies to P1-07..P1-11)

1. Failed sweep points are retained; indices never dropped/reordered.
2. Failed complex scalar values use canonical complex NaN sentinel.
3. Failed complex matrix payloads use full-matrix sentinel fill (no partial-row/partial-cell mixed validity unless explicitly documented by future DR).
4. Sentinel representation and fill policy are identical across Y/Z/S/Zin/Zout paths.
5. Every failed point has corresponding deterministic diagnostic entry with stable witness shape.

---

## Execution order

1. P1-00
2. P1-01
3. P1-02
4. P1-03
5. P1-04
6. P1-05
7. P1-06
8. P1-07
9. P1-08
10. P1-09
11. P1-10
12. P1-11
13. P1-12
14. P1-13
15. P1-14
16. P1-15

---

## Codex per-task runbook (attach verbatim)

For each task:

1. Modify only scoped files.
2. Add/adjust tests for acceptance criteria.
3. Do not modify normative docs unless task explicitly includes DR + migration note + conformance evidence updates.
4. Include assumptions note in PR description.
5. Run:

   * `uv run ruff check .`
   * `uv run mypy src`
   * `uv run pytest -m "unit or conformance"`

Additional mandatory checks for tasks touching determinism:

* permutation-invariance tests for ordering
* stable witness serialization checks
* point-index alignment assertions for sweep payloads
* diagnostics ordering assertions via canonical sort key
* diagnostics catalog uniqueness/required-field checks for newly introduced codes
* numeric reproducibility assertions aligned with project tolerance policy across supported platforms/backends

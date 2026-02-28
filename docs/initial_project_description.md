## Personal RF MNA Solver — Revised Project Description (v4 Contract)

Build a **desktop-first, scriptable RF circuit simulator** for **linear AC small-signal analysis** using **Modified Nodal Analysis (MNA)**, optimized for small-to-medium analog/RF networks used for design exploration, sanity checks, and model validation.

This is a **personal engineering tool**: no multi-tenant architecture, no billing/auth, no SaaS concerns. Priorities are **correctness, numerical robustness, transparent diagnostics, deterministic behavior, and fast iteration**.

---

## 1) Purpose, Scope, and Mathematical Contract

### 1.1 Objective

Implement an RF-oriented solver that computes frequency-domain responses for linear networks with reliable diagnostics, deterministic execution, and reproducible workflows.

### 1.2 v4 Analysis Contract (normative)

At each sweep point ((\omega,\theta)), solve:
[
A(\omega,\theta)x(\omega,\theta)=b(\omega,\theta)
]
with block-structured MNA assembly:
[
A = A_{\text{topo}} + A_{\text{dyn}}(\omega) + A_{\text{fd}}(\omega,\theta) + A_{\text{aug}}
]

Where:

* (x): stacked unknown vector of non-reference node voltages and auxiliary unknowns.
* (A_{\text{topo}}): topology- and parameter-dependent static linear terms.
* (A_{\text{dyn}}(\omega)): dynamic AC terms (e.g., (j\omega C), inductor equations).
* (A_{\text{fd}}(\omega,\theta)): frequency-dependent compact linear terms.
* (A_{\text{aug}}): augmentation from voltage-defined elements, inductors, and controlled sources requiring auxiliary unknowns.

**v4 is LTI phasor-domain only.**
No nonlinear operating-point solve. No transient engine.

### 1.3 Linear algebra class (normative)

The solver treats (A) as **general complex sparse unsymmetric**.
No symmetry/Hermitian assumptions are permitted in core solve logic.

### 1.4 In-scope (v4)

* Lumped (R, L, C, G)
* Independent current/voltage sources
* Controlled sources: **VCCS, VCVS** (CCCS/CCVS deferred)
* Frequency-dependent compact linear forms (e.g., ESR/Q-style)
* 1-port/2-port linear blocks via **Y** form (normative in v4)
* 1-port/2-port **Z** form via either:

  * explicit auxiliary-current formulation, or
  * deterministic (Z\rightarrow Y) conversion with strict singularity checks and diagnostics
* AC frequency sweeps + arbitrary parameter sweeps
* Port metrics: Zin/Zout, Y/Z/S under explicitly defined conditions
* Export + plotting + reproducible CLI/API runs

### 1.5 Non-goals (v4)

* Full SPICE compatibility
* Nonlinear/transient simulation
* Layout extraction
* Collaboration/cloud features

---

## 2) Deterministic Unknown Indexing and Stamping Contract

### 2.1 Unknown vector ordering (global and fixed)

For (N) non-reference nodes and (M) auxiliary unknowns:
[
x = [V_1,\dots,V_N,; I_1^{\mathrm{aux}},\dots,I_M^{\mathrm{aux}}]^T
]

Rules:

* Ground/reference node excluded from (x).
* Node order deterministic (canonical parser/IR order).
* Auxiliary variables allocated deterministically by canonical element-instance order.
* Deterministic sparse row/col index ordering is mandatory.

### 2.2 Inductor formulation (fixed)

**Inductor branch-current auxiliary unknown formulation is mandatory in v4** (not optional).
All stamps, current signs, and branch equations must follow the normative appendix.

### 2.3 Element stamp API contract

Every element class must define:

1. Unknown indices touched.
2. Sparse pattern footprint.
3. Complex value contribution function for (A(\omega,\theta)).
4. RHS contribution function for (b(\omega,\theta)).
5. Parameter validation domain + error codes.
6. Optional physical validity checks (warning/error policy).

Stamping must be deterministic, side-effect free, and pure with respect to inputs.

### 2.4 Canonical stamp equations (normative appendix requirement)

A normative appendix defines exact sign conventions and matrix/RHS contributions for:

* (R, C, L, G)
* Independent I/V sources
* VCCS, VCVS
* 1-port and 2-port **Y embedded blocks**
* 1-port and 2-port **Z embedded blocks** (either explicit aux-current or (Z\rightarrow Y) path)

No implementation may deviate.

### 2.5 2-port embedded block equations (normative)

For 2-port Y block:
[
\begin{bmatrix}I_1\I_2\end{bmatrix}
===================================

\begin{bmatrix}Y_{11}&Y_{12}\Y_{21}&Y_{22}\end{bmatrix}
\begin{bmatrix}V_1\V_2\end{bmatrix}
]
with
[
V_k = V(p_k^+) - V(p_k^-), \quad I_k > 0 \text{ into DUT}
]
Stamping maps exactly to node-voltage unknowns using port incidence signs fixed in appendix.

For 2-port Z block:
[
\begin{bmatrix}V_1\V_2\end{bmatrix}
===================================

\begin{bmatrix}Z_{11}&Z_{12}\Z_{21}&Z_{22}\end{bmatrix}
\begin{bmatrix}I_1\I_2\end{bmatrix}
]
Either:

* stamp with auxiliary currents ((I_1,I_2)), or
* convert (Z\to Y=Z^{-1}) per point, fail on singular/near-singular (Z) by explicit (E_NUM_*) diagnostics.

---

## 3) Input, Parsing, IR, and Model Validation

### 3.1 Input formats

* SPICE-like subset and/or YAML front-end
* Expressions for parameters and sweeps
* `freq` mandatory for AC
* Arbitrary design parameters
* Optional temperature passthrough parameter (no thermal physics in v4)

### 3.2 Canonical internal IR

* Node table (canonical IDs, reference handling)
* Auxiliary unknown table
* Element table with normalized units + validated params
* Port declarations with orientation + (Z_0) metadata
* Frozen resolved-parameter table used for hashing/manifests

### 3.3 Deterministic expression and unit rules (required)

* Explicit unit suffix grammar (documented and versioned)
* Deterministic expression evaluation order
* Fixed numeric parsing policy (locale-independent)
* Fixed override precedence (file < CLI/API override)
* Canonical serialization for hash generation
* Parameter dependency cycle detection with explicit `E_MODEL_PARAM_CYCLE`
* No nondeterministic functions in expressions

### 3.4 Frequency grammar and sweep generation (required)

Accepted units: `Hz`, `kHz`, `MHz`, `GHz`

Sweep modes: linear/log

Normative deterministic point generation:

* **Linear**:
  [
  f_i = f_{\min} + i\cdot\frac{f_{\max}-f_{\min}}{N-1}, \quad i=0,\dots,N-1
  ]
  (for (N=1), (f_0=f_{\min}), require (f_{\min}=f_{\max}))
* **Log** ((f_{\min}>0)):
  [
  \log_{10} f_i = \log_{10}f_{\min} + i\cdot\frac{\log_{10}f_{\max}-\log_{10}f_{\min}}{N-1}
  ]
* Endpoints included exactly by construction.
* Grid generated in index space; no adaptive point insertion/removal.
* Export/manifest formatting precision fixed and versioned.
* Generated frequency vector hashed from canonical binary representation.

### 3.5 Pre-solve structural checks (required)

* Floating/disconnected nodes under analysis assumptions
* Connected components + reference reachability
* Port declaration validity (existence/orientation uniqueness)
* Duplicate/invalid reference declaration
* Contradictory/degenerate source constraints
* Ideal voltage source loop consistency checks
* Constraint inconsistencies among hard sources

### 3.6 Structural contradiction algorithms (normative)

Preflight must include deterministic graph checks:

1. **Voltage-source loop constraint check**
   Build directed equations (V_a - V_b = V_s).
   For each independent cycle, sum signed source voltages; nonzero residual above tolerance => `E_TOPO_VSRC_LOOP_INCONSISTENT`.

2. **Hard-source contradiction check**
   Detect multiple hard constraints imposing inconsistent values on same branch/node relation => `E_TOPO_HARD_CONSTRAINT_CONFLICT`.

3. **Witness policy**
   Emit deterministic witness list (canonical sorted element IDs, involved nodes, residual).

### 3.7 Frequency-conditional singularity risk checks (warning-level)

Flag likely ill-conditioned/asymptotic-open/short behaviors at sweep extremes.
Warnings do not block runs unless hard failure is detected.

---

## 4) MNA Assembly and Solve Pipeline

### 4.1 Assembly

* Complex sparse assembly per AC point
* Strict separation of topology-dependent and value-dependent contributions
* Two-stage assembly:

  1. **Pattern compile** (fixed sparse structure + index maps)
  2. **Numeric fill** per ((\omega,\theta))
* Pattern reuse and cached index maps across sweeps if topology unchanged
* Deterministic sparse index ordering

### 4.2 Solver backend (abstracted)

Backend interface supports:

* Sparse LU factorization for complex unsymmetric sparse (A)
* Retry ladder with configurable stabilization controls
* Residual computation + diagnostics
* Backend substitution without changing element code
* Return of factorization metadata (permutation info, pivot stats, failure reason)

### 4.3 Stabilization and conditioning controls

* `gmin` shunt ladder to non-reference nodes (normative default sequence)
* Optional row/column scaling
* Pivot/near-pivot monitoring
* Condition indicator estimation (fixed estimator type)

### 4.4 Mandatory fallback ladder (normative order)

For each failing/degraded point, retries occur in this fixed order:

1. Baseline factorization/solve
2. Alternative permutation/pivot configuration
3. Scaling enabled + retry
4. Incremental `gmin` ladder retries
5. Final fail with structured diagnostic bundle

Any order change requires version bump + decision record + changelog entry.

### 4.5 Residual and quality contract (required)

For every solved point, compute/store:

* (|r|*2), (|r|*\infty), (r=Ax-b)
* Relative residual:
  [
  r_{\text{rel}}=
  \frac{|r|*\infty}
  {|A|*\infty|x|*\infty+|b|*\infty+\epsilon}
  ]
  with fixed (\epsilon=10^{-30})
* Condition indicator (fixed estimator)
* Status label: `pass` / `degraded` / `fail`

Denominator-zero edge case handled by (\epsilon); no undefined result permitted.

### 4.6 Condition indicator contract (required)

* Estimator type fixed and versioned (e.g., LU-based reciprocal condition proxy).
* Per-point `cond_ind` scalar emitted.
* If estimator unavailable, emit `NaN` + warning `W_NUM_COND_UNAVAILABLE`.
* Threshold table defines warning/fail bands and is version-controlled.

### 4.7 Normative numeric defaults (v4.0.0)

Defaults are versioned and cannot silently change:

* dtype: `float64`, `complex128`
* Relative residual thresholds (default):

  * `pass`: (\le 1\times10^{-9})
  * `degraded`: (>10^{-9}) and (\le 10^{-6})
  * `fail`: (>10^{-6})
* `gmin` ladder (S): `[0, 1e-15, 1e-12, 1e-9, 1e-6]`
* Ladder resets per sweep point (no cross-point carryover)
* Condition-warning/fail bands fixed in threshold table (versioned)

---

## 5) Port and Network-Parameter Semantics (strict)

### 5.1 Global conventions (normative)

* Port voltage: (V_p = V(p^+) - V(p^-))
* Port current sign: positive **into** DUT
* Orientation must be consistent in all extraction and reports

### 5.2 Port extraction algorithms (normative)

Algorithms are fixed and appendix-specified:

* **Zin/Zout**: exact excitation/termination conditions defined.
* **Y-parameters**:

  * impose independent port voltages,
  * solve for resulting port currents,
  * assemble (Y) column-by-column.
* **Z-parameters**:

  * impose independent port currents,
  * solve for resulting port voltages,
  * assemble (Z) column-by-column.
* Inactive-port boundary conditions and constraint injection method are fixed and tested.
* Boundary-condition singularity returns explicit diagnostics.

### 5.3 Well-posedness checks for Y/Z (required)

Y or Z emitted only when all hold:

* structural solvability passes,
* numerical thresholds pass,
* boundary constraints consistent and non-singular.

Else emit explicit `E_TOPO_*` or `E_NUM_*`.

### 5.4 S-parameter contract (v4)

* Requires explicit real, positive (Z_0) per port (scalar/vector), default (50\Omega)
* Complex (Z_0) rejected
* S derived from one fixed documented wave convention
* Diagonal per-port (Z_0) matrix semantics explicit
* Near-singular conversion steps guarded by diagnostics (no silent regularization)

### 5.5 Optional RF sanity checks (warning-level)

* Reciprocity residual with tolerance
* Passivity sanity indicators
* Non-blocking in v4 unless explicitly configured as strict mode

---

## 6) Outputs and Post-processing

### 6.1 Core outputs

* Node voltages
* Selected branch/aux currents
* Transfer functions
* Port/2-port data (Zin/Zout, Y, Z, S where valid)

### 6.2 Derived RF metrics

* Gain / return loss / insertion loss
* Group delay
* Q estimates from impedance behavior
* Smith-chart-ready exports

### 6.3 Export and visualization

* CSV + NPZ exports
* Static plot snapshots
* Machine-readable diagnostics report (JSON)

### 6.4 Deterministic output ordering (required)

Stable ordering of nodes, ports, traces, sweeps, diagnostics, and columns.

### 6.5 Canonical data shapes (required)

* `V_nodes`: `complex[n_points, n_nodes]`
* `I_aux`: `complex[n_points, n_aux]`
* `res_l2`, `res_linf`, `res_rel`: `float[n_points]`
* `cond_ind`: `float[n_points]`
* `status`: `enum[n_points]`
* Per-point diagnostics: stable sorted event list with deterministic keys

### 6.6 Fail-point sentinel policy (normative)

For any failed point:

* Complex arrays (`V_nodes`, `I_aux`, complex transfer/port matrices): `nan + 1j*nan`
* Real arrays (`res_*`, `cond_ind`, derived scalar metrics): `nan`
* `status[i]` must be `fail`
* Diagnostics entry mandatory
* No point omission allowed

---

## 7) Diagnostics Taxonomy and Reporting Contract

Diagnostic families:

* `E_MODEL_*`: invalid/nonphysical params or model-region violations
* `E_TOPO_*`: structural/topology errors
* `E_NUM_*`: numerical failures/conditioning
* `W_RF_*`: RF sanity warnings
* `W_NUM_*`: numerical warnings (including conditioning availability)

Every diagnostic includes:

* `code`, `severity`, `message`
* `element_id` and/or node/port context
* frequency/sweep-point context (if applicable)
* `suggested_action`
* `solver_stage`: `parse|preflight|assemble|solve|postprocess`
* deterministic `witness` payload when applicable

Severity policy:

* Errors block affected solve path.
* Warnings never alter numerical results, only annotate.

---

## 8) Verification and Test Strategy

### 8.1 Unit tests (required)

* Each stamp class: nominal/edge/invalid-domain
* Deterministic indexing and reproducibility tests
* Sign-convention tests for all source/control orientations

### 8.2 Golden regression suite (required)

* RC/RLC sanity
* Resonance and damping behavior
* Controlled-source cases
* 2-port consistency checks (Y↔Z↔S where valid)

Golden artifacts include config/model hashes; mismatch invalidates stale golden usage.

### 8.3 Property-based tests (required, Hypothesis)

* Node relabeling invariance
* Reciprocity for randomized passive reciprocal topologies
* Positive-real/passivity sanity in constrained passive domains
* Conditioning/well-posedness filters to prevent flaky invalid cases

### 8.4 Cross-check harness

* Analytical references when available
* External trusted solver scripts for selected benchmarks
* Documented tolerances per metric (magnitude and phase separately)

### 8.5 Numerical acceptance thresholds (required)

Project-wide threshold table includes:

* residual pass/degraded/fail bands,
* condition warning/fail bands,
* regression tolerances for transfer and S-parameter metrics,
* conversion tolerances for Y/Z/S round trips.

Threshold changes require versioned decision record.

### 8.6 Conformance suite (required)

Dedicated conformance tests for:

* canonical stamp appendix,
* port current/voltage sign conventions,
* wave convention and S conversion formulas,
* fallback ladder order and config defaults,
* frequency-grid generation and endpoint behavior,
* fail-point sentinel policy.

---

## 9) Architecture

1. `parser` — formats, units, expressions, canonical IR
2. `elements` — stamp definitions + model validation
3. `assembler` — sparse pattern compile + per-point numeric fill
4. `solver` — backend abstraction, scaling, fallback ladder, diagnostics
5. `rf_metrics` — ports, Y/Z/S, derived RF metrics
6. `sweep_engine` — freq/parameter orchestration + cache policy
7. `viz_io` — plotting/export/report

Interfaces:

* CLI:

  * `rfmna run <design> --analysis ac`
  * `rfmna check <design>`
* Python API:

  * immutable model object
  * immutable result bundle (arrays + diagnostics + metadata)
  * per-point immutable solve record

---

## 10) Reproducibility and Artifact Contract

Each run emits manifest with:

* Tool version
* Git commit SHA (or source hash)
* Python version
* Dependency versions
* OS/platform info
* Input file hash + resolved-parameter hash
* Timestamp + timezone
* Solver config snapshot (`gmin`, scaling, retry settings)
* Thread/runtime fingerprint (thread env vars)
* Numerical backend fingerprint (SciPy runtime/build details as available)
* Frequency-grid generation metadata/version

Outputs are deterministically ordered and hash-stable under identical inputs/config.

---

## 11) Performance Envelope (v4 targets)

Guideline target class:

* Unknowns: ~50 to ~5,000
* Frequency points: ~100 to ~20,000
* Intended workflow: interactive design iteration on workstation/laptop for typical mid-range cases

Acceptance via bundled benchmarks with documented hardware baseline:

* per-point assembly time
* per-point and total solve time
* peak memory
* export throughput

Worst-case combinations are treated as batch workloads, not guaranteed interactive.

---

## 12) Technology Stack and Version Policy

### 12.1 Language baseline (strict)

* **Python 3.14 is mandatory baseline and primary tested target.**

### 12.2 Core dependencies (v4)

* NumPy
* SciPy (sparse linear algebra)
* Pandas
* Matplotlib (Plotly optional)
* Pydantic v2
* PyTest
* Hypothesis

### 12.3 Explicit ban

* `scikit-rf` (project remains independent)

### 12.4 Downgrade rule (strict)

Do **not** downgrade from Python 3.14 unless **any required pinned dependency for v4** is unavailable on supported platforms for 3.14.

If downgrade is required:

1. Record blocking package(s) and platform(s)
2. Pin lowest acceptable Python version
3. Emit decision record in repo docs + CI config
4. Re-run full unit/property/regression/conformance suite before merge

---

## 13) Development Plan (phased)

### Phase 0 — Foundations and Freeze Artifacts

* Repo skeleton, CI, linting, parser/IR baseline
* R/C/L/G + independent I/V sources
* Inductor auxiliary-current formulation implemented
* Complex AC solve + baseline plots
* Initial diagnostics scaffolding
* Normative stamp appendix drafted and frozen
* Residual formula, condition estimator, threshold table frozen
* Frequency-grid and sentinel policies frozen

### Phase 1 — RF Utility

* VCCS/VCVS
* Port declarations + Zin/Zout
* Y/Z/S utilities with strict validity checks
* Frequency + parameter sweeps
* Export/comparison tooling
* Port/wave convention appendix frozen

### Phase 2 — Robustness

* Conditioning controls (`gmin`/scaling/retry ladder)
* Diagnostics taxonomy finalized
* Regression expansion + tolerance calibration
* Hardened `check` command
* Full conformance suite enforced in CI

### Phase 3 — Personal Productivity

* Subcircuits/macros
* Practical RF model cards
* Notebook/report templates
* Optional CCCS/CCVS or mutual inductance (usage-driven)

---

## 14) Deliverables (v4)

1. Solver package with CLI + Python API
2. Input/model documentation
3. Test suite (unit/property/regression/cross-check/conformance)
4. Example projects:

   * L-match network
   * Narrowband RLC filter
   * Passive 2-port with S-parameter export
5. Engineering manual including:

   * formulation contract
   * conventions/sign definitions
   * canonical stamp equations
   * supported elements
   * numerical limits + troubleshooting
   * diagnostic reference
   * port extraction algorithms
   * reproducibility manifest schema
   * canonical data-shape/API conventions

---

## 15) Risk Register and Mitigations

1. **Singular/ill-conditioned systems**
   Structural preflight + scaling + fixed retry ladder + explicit diagnostics

2. **RF sign/wave convention ambiguity**
   Single-source normative appendix + conformance tests

3. **Scope creep toward SPICE clone**
   Strict v4 contract + controlled change process

4. **False confidence from plots**
   Mandatory reference comparisons + residual/invariant checks in CI

5. **Toolchain fragility**
   Pinned ranges + CI matrix on Python 3.14 for supported platforms

6. **Regression nondeterminism**
   Deterministic ordering + stable manifests + repeatability checks

7. **Extraction near singular boundaries**
   Explicit well-posedness gates + strict `E_NUM_*`/`E_TOPO_*` failures

---

## 16) Acceptance Criteria (v4 exit)

* Reproducible AC results across bundled examples
* All stamp unit tests, conformance tests, and golden regressions passing
* Property-based invariants passing in defined domains
* Diagnostics correctly classify intentionally broken circuits
* Y/Z/S match references within documented tolerances
* `rfmna check` catches defined structural and pre-solve issues
* CLI + API workflows documented and executable end-to-end
* Reproducibility manifest emitted for every run
* Frozen appendices implemented and tested
* No silent threshold/default drift (versioned change control enforced)

---

## 17) Immediate Implementation Freeze List (must complete first)

Before feature expansion, freeze and version:

1. Canonical element stamp equations (including inductor auxiliary-current form)
2. 2-port Y/Z embedded block equations and stamping policy
3. Port current/voltage and wave conventions
4. Residual formula and conditioning estimator definition
5. Residual/conditioning threshold table
6. Retry ladder order and default parameters
7. IR serialization/hash rules
8. Frequency grammar and sweep generation rules
9. CLI exit semantics and partial-sweep failure behavior
10. Canonical API data shapes and ordering rules
11. Fail-point sentinel policy
12. Deterministic thread-control defaults for CI/runtime

No v4 release without these frozen artifacts.

---

## 18) CLI Exit and Partial-Sweep Semantics (normative)

For `rfmna run`:

* Exit `0`: all points `pass`
* Exit `1`: at least one `degraded`, none `fail`
* Exit `2`: any `fail` point or any preflight structural error

Partial sweep handling:

* All requested points are reported in order.
* Failed points carry explicit status + diagnostics.
* Numeric result arrays use defined sentinel policy, never silent omission.

---

## 19) Change Control

Any change to normative appendices, thresholds, ladder defaults, ordering rules, extraction formulas, residual formula, condition estimator, frequency-grid algorithm, or sentinel policy requires:

* semantic version bump,
* migration note,
* conformance suite update,
* reproducibility impact statement.
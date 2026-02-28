<!-- docs/spec/v4_contract.md -->
# RFMNA v4 Contract (Normative)

Version: **4.0.0**  
Status: **Normative**  
Scope: **Linear AC small-signal phasor-domain only (LTI)**

## 1. Purpose

This project implements a deterministic RF-oriented MNA solver for personal engineering use, prioritizing:

- correctness,
- numerical robustness,
- transparent diagnostics,
- deterministic behavior,
- reproducible workflows.

No SaaS, auth, billing, or multi-tenant concerns are in scope.

---

## 2. Mathematical Contract

At each sweep point \((\omega,\theta)\), solve:

\[
A(\omega,\theta)\,x(\omega,\theta)=b(\omega,\theta)
\]

with block decomposition:

\[
A=A_{\text{topo}}+A_{\text{dyn}}(\omega)+A_{\text{fd}}(\omega,\theta)+A_{\text{aug}}
\]

Where:

- \(x\): non-reference node voltages and auxiliary unknowns,
- \(A_{\text{topo}}\): static topology/parameter terms,
- \(A_{\text{dyn}}(\omega)\): dynamic terms (\(j\omega C\), inductor equations),
- \(A_{\text{fd}}(\omega,\theta)\): compact linear frequency-dependent terms,
- \(A_{\text{aug}}\): auxiliary-equation augmentation (voltage-defined elements, inductors, controlled sources as applicable).

### 2.1 Linear algebra class (strict)

Core solve logic treats \(A\) as **general complex sparse unsymmetric**.  
No symmetry/Hermitian assumptions are permitted.

---

## 3. In-Scope vs Out-of-Scope (v4)

### 3.1 In-scope

- Lumped R, L, C, G
- Independent current/voltage sources
- Controlled sources: VCCS, VCVS
- Frequency-dependent compact linear forms
- 1-port/2-port linear blocks in **Y** form (normative)
- 1-port/2-port **Z** form via:
  - explicit auxiliary-current formulation, or
  - deterministic \(Z \to Y\) conversion with explicit singularity diagnostics
- AC frequency sweeps + arbitrary parameter sweeps
- Port metrics: Zin, Zout, Y, Z, S under fixed semantics
- CLI/API reproducible runs + exports/plots

### 3.2 Non-goals

- Full SPICE compatibility
- Nonlinear operating point
- Transient simulation
- Layout extraction
- Collaboration/cloud features

---

## 4. Deterministic Unknown Ordering

Unknown vector:

\[
x=[V_1,\dots,V_N,\ I^{\text{aux}}_1,\dots,I^{\text{aux}}_M]^T
\]

Rules:

1. Reference node excluded.
2. Node order canonical and deterministic.
3. Auxiliary allocation deterministic by canonical element-instance order.
4. Sparse row/column index ordering deterministic.

### 4.1 Inductor policy (strict)

Inductor formulation uses **mandatory auxiliary branch-current unknowns** in v4.

---

## 5. Assembly + Solve Pipeline

## 5.1 Two-stage assembly (strict)

1. Pattern compile (fixed sparsity + index maps)  
2. Numeric fill per \((\omega,\theta)\)

Pattern reuse across sweeps is required when topology is unchanged.

## 5.2 Solver backend requirements

- Sparse LU for complex unsymmetric sparse systems
- Retry ladder with deterministic order
- Residual diagnostics
- Backend substitution without element-code changes
- Metadata return (permutation/pivot/failure context)

## 5.3 Fallback ladder (normative order)

For degraded/failing points:

1. baseline factorization/solve
2. alternative permutation/pivot configuration
3. scaling enabled + retry
4. incremental `gmin` ladder retries
5. final fail with structured diagnostics

Any order change requires semver bump + decision record + conformance update.

---

## 6. Residual + Status Contract

Per solved point store:

- \(||r||_2\), \(||r||_\infty\), with \(r=Ax-b\)
- relative residual:
\[
r_{\text{rel}}=\frac{||r||_\infty}{||A||_\infty\,||x||_\infty+||b||_\infty+\epsilon}
\]
with fixed \(\epsilon=10^{-30}\)
- condition indicator scalar (`cond_ind`)
- status: `pass|degraded|fail`

No undefined residual values are permitted.

---

## 7. Port and Wave Semantics

Global convention:

- port voltage \(V_p = V(p^+) - V(p^-)\)
- port current positive **into DUT**

Algorithms for Zin/Zout/Y/Z/S are normative in appendix docs and must not drift silently.

---

## 8. Fail-Point Sentinel Policy (strict)

If a point fails:

- complex arrays: `nan + 1j*nan`
- real arrays: `nan`
- status: `fail`
- diagnostics entry: mandatory
- point omission: forbidden

All requested points must be present and ordered.

---

## 9. Diagnostics Contract

Diagnostic families:

- `E_MODEL_*`
- `E_TOPO_*`
- `E_NUM_*`
- `W_RF_*`
- `W_NUM_*`

Each diagnostic must include:

- `code`, `severity`, `message`,
- element/node/port context,
- sweep/frequency context when applicable,
- `suggested_action`,
- `solver_stage` in `{parse, preflight, assemble, solve, postprocess}`,
- deterministic `witness` payload when applicable.

Warnings annotate only; they do not alter numerical results.

---

## 10. Reproducibility Contract

Each run emits manifest including:

- tool version, git/source hash,
- Python/dependency versions,
- OS/platform,
- input + resolved-parameter hashes,
- timestamp/timezone,
- solver config snapshot,
- thread/runtime fingerprint,
- backend fingerprint,
- frequency-grid metadata/version.

Outputs must be deterministically ordered and hash-stable under identical inputs/configs.

---

## 11. CLI Exit Semantics (normative)

`rfmna run <design> --analysis ac`:

- exit `0`: all points `pass`
- exit `1`: at least one `degraded`, none `fail`
- exit `2`: any `fail` point or any preflight structural error

`rfmna check <design>` returns non-zero on structural/topology-contract violations.

---

## 12. Change Control

Changes to frozen artifacts require:

1. semver bump,
2. decision record,
3. conformance update,
4. migration note,
5. reproducibility impact statement.

Baseline governance references for this branch:

- decision record: `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`
- migration note: `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`

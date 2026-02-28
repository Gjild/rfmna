<!-- docs/spec/port_wave_conventions_v4_0_0.md -->
# Port and Wave Conventions v4.0.0 (Normative)

Version: **4.0.0**  
Status: **Normative**

## 1. Port conventions

For each declared port \(p\):

- terminal ordering is explicit: \((p^+, p^-)\)
- port voltage:
  \[
  V_p = V(p^+) - V(p^-)
  \]
- port current \(I_p\): positive **into DUT**

These sign conventions apply globally to Zin/Zout/Y/Z/S extraction and reporting.

---

## 2. Y and Z extraction conventions

### 2.1 Y-parameters

\[
\mathbf{I}=\mathbf{Y}\mathbf{V}
\]

Construct columns by imposing independent port voltages and solving resulting port currents under fixed inactive-port boundary conditions.

### 2.2 Z-parameters

\[
\mathbf{V}=\mathbf{Z}\mathbf{I}
\]

Construct columns by imposing independent port currents and solving resulting port voltages under fixed inactive-port boundary conditions.

Boundary-condition injection strategy is implementation-defined but must be deterministic and covered by conformance tests.

---

## 3. S-parameter convention

S parameters are derived from Y or Z using one fixed wave convention and diagonal real positive \(Z_0\).

- \(Z_0\) must be explicit scalar or per-port vector.
- Default: \(50\Omega\) per port.
- Complex \(Z_0\): rejected with `E_MODEL_PORT_Z0_COMPLEX`.
- Non-positive \(Z_0\): rejected with `E_MODEL_PORT_Z0_NONPOSITIVE`.

### 3.1 Preferred formulas

Given \(Z\) and diagonal \(Z_0\):

\[
S=(Z-Z_0)(Z+Z_0)^{-1}
\]

Given \(Y\) and diagonal \(Y_0=Z_0^{-1}\):

\[
S=(I-Z_0Y)(I+Z_0Y)^{-1}
\]

Numerically singular conversion steps must fail explicitly (no silent regularization), e.g. `E_NUM_S_CONVERSION_SINGULAR`.

---

## 4. Well-posedness gates for emitting Y/Z/S

Emit only when all hold:

1. structural solvability passes,
2. numerical quality passes configured thresholds,
3. boundary constraints are consistent and non-singular.

Otherwise, emit explicit `E_TOPO_*` or `E_NUM_*`.

---

## 5. Optional RF sanity warnings (non-blocking by default)

- reciprocity residual warning: `W_RF_RECIPROCITY`
- passivity sanity warning: `W_RF_PASSIVITY`

Warnings annotate outputs and do not alter numerical results.

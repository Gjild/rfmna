<!-- docs/spec/stamp_appendix_v4_0_0.md -->
# Canonical Stamp Appendix v4.0.0 (Normative)

Version: **4.0.0**  
Status: **Normative**  
Applies to: AC phasor-domain MNA with deterministic indexing/sign policy.

Definitions:

- Node voltages are unknowns for all non-reference nodes.
- For any two-terminal element between nodes \(a,b\), define branch voltage:
  \[
  v_{ab}=V_a-V_b
  \]
- Stamps use additive contributions into matrix \(A\) and RHS \(b\).
- \(j=\sqrt{-1}\), \(\omega=2\pi f\).

---

## 1. Conductance-like two-terminal stamp template

For branch admittance \(y\) between nodes \(a,b\) (with ground handled by omission of ref index):

- \(A_{aa} += y\)
- \(A_{bb} += y\)
- \(A_{ab} -= y\)
- \(A_{ba} -= y\)

### 1.1 Resistor \(R>0\)

\[
y=\frac{1}{R}
\]

Invalid domain (`R <= 0`) -> `E_MODEL_R_NONPOSITIVE`.

### 1.2 Conductance \(G \ge 0\)

\[
y=G
\]

Invalid domain (`G < 0`) -> `E_MODEL_G_NEGATIVE`.

### 1.3 Capacitor \(C \ge 0\)

\[
y=j\omega C
\]

Invalid domain (`C < 0`) -> `E_MODEL_C_NEGATIVE`.

---

## 2. Independent current source stamp

Current source from \(a \to b\) with source value \(I_s\) (positive from \(a\) to \(b\)) contributes only to RHS:

- \(b_a -= I_s\)
- \(b_b += I_s\)

Invalid/non-finite source values -> `E_MODEL_ISRC_INVALID`.

---

## 3. Independent voltage source stamp (auxiliary current required)

Voltage source between \(a,b\) with value \(V_s\), oriented \(V_a - V_b = V_s\).  
Allocate auxiliary unknown \(I_k\) (current through source, positive from \(a\) to \(b\)).

Unknown order is fixed globally: node voltages first, then auxiliaries.

Stamp:

- KCL coupling:
  - \(A_{a,k} += 1\)
  - \(A_{b,k} -= 1\)
- Source equation row \(k\):
  - \(A_{k,a} += 1\)
  - \(A_{k,b} -= 1\)
  - \(b_k += V_s\)

Invalid/non-finite source values -> `E_MODEL_VSRC_INVALID`.

---

## 4. Inductor stamp (mandatory auxiliary-current formulation)

Inductor between \(a,b\), inductance \(L>0\), auxiliary current \(I_k\) oriented \(a \to b\).

Equations:

- KCL coupling:
  - \(A_{a,k} += 1\)
  - \(A_{b,k} -= 1\)
- Branch equation:
  \[
  V_a - V_b - j\omega L\,I_k = 0
  \]
  hence:
  - \(A_{k,a} += 1\)
  - \(A_{k,b} -= 1\)
  - \(A_{k,k} += -j\omega L\)

Invalid domain (`L <= 0`) -> `E_MODEL_L_NONPOSITIVE`.

---

## 5. VCCS stamp (G element with control voltage)

VCCS output branch between \(a,b\), control voltage across \(c,d\), transconductance \(g_m\):

\[
I_{a\to b}=g_m\,(V_c-V_d)
\]

KCL contributions:

- \(A_{a,c} += g_m\)
- \(A_{a,d} -= g_m\)
- \(A_{b,c} -= g_m\)
- \(A_{b,d} += g_m\)

Invalid/non-finite `g_m` -> `E_MODEL_VCCS_INVALID`.

---

## 6. VCVS stamp (auxiliary current required)

VCVS output branch \(a,b\), control branch \(c,d\), gain \(\mu\), with equation:

\[
V_a - V_b = \mu\,(V_c - V_d)
\]

Allocate auxiliary current \(I_k\) for output branch.

Stamp:

- KCL coupling:
  - \(A_{a,k} += 1\)
  - \(A_{b,k} -= 1\)
- Constraint row \(k\):
  - \(A_{k,a} += 1\)
  - \(A_{k,b} -= 1\)
  - \(A_{k,c} -= \mu\)
  - \(A_{k,d} += \mu\)

RHS row \(k\) unchanged (0).

Invalid/non-finite `mu` -> `E_MODEL_VCVS_INVALID`.

---

## 7. 1-port Y block

For differential port nodes \(p^+, p^-\), branch current into DUT:

\[
I = Y\,V,\quad V=V(p^+)-V(p^-)
\]

Stamp as admittance-like template with \(y=Y\).

---

## 8. 2-port Y block

Define port voltages:

\[
V_1=V(p_1^+)-V(p_1^-),\quad V_2=V(p_2^+)-V(p_2^-)
\]

Currents positive into DUT:

\[
\begin{bmatrix}I_1\\I_2\end{bmatrix}
=
\begin{bmatrix}Y_{11}&Y_{12}\\Y_{21}&Y_{22}\end{bmatrix}
\begin{bmatrix}V_1\\V_2\end{bmatrix}
\]

Implementation shall stamp equivalent nodal injections using fixed port incidence signs (from `port_wave_conventions_v4_0_0.md`).

---

## 9. Z block policy

For Z-parameter blocks:

\[
\begin{bmatrix}V_1\\V_2\end{bmatrix}
=
\begin{bmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{bmatrix}
\begin{bmatrix}I_1\\I_2\end{bmatrix}
\]

Allowed paths:

1. explicit auxiliary-current stamping with \(I_1, I_2\) unknowns, or
2. deterministic per-point conversion \(Y=Z^{-1}\), with explicit singular/near-singular diagnostics:
   - `E_NUM_ZBLOCK_SINGULAR`
   - `E_NUM_ZBLOCK_ILL_CONDITIONED`

No silent regularization.

---

## 10. Determinism requirements for all stamps

Each element class must define:

1. touched unknown indices,
2. sparse footprint (deterministic ordering),
3. value contributions to \(A\),
4. RHS contributions to \(b\),
5. validation domains with explicit error codes.

Stamp methods must be pure and side-effect free.

<!-- docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md -->
# Frequency Grid and Sweep Rules v4.0.0 (Normative)

Version: **4.0.0**  
Status: **Normative**

## 1. Accepted units

`Hz`, `kHz`, `MHz`, `GHz`

Numeric parsing is locale-independent and deterministic.

---

## 2. Sweep modes

Supported: `linear`, `log`

Inputs:

- \(f_{\min}\),
- \(f_{\max}\),
- \(N\) points.

Endpoints are included exactly by construction.

---

## 3. Linear grid formula

For \(N>1\):

\[
f_i=f_{\min}+i\cdot\frac{f_{\max}-f_{\min}}{N-1},\quad i=0,\dots,N-1
\]

For \(N=1\):

- require \(f_{\min}=f_{\max}\),
- return \(f_0=f_{\min}\).

Violations -> `E_MODEL_FREQ_GRID_INVALID`.

---

## 4. Log grid formula

Domain requirement: \(f_{\min}>0\), \(f_{\max}>0\), \(N\ge1\).

For \(N>1\):

\[
\log_{10} f_i=\log_{10}f_{\min} + i\cdot\frac{\log_{10}f_{\max}-\log_{10}f_{\min}}{N-1}
\]

For \(N=1\): same single-point rule as linear mode.

Invalid domain -> `E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN`.

---

## 5. Determinism constraints

- Grid generated from index space (no adaptive insertion/removal).
- Canonical floating representation for hashing is fixed (`float64`, native byte order normalization in serializer).
- Export precision is fixed and versioned.
- Frequency vector hash is derived from canonical binary representation only.

---

## 6. Point orchestration and failures

- All requested points must be produced in deterministic order.
- Failed points are retained and flagged; no omissions.
- Sentinel policy applies to failed points exactly as specified in v4 contract.

<!-- docs/spec/diagnostics_taxonomy_v4_0_0.md -->
# Diagnostics Taxonomy v4.0.0 (Normative)

Version: **4.0.0**  
Status: **Normative**

## 1. Families

- `E_MODEL_*`: model/parameter errors
- `E_TOPO_*`: structural/topology errors
- `E_NUM_*`: numerical failures/conditioning issues
- `W_RF_*`: RF sanity warnings
- `W_NUM_*`: numerical warnings

---

## 2. Required fields (every diagnostic)

1. `code`
2. `severity` (`error|warning`)
3. `message`
4. context (`element_id` and/or node/port context)
5. sweep/frequency context if applicable
6. `suggested_action`
7. `solver_stage` in:
   - `parse`
   - `preflight`
   - `assemble`
   - `solve`
   - `postprocess`
8. deterministic `witness` payload when applicable

---

## 3. Severity semantics

- **Errors**: block affected solve path.
- **Warnings**: annotation only; must not alter numerical results.

---

## 4. Deterministic ordering

Diagnostics must be emitted in stable deterministic order under equivalent inputs.

Recommended canonical sort tuple:

1. `severity` rank,
2. `solver_stage` rank,
3. `code`,
4. `element_id` (None last),
5. point indices ordered lexicographically as:
   - `frequency_index` None-last rank, then value
   - `sweep_index` None-last rank, then value
6. message,
7. canonical witness serialization.

This order must be frozen in implementation + conformance tests.

---

## 5. Required codes (minimum set in Phase 0/1)

## 5.1 Model

- `E_MODEL_PARAM_CYCLE`
- `E_MODEL_R_NONPOSITIVE`
- `E_MODEL_L_NONPOSITIVE`
- `E_MODEL_C_NEGATIVE`
- `E_MODEL_G_NEGATIVE`
- `E_MODEL_FREQ_GRID_INVALID`
- `E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN`
- `E_MODEL_PORT_Z0_COMPLEX`
- `E_MODEL_PORT_Z0_NONPOSITIVE`

## 5.2 Topology

- `E_TOPO_VSRC_LOOP_INCONSISTENT`
- `E_TOPO_HARD_CONSTRAINT_CONFLICT`
- `E_TOPO_FLOATING_NODE`
- `E_TOPO_REFERENCE_INVALID`
- `E_TOPO_PORT_INVALID`

## 5.3 Numerical

- `E_NUM_SOLVE_FAILED`
- `E_NUM_SINGULAR_MATRIX`
- `E_NUM_ZBLOCK_SINGULAR`
- `E_NUM_ZBLOCK_ILL_CONDITIONED`
- `E_NUM_S_CONVERSION_SINGULAR`

## 5.4 Warnings

- `W_NUM_COND_UNAVAILABLE`
- `W_NUM_ILL_CONDITIONED`
- `W_RF_RECIPROCITY`
- `W_RF_PASSIVITY`

---

## 6. Witness policy

Where applicable, witness payload shall include deterministic canonical IDs and residuals/constraint details.  
Witness lists must be sorted canonically for stable output and hashing.

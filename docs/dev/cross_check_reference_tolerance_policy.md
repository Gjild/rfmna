# P2-08 Cross-Check Reference and Tolerance Policy

This document defines the deterministic reference set and tolerance policy for `tests/cross_check`.

## Reference Sources

- Fixture corpus: `tests/fixtures/cross_check/*.json` (hash-locked by `approved_hashes_v1.json`).
- Equation source: closed-form linear RF relations from the v4 contract semantics:
  - `eq-1` (fixture `one_port_shunt_rc_v1`): one-port shunt RC
    - \(Y_{11}=G+j\omega C\)
    - \(Z_{11}=1 / Y_{11}\)
    - \(S_{11}=(Z_{11}-Z_0)/(Z_{11}+Z_0)\)
  - `eq-2` (fixture `two_port_coupled_rc_v1`): two-port coupled RC
    - \(Y_{11}=G_1+G_{12}+j\omega(C_1+C_{12})\)
    - \(Y_{22}=G_2+G_{12}+j\omega(C_2+C_{12})\)
    - \(Y_{12}=Y_{21}=-(G_{12}+j\omega C_{12})\)
    - \(Z=Y^{-1}\)
    - \(S=(Z-Z_0I)(Z+Z_0I)^{-1}\)

## Gating Tolerance Source

- Merge-gating tolerance source for cross-check assertions is:
  - `docs/dev/tolerances/regression_baseline_v1.yaml`
- Per-metric profile map enforced by tests:
  - `y -> rf_matrix_tight`
  - `z -> rf_matrix_tight`
  - `s -> rf_matrix_loose`

## Classification and CI Gating Rule

- Cross-check tests must use only `normative_gating` tolerance sources from
  `docs/dev/threshold_tolerance_classification.yaml`.
- `docs/dev/tolerances/calibration_seed_v1.yaml` is `calibration_only` and is explicitly non-gating.
- Tests fail if any cross-check fixture references a `calibration_only` source.

## Platform/Backend Variability Policy

- `Y` and `Z` checks are run with `rf_matrix_tight` (direct extraction paths).
- `S` checks use `rf_matrix_loose` to absorb backend/platform variability from conversion solve paths.
- Tolerance enforcement is explicit (`assert_allclose` with deterministic `rtol/atol`); no ad-hoc pass/fail logic is allowed.

## P2-00 Marker/Strictness Drift Guard

- `tests/cross_check` includes an explicit guard that `pytest.ini` still contains:
  - `--strict-markers`
  - `cross_check` marker declaration

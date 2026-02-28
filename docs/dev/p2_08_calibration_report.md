# P2-08 Calibration Report

## Scope

- Task: `P2-08` Threshold/tolerance calibration + cross-check harness.
- Dataset: `tests/fixtures/cross_check/*.json`.
- Calibration script: `scripts/cross_check/calibrate_tolerances.py`.
- Gating tolerance source retained: `docs/dev/tolerances/regression_baseline_v1.yaml`.

## Procedure

Run:

```bash
uv run python scripts/cross_check/calibrate_tolerances.py
```

The script replays deterministic cross-check fixtures, computes analytic `Y/Z/S` references,
and reports max absolute/relative error by metric.

Observed replay summary:

- `one_port_shunt_rc_v1`
  - max abs error: `y=0.0`, `z=3.552713678800501e-15`, `s=5.721958498152797e-17`
  - max rel error: `y=0.0`, `z=8.39157942778994e-17`, `s=1.90912168741648e-16`
- `two_port_coupled_rc_v1`
  - max abs error: `y=0.0`, `z=0.0`, `s=2.775557561563207e-17`
  - max rel error: `y=0.0`, `z=0.0`, `s=2.5951463200600584e-16`
- aggregate max abs error by metric:
  - `y=0.0`
  - `z=3.552713678800501e-15`
  - `s=5.721958498152797e-17`

## Ratified Profile Map

- `y -> rf_matrix_tight`
- `z -> rf_matrix_tight`
- `s -> rf_matrix_loose`

Rationale:

- `Y/Z` are direct extraction paths with very low numeric dispersion in calibration replay.
- `S` includes conversion solve sensitivity and therefore keeps the looser profile for backend/platform stability.

## Governance Classification Note

- `docs/dev/tolerances/calibration_seed_v1.yaml` remains `calibration_only` and non-gating.
- CI pass/fail assertions in `tests/cross_check` are enforced only through
  `docs/dev/tolerances/regression_baseline_v1.yaml` (`normative_gating`).
- No frozen artifact IDs were touched for P2-08 implementation scope.

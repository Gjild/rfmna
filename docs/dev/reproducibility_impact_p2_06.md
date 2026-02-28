# Reproducibility Impact Statement: P2-06

Date: `2026-02-28`  
Task: `P2-06` regression suite expansion (golden + tolerance-aware)

## Scope

- Deterministic golden fixtures under `tests/fixtures/regression/`.
- Deterministic approved-hash lock using canonical JSON serialization and SHA-256.
- Deterministic fixture-set parity enforcement between fixture directory and approved-hash lock entries.
- Deterministic tolerance-table-driven numeric assertions from `docs/dev/tolerances/regression_baseline_v1.yaml`.

## Reproducibility assessment

- Golden comparisons are stable under repeated runs with identical fixture/tolerance artifacts.
- Hash-lock checks reject unapproved fixture mutations and stale lock-file entries.
- Sentinel representation is explicit and deterministic: only `[null, null]` encodes complex fail sentinels.
- Regression tolerances classified as `normative_gating` are merge-gating with governance evidence requirements.

## Validation evidence

- `uv run pytest -m "unit or conformance"`
- `uv run pytest -m cross_check`
- `uv run pytest -m regression`

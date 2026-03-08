<!-- docs/spec/migration_notes/2026-03-07-v0-1-3-p3-01-design-bundle-loader-cli.md -->
# Migration Note: v0.1.3 P3-01 design bundle loader integration

Date: `2026-03-07`  
Related DR: `docs/spec/decision_records/2026-03-07-p3-01-design-bundle-loader-cli-governance.md`

## Summary

This release introduces the canonical `design_bundle_v1` input contract and replaces the CLI loader stub with a deterministic in-repo loader for supported AC bundles.

## Consumer Impact

- `rfmna check <design>` now accepts JSON design bundles conforming to `docs/spec/schemas/design_bundle_v1.json`.
- `rfmna run <design> --analysis ac` now executes supported design bundles through the in-repo loader.
- Exit mappings remain unchanged:
  - `check`: `0` or `2`
  - `run`: `0`, `1`, or `2`
- Deferred in-scope capabilities are explicit in `docs/dev/p3_loader_temporary_exclusions.yaml` and fail with deterministic diagnostics instead of a stubbed loader boundary.

## Required Action

1. Provide explicit `schema` and `schema_version` fields in design bundles.
2. Use only the supported interim element surface (`R`, `C`, `G`, `L`, `I`, `V`, `VCCS`, `VCVS`) until the exclusion list is retired.
3. Keep parameter sweeps, Y/Z block elements, and frequency-dependent compact linear forms out of `design_bundle_v1` inputs until Phase 3 closure lifts the interim exclusions.

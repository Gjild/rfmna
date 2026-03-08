# Reproducibility Impact: P3-01 design bundle loader

Date: `2026-03-07`

## Scope

This note covers the reproducibility impact of replacing the CLI loader stub with the canonical `design_bundle_v1` loader path.

## Impact Statement

- Schema selection is explicit and deterministic through the payload pair:
  - `schema`
  - `schema_version`
- Parameter resolution remains deterministic and file-local for the new loader surface.
- RF port ordering is canonicalized by existing port-id sorting rules before RF extraction.
- Sparse assembly preserves the repository contract:
  - compile pattern once,
  - numeric fill per point only.
- Frequency-grid generation delegates to the existing frozen implementation and therefore does not change the grid algorithm.
- Run manifest hashing now includes the loaded design payload and resolved parameter map emitted by the loader, so manifest hashes track real design-content changes even when the source path is unchanged.

## Non-Impact Claims

- No change to `run` exit semantics.
- No change to `check` exit semantics.
- No change to fail-point sentinel semantics.
- No change to RF port/wave sign conventions.

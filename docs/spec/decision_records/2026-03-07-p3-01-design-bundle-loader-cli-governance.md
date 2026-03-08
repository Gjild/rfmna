<!-- docs/spec/decision_records/2026-03-07-p3-01-design-bundle-loader-cli-governance.md -->
# DR: P3-01 design bundle contract and CLI loader governance record

Status: `accepted`  
Date: `2026-03-07`

## Context

P3-01 replaces the CLI loader stub with a deterministic in-repo design-bundle contract and loader path.

The task introduces:

- canonical schema artifact `docs/spec/schemas/design_bundle_v1.json`,
- deterministic parser/loader modules for supported AC design bundles,
- explicit interim exclusion handling for deferred v4 loader capabilities,
- loader-backed `check` / `run --analysis ac` execution with structured diagnostics.

Because the integration touches `src/rfmna/cli/main.py`, the current governance rule table detects frozen artifact `#9` (CLI exit semantics and partial-sweep behavior).

Frozen artifacts `#3`, `#8`, `#10`, and `#11` were explicitly assessed as `non-impact`:

- `#3`: port orientation and wave conventions are forwarded unchanged into existing RF boundary/request logic,
- `#8`: frequency grammar delegates to the existing frozen `frequency_grid` implementation,
- `#10`: canonical API payload shapes and ordering continue to flow through the existing sweep/RF export path,
- `#11`: fail-point sentinel handling remains in the existing sweep engine and is not semantically changed.

## Decision

1. Accept the P3-01 loader integration as a governed frozen-scope change set for artifact `#9`.
2. Keep `rfmna check` and `rfmna run` exit semantics unchanged while replacing the loader stub with the real in-repo loader.
3. Bump package version from `0.1.2` to `0.1.3`.
4. Publish the companion migration note and reproducibility impact statement:
   - `docs/spec/migration_notes/2026-03-07-v0-1-3-p3-01-design-bundle-loader-cli.md`
   - `docs/dev/reproducibility_impact_p3_01.md`

## Consequences

- Supported JSON design bundles now execute through the repository loader without monkeypatched integration glue.
- Deferred loader capabilities are explicit and deterministic through `docs/dev/p3_loader_temporary_exclusions.yaml`.
- Loader failures now emit cataloged structured diagnostics instead of a repository-local stub boundary.
- `rfmna check` and `rfmna run` exit mappings remain unchanged.

## Conformance Impact

Conformance coverage updated for:

- loader-backed `check` / `run` success paths,
- loader-backed RF ordering and matched-port non-regression,
- loader-backed frequency grammar non-regression,
- loader diagnostic completeness and Phase 3 surface-governance updates.

## Reproducibility Impact

Design loading now uses a canonical schema/version pair, deterministic parameter resolution, deterministic port ordering, and one-time sparse-pattern compilation before numeric fill across the sweep.

## Migration Note

See:
`docs/spec/migration_notes/2026-03-07-v0-1-3-p3-01-design-bundle-loader-cli.md`

## Semver Impact

Version bump: `0.1.2` -> `0.1.3`.

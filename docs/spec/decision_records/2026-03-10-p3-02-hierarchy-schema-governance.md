<!-- docs/spec/decision_records/2026-03-10-p3-02-hierarchy-schema-governance.md -->
# DR: P3-02 hierarchy grammar/schema governance record

Status: `accepted`  
Date: `2026-03-10`

## Context

P3-02 extends `design_bundle_v1` with deterministic hierarchy grammar support for macros, subcircuits, and hierarchy instances. The task also introduces a packaged runtime copy of the schema at `src/rfmna/parser/resources/design_bundle_v1.json` so installed builds load the same contract as the repository source.

Once the packaged mirror becomes a tracked runtime artifact, the frozen-governance rule table detects frozen artifacts `#3`, `#8`, `#9`, and `#10` because the added file contains governed schema tokens for:

- RF port boundary fields and `z0_ohm`,
- frequency grammar tokens,
- explicit schema/default-selection compatibility policy,
- canonical ordering compatibility policy.

The task intent is additive-only. The touched frozen artifacts are governance touches caused by the newly tracked mirror, not semantic changes to those frozen behaviors.

## Decision

1. Accept P3-02 as a governed frozen-scope change set for artifacts `#3`, `#8`, `#9`, and `#10`.
2. Keep the active schema contract on the explicit `docs/spec/schemas/design_bundle_v1.json` + `schema_version: 1` path.
3. Require the packaged runtime schema mirror to stay byte-aligned with the repository schema artifact, enforced by conformance/governance rather than parser runtime failure.
4. Bump package version from `0.1.3` to `0.1.4`.
5. Publish the companion migration note and reproducibility impact statement:
   - `docs/spec/migration_notes/2026-03-10-v0-1-4-p3-02-hierarchy-schema-governance.md`
   - `docs/dev/reproducibility_impact_p3_02.md`

## Consequences

- Installed parser builds now consume a governed packaged schema mirror rather than relying only on repository-relative schema access.
- Hierarchy instance identifiers are normalized deterministically for duplicate detection.
- Hierarchy instantiation now fails early with explicit diagnostics when referenced macro/subcircuit arity does not match the supplied node list.
- Existing flat-input behavior, frequency interpretation, schema selection, and canonical ordering remain unchanged.

## Conformance Impact

Conformance coverage updated for:

- additive `v1` hierarchy schema compatibility,
- canonical hierarchy parse-product permutation invariance,
- hierarchy duplicate/reference/arity diagnostic completeness,
- frozen-governance detection for the packaged runtime schema mirror,
- Phase 3 gate cross-consistency between frozen scope and non-frozen surface declarations.

## Reproducibility Impact

Hierarchy parsing remains deterministic: identifier normalization is explicit, duplicate handling is normalization-aware, arity failures are emitted in stable sorted diagnostic order, and the packaged schema mirror uses the same governed bytes as the repository schema artifact.

## Migration Note

See:
`docs/spec/migration_notes/2026-03-10-v0-1-4-p3-02-hierarchy-schema-governance.md`

## Semver Impact

Version bump: `0.1.3` -> `0.1.4`.

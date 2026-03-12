# Reproducibility Impact: P3-02 hierarchy grammar/schema support

Date: `2026-03-10`

## Scope

P3-02 adds deterministic hierarchy grammar/schema support on the existing `design_bundle_v1` contract and introduces a packaged runtime schema mirror for installed builds.

## Determinism Statement

- Hierarchy definition and instance name matching use explicit Unicode NFC normalization followed by the existing uppercase/separator-collapsing policy.
- Duplicate hierarchy instance detection now keys on that normalized identifier, so canonically equivalent spellings cannot produce scope-dependent behavior.
- Hierarchy instance arity mismatches are rejected during parse with stable diagnostic payloads rather than leaking malformed declarations into later phases.
- Canonical parse-product ordering remains permutation-invariant for equivalent hierarchy declaration sets.
- Manifest input hashing continues to preserve source declaration order; only canonical parse-product hashing is permutation-invariant.

## Frozen-Surface Statement

The packaged runtime schema mirror adds a new tracked path that contains governed schema tokens for frozen artifacts `#3`, `#8`, `#9`, and `#10`. This is a governance-scope touch, not a semantic change to port conventions, frequency grammar, schema selection, or canonical ordering.

## Runtime Statement

Installed builds now load the packaged schema mirror from `src/rfmna/parser/resources/design_bundle_v1.json`. The repository schema at `docs/spec/schemas/design_bundle_v1.json` remains the authoritative contract source, and conformance/governance coverage keeps the two artifacts aligned without turning mirror drift into a parser runtime outage during source checkouts.

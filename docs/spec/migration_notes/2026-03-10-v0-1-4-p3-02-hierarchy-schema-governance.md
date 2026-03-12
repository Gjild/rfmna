<!-- docs/spec/migration_notes/2026-03-10-v0-1-4-p3-02-hierarchy-schema-governance.md -->
# Migration Note: v0.1.4 P3-02 hierarchy grammar/schema governance

Date: `2026-03-10`  
Related DR: `docs/spec/decision_records/2026-03-10-p3-02-hierarchy-schema-governance.md`

## Summary

This release extends `design_bundle_v1` with deterministic hierarchy grammar declarations for macros, subcircuits, and hierarchy instances, and adds a packaged runtime schema mirror used by installed parser builds.

## Consumer Impact

- Flat `design_bundle_v1` payloads remain valid without hierarchy declarations.
- Hierarchy declarations may now include `design.macros`, `design.subcircuits`, and `design.instances`.
- Macro definitions may provide partial defaults; required element-model parameters may be supplied at instantiation.
- Duplicate hierarchy instance identifiers are rejected after deterministic normalization.
- Hierarchy instances with node-list arity mismatches now fail at parse time with explicit diagnostics.
- Schema selection remains explicit:
  - `schema: "docs/spec/schemas/design_bundle_v1.json"`
  - `schema_version: 1`

## Required Action

1. Keep hierarchy instance ids unique within each scope after normalization (`NFC`, uppercase, separator collapse).
2. Match hierarchy instance node counts to the referenced macro `node_formals` or subcircuit `ports`.
3. If you vendor or validate the schema outside the repository tree, refresh against both the repository schema and the packaged runtime mirror to preserve byte alignment.
4. In source checkouts, schema-mirror alignment is enforced by CI/conformance rather than by failing design loads at parser startup.

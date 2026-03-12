# P3-02 Design Bundle Schema Evolution

Selected path: additive extension of `docs/spec/schemas/design_bundle_v1.json`.

## Decision

- `design_bundle_v1.json` remains the active schema artifact.
- `schema` must remain `docs/spec/schemas/design_bundle_v1.json`.
- `schema_version` must remain `1`.
- Hierarchy support is introduced only through new optional `design` fields:
  - `macros`
  - `subcircuits`
  - `instances`
- Hierarchy identifier normalization is deterministic and Unicode-canonical:
  - apply Unicode NFC before uppercase/separator-collapsing
  - canonically equivalent spellings must resolve to the same normalized identifier

## Compatibility Policy

- No existing required root, `design`, or `analysis` fields are removed.
- No validation constraints are tightened for existing flat-design fields.
- No default-selection behavior changes:
  - no filename inference,
  - no highest-version inference,
  - no implicit upgrade from `v1` to a future schema.
- No existing flat-input ordering semantics change, including subcircuit-local flat element order.
- Flat `v1` payloads remain valid without hierarchy declarations.

## Current Execution Boundary

- `P3-02` adds deterministic hierarchy grammar parsing and canonical parse-product support.
- `parse_design_bundle_document()` remains available from `rfmna.parser.design_bundle` as the canonical hierarchy parse-product entry point and therefore accepts schema-valid inputs without applying interim loader exclusions.
- `parse_design_bundle_document()` still applies hierarchy diagnostics, hierarchy value validation, and the existing flat bundle validation contract before producing a canonical parse product; composed macro defaults plus instance overrides and instantiated subcircuit parameter scopes are validated even though elaboration remains deferred. Its carve-out is limited to interim loader exclusions and non-runnable top-level elaboration policy.
- `load_design_bundle_document()` remains loader-policy-aware and continues to enforce interim exclusions for loader-backed `check` / `run` before loader-only hierarchy value validation.
- Top-level `design.instances` remain non-runnable in `check/run` until `P3-03` adds deterministic hierarchy elaboration.
- Unused hierarchy definitions may coexist with flat runnable payloads.

## Frozen Classification

- Frozen artifact `#3`:
  - detection touched through the newly tracked packaged runtime schema mirror
  - explicit no semantic change; port orientation and `z0_ohm` semantics remain unchanged
- Frozen artifact `#8`:
  - detection touched through the newly tracked packaged runtime schema mirror
  - explicit no semantic change; frequency grammar and grid generation remain unchanged
- Frozen artifact `#9`:
  - detection touched through the newly tracked packaged runtime schema mirror
  - explicit no semantic change; schema/default selection remains the explicit `schema` + `schema_version` pair
- Frozen artifact `#10`:
  - detection touched through the newly tracked packaged runtime schema mirror
  - explicit no semantic change; existing flat-input output ordering remains unchanged

## Conformance IDs

- `P3-02-CID-001`:
  - additive-`v1` compatibility policy remains enforced for existing root/design/analysis fields
  - executable test: `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_additive_v1_schema_evolution_keeps_existing_required_fields_and_defaults`
- `P3-02-CID-001A`:
  - canonical parse-product support preserves existing flat-input order semantics, including subcircuit-local flat element order, while canonicalizing only hierarchy declaration permutations
  - executable test: `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_canonical_parse_product_preserves_flat_v1_order_semantics`
- `P3-02-CID-002`:
  - equivalent hierarchy declaration permutations and supported kind aliases produce identical canonical parse-product JSON and canonical SHA-256 hash
  - executable tests:
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_canonical_hierarchy_parse_product_is_order_and_hash_stable`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_canonical_parse_product_normalizes_supported_kind_aliases`
- `P3-02-CID-003`:
  - hierarchy diagnostics remain ordered and witness-stable for duplicate/conflict/arity/undefined/recursion/value-validation failures, including ambiguous references, duplicate-normalized recursive subcircuits, illegal local namespace collisions, and composed macro/subcircuit instantiation checks
  - executable tests:
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_diagnostics_are_ordered_and_witness_stable`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_illegal_hierarchy_interface_declarations_are_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_parse_surface_preserves_flat_validation_contract`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_macro_instance_requires_complete_composed_model_before_unsupported`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_instantiated_subcircuit_override_validation_applies_to_target_body`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_subcircuit_instance_override_scope_can_reference_overridden_siblings`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_recursion_diagnostics_are_bounded_per_recursive_component`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_duplicate_subcircuit_definition_still_emits_recursion_diagnostic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_reference_to_definition_conflict_is_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_reference_to_duplicate_definition_is_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_mixed_duplicate_and_conflicting_hierarchy_diagnostics_are_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_duplicate_definition_diagnostic_is_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_duplicate_local_element_id_diagnostic_is_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_duplicate_instance_id_diagnostic_is_taxonomy_complete_and_deterministic`
    - `tests/conformance/test_design_bundle_hierarchy_conformance.py::test_hierarchy_instance_arity_diagnostic_is_taxonomy_complete_and_deterministic`

## Reproducibility Impact

- Hierarchy duplicate detection now keys on deterministic normalized instance identifiers, so canonically equivalent instance spellings cannot diverge by declaration spelling alone.
- Hierarchy instantiation now emits explicit parse diagnostics for macro/subcircuit node-arity mismatches instead of allowing malformed declarations into later phases.
- Conformance/governance keeps the packaged runtime schema mirror byte-for-byte aligned with `docs/spec/schemas/design_bundle_v1.json`; adding the packaged mirror to governance scope does not introduce alternate schema selection or ordering behavior.

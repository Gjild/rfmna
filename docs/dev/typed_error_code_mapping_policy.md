# Typed Error Code Mapping Policy (Track B)

This policy governs **non-diagnostic typed error codes** used by internal exception paths.
It is distinct from runtime structured diagnostics (`DiagnosticEvent`/`SweepDiagnostic`).

## Artifacts

- Registry: `docs/dev/typed_error_code_registry.yaml`
- Family matrix: `docs/dev/typed_error_mapping_matrix.yaml`

## Mapping Modes

- `typed_error_only`
  - Code stays internal to typed exception paths.
  - `diagnostic_equivalent_codes` must be empty.
- `diagnostic_equivalent_required`
  - Code family must map to runtime diagnostics taxonomy behavior.
  - Family matrix must declare non-empty `mapped_runtime_diagnostic_codes`.
  - Each registry entry in that family must declare non-empty `diagnostic_equivalent_codes`.

## Mandatory Family Rules

The following family rules are required and enforced:

- `E_PARSE_*` -> `typed_error_only`
- `E_ASSEMBLER_*` -> `typed_error_only`
- `E_INDEX_*` -> `typed_error_only`
- `E_MANIFEST_*` -> `typed_error_only`
- `E_SOLVER_CONFIG_*` -> `typed_error_only`

Additional families may be declared for related non-diagnostic typed paths if they are:

- deterministic,
- source-scoped via explicit `source_paths`, and
- mapping-mode compliant.

## CI/Policy Guard Requirements

Validation must fail if any of the following occur:

- typed code discovered in scoped source paths is missing from registry,
- duplicate typed code entries exist,
- registry declares code not discovered in scoped source paths,
- registry entry family is missing from matrix,
- registry code does not match its declared family prefix,
- required family from mandatory rules is missing or mode-mismatched,
- `typed_error_only` family/entry declares diagnostic mappings,
- `diagnostic_equivalent_required` family/entry omits required mappings,
- mapped diagnostic code is not cataloged in canonical diagnostics catalog.

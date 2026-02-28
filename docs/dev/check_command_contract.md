# Check Command Contract (P2-05)

This document defines the hardened `rfmna check` contract introduced in P2-05.

## Scope

- Command: `rfmna check <design>`
- Output modes:
  - `--format text` (default): existing line grammar mode
  - `--format json`: deterministic machine-readable mode

## Exit Mapping (Locked)

- Exit `0`: no error diagnostics (warnings-only and empty-diagnostic outcomes are allowed).
- Exit `2`: any error diagnostic outcome, including:
  - structural/topology preflight errors,
  - loader-boundary failures,
  - internal check-command failures.

No alternate `check` exit code is allowed.

## Text Output Compatibility

- Default `text` mode preserves the existing `DIAG severity=... stage=... code=... message=...` grammar.
- Diagnostic ordering remains deterministic via canonical diagnostic sorting.

## JSON Output Contract

- Canonical schema artifact: `docs/spec/schemas/check_output_v1.json`
- Schema identifier emitted by CLI JSON output: `docs/spec/schemas/check_output_v1.json`
- Required top-level keys:
  - `schema`
  - `schema_version`
  - `design`
  - `status`
  - `exit_code`
  - `diagnostics`
- Diagnostics are emitted in deterministic order and include deterministic witness payloads.
- Key order is stable in emitted JSON payloads.

## Versioning + Compatibility Policy

- `check_output_v1.json` is the versioned baseline.
- Future incompatible changes require a new schema file/version (`check_output_vN.json`).
- Additive extensions must not break existing required fields or ordering semantics for the declared schema version.

## Governance Classification Rule

- `run/check` exit-semantics changes are frozen-artifact #9 candidates and require full governance evidence:
  - semver bump,
  - decision record,
  - conformance updates,
  - migration note,
  - reproducibility impact statement.
- Additive JSON-surface changes that do not alter locked exit semantics are treated as non-frozen changes and must be explicitly documented in task evidence.

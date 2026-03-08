# Design Bundle Contract (`P3-01`)

This document defines the canonical in-repo design input contract introduced by `P3-01`.

## Scope

- Command paths:
  - `rfmna check <design>`
  - `rfmna run <design> --analysis ac`
- Canonical schema artifact:
  - `docs/spec/schemas/design_bundle_v1.json`
- Interim exclusion artifact:
  - `docs/dev/p3_loader_temporary_exclusions.yaml`
  - wheel-packaged copy of the same governed artifact, resolved by `rfmna.parser._loader_exclusions_runtime`

## Active Schema Selection Policy

- Active schema ID: `docs/spec/schemas/design_bundle_v1.json`
- Active schema version: `1`
- Design-bundle selection is resolved only from the explicit payload pair:
  - `schema`
  - `schema_version`
- No filename-based, extension-based, or highest-version inference is permitted.
- Future incompatible additions require a new schema artifact `design_bundle_vN.json` plus explicit loader support.

## Supported `design_bundle_v1` Interim Surface

- Analysis type: `ac`
- Frequency sweep grammar:
  - `mode` in `{linear,log}`
  - `start` / `stop` with deterministic units `{Hz,kHz,MHz,GHz}`
  - `points >= 1`
- Supported element kinds through the interim loader:
  - `R | RES | RESISTOR`: `nodes=[p,n]`, required `params.resistance_ohm`
  - `C | CAP | CAPACITOR`: `nodes=[p,n]`, required `params.capacitance_f`
  - `G | COND | CONDUCTANCE`: `nodes=[p,n]`, required `params.conductance_s`
  - `L | IND | INDUCTOR`: `nodes=[p,n]`, required `params.inductance_h`
  - `I | ISRC | CURRENT_SOURCE`: `nodes=[p_plus,p_minus]`, required `params.current_a`
  - `V | VSRC | VOLTAGE_SOURCE`: `nodes=[p_plus,p_minus]`, required `params.voltage_v`
  - `VCCS`: `nodes=[out_plus,out_minus,ctrl_plus,ctrl_minus]`, required `params.transconductance_s`
  - `VCVS | E`: `nodes=[out_plus,out_minus,ctrl_plus,ctrl_minus]`, required `params.gain_mu`
- Node ordering is part of the contract:
  - 2-node elements use the declared ordered pair exactly as stamped.
  - 4-node controlled sources use output pair first, control pair second.
- The schema artifact encodes the same kind-specific arity and required-parameter rules for all supported interim kinds.
- `design.elements[].kind` is intentionally closed to the supported tokens above plus the currently recognized interim-excluded `FD*`, `Y*`, and `Z*` tokens, so schema validation cannot produce arbitrary unknown kinds that the loader will later refuse.
- RF port declarations:
  - `id`
  - `p_plus`
  - `p_minus`
  - optional `z0_ohm`
- Static parameter resolution:
  - file-local numeric literals or expressions
  - deterministic precedence remains file-only for P3-01

## Interim vs Closure

- Interim merge criteria:
  - supported `design_bundle_v1` inputs execute through the in-repo loader for `check` and `run --analysis ac`
  - unsupported in-scope v4 capabilities are rejected with deterministic structured diagnostics and must be listed exhaustively in `docs/dev/p3_loader_temporary_exclusions.yaml`; empty or partial interim artifacts are policy-invalid while deferred capabilities remain active
  - in a source checkout, omitting `docs/dev/p3_loader_temporary_exclusions.yaml` is policy-invalid while the packaged runtime mirror still declares deferred capabilities; absence or an empty artifact is only valid once the packaged mirror is also closure-state empty. Installed distributions use the packaged copy directly
- Phase 3 closure criteria for `P3-01`:
  - no temporary exclusions remain
  - currently in-scope v4 inputs reach contract parity for loader-backed `check` and `run`
  - `docs/dev/p3_loader_temporary_exclusions.yaml` is absent or schema-valid empty, and the wheel-packaged copy correspondingly carries an empty exclusions list

## Diagnostics

- Loader failures emit structured diagnostics with deterministic ordering.
- Loader-path diagnostics do not reinterpret frozen assemble-stage codes as parse-stage diagnostics; when the loader surfaces an existing catalog code, its canonical `solver_stage` remains unchanged.
- Governed temporary exclusions carry explicit `check_diagnostic_code` and `run_diagnostic_code` fields; in the current interim policy both must use the exclusion-specific runtime code `E_CLI_DESIGN_EXCLUDED_CAPABILITY`, and the source artifact must match the packaged mirror entry-for-entry so source and installed behavior stay aligned.
- Run manifests hash the loaded design payload and resolved parameter map, not only the CLI path arguments.
- Schema/read/validation/exclusion failures use cataloged runtime codes:
  - `E_CLI_DESIGN_READ_FAILED`
  - `E_CLI_DESIGN_PARSE_FAILED`
  - `E_CLI_DESIGN_SCHEMA_UNSUPPORTED`
  - `E_CLI_DESIGN_SCHEMA_INVALID`
  - `E_CLI_DESIGN_VALUE_INVALID`
  - `E_CLI_DESIGN_EXCLUSION_POLICY_INVALID`
  - `E_CLI_DESIGN_EXCLUDED_CAPABILITY`
  - `E_IR_KIND_UNKNOWN` (canonical assemble-stage semantics preserved when surfaced by the loader)
  - `E_MODEL_PORT_Z0_COMPLEX` / `E_MODEL_PORT_Z0_NONPOSITIVE` (canonical assemble-stage semantics preserved when surfaced by the loader)
- Diagnostics remain taxonomy-complete:
  - `code`
  - `severity`
  - `message`
  - context
  - `suggested_action`
  - `solver_stage`
  - deterministic `witness`

## Frozen Impact Determination

- Frozen artifact `#9`:
  - touched by `src/rfmna/cli/main.py` integration
  - governed in this task with semver, DR, migration note, conformance update, and reproducibility impact statement
  - exit semantics remain unchanged: `check` still returns `0|2`, `run` still returns `0|1|2`
- Frozen artifact `#8`:
  - no semantic change
  - loader delegates frequency-grid generation to `rfmna.sweep_engine.frequency_grid`
  - conformance evidence: loader-backed linear/log grid tests
- Frozen artifact `#10`:
  - no semantic change
  - loader preserves existing CLI/API payload shapes and canonical ordering by reusing current sweep/RF export structures
  - conformance evidence: loader-backed check JSON compatibility and canonical RF port ordering tests
- Frozen artifact `#3`:
  - no semantic change
  - loader maps RF ports directly to existing `(p_plus, p_minus)` boundary semantics and forwards canonical `z0_ohm`
  - conformance evidence: loader-backed matched-port RF tests
- Frozen artifact `#11`:
  - no semantic change
  - loader feeds the existing sweep engine and retains fail-point sentinel handling unchanged
  - regression evidence: loader-backed sentinel/no-omission test

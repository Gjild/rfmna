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
- Hierarchy grammar additions in `design_bundle_v1`:
  - optional `design.macros`
  - optional `design.subcircuits`
  - optional `design.instances`
  - hierarchy parse-product helpers live in `rfmna.parser.design_bundle`; package-root `rfmna.parser` continues to expose the existing loader-oriented surface only
  - macro defaults may be partial; instance `params` are composed over macro defaults during hierarchy value validation and may supply required element-model values during hierarchy instantiation
  - macro default keys for supported interim kinds must stay within the target kind's legal model parameter names
  - macro instance override keys for supported interim kinds must stay within the target macro kind's legal model parameter names
  - subcircuit instance override keys must stay within the target subcircuit's declared `parameters`
  - hierarchy definition/reference ids are normalized deterministically before duplicate/conflict/reference checks
  - hierarchy id normalization applies Unicode NFC before uppercase/separator-collapsing so canonically equivalent spellings resolve identically
  - subcircuit-local element ids use that same normalization policy for duplicate detection
  - subcircuit-local element ids and hierarchy instance ids share one normalized local declaration namespace; cross-kind collisions are illegal
  - hierarchy instance ids must be unique within each scope after that normalization step
  - hierarchy instance node counts must match referenced macro `node_formals` or subcircuit `ports` arity
  - loader-backed `check` / `run` stay exclusion-aware: hierarchy diagnostics run first, interim exclusion checks run before loader-only hierarchy value validation, and only exclusion-free payloads proceed to declared macro/default and instantiated-body value validation
  - the module-local parse-product helper also applies hierarchy value validation and preserves the existing flat model/port/frequency validation contract; its carve-out is limited to loader exclusions and runnable top-level hierarchy admission
  - canonical parse-product JSON/hash normalize supported kind aliases (for example `RES`/`RESISTOR` -> `R`) while preserving existing flat-order semantics
  - active top-level hierarchy instantiation is parsed deterministically but remains non-runnable until `P3-03` elaboration support lands
  - when top-level `design.instances` are present, loader-backed `check` / `run` still perform hierarchy diagnostics, interim exclusion checks, and loader-only hierarchy value validation before falling back to `E_CLI_DESIGN_HIERARCHY_UNSUPPORTED`
  - the module-local parse-product helper accepts schema-valid parse-surface inputs even when interim loader exclusions would block `check`/`run`; loader-backed commands remain exclusion-aware

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
- Hierarchy declaration order remains part of the loaded design payload hash; canonical hierarchy parse-product hashing is tracked separately from manifest input hashing.
- Schema/read/validation/exclusion failures use cataloged runtime codes:
  - `E_CLI_DESIGN_READ_FAILED`
  - `E_CLI_DESIGN_PARSE_FAILED`
  - `E_CLI_DESIGN_SCHEMA_UNSUPPORTED`
  - `E_CLI_DESIGN_SCHEMA_INVALID`
  - `E_CLI_DESIGN_VALUE_INVALID`
  - `E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION`
  - `E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT`
  - `E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL`
  - `E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID`
  - `E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID`
  - `E_CLI_DESIGN_HIERARCHY_INSTANCE_ARITY_INVALID`
  - `E_CLI_DESIGN_HIERARCHY_REFERENCE_TYPE_MISMATCH`
  - `E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED`
  - `E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL`
  - `E_CLI_DESIGN_HIERARCHY_UNSUPPORTED`
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
  - loader preserves existing CLI/API payload shapes and canonical ordering by reusing current sweep/RF export structures, while canonical parse-product helpers stay module-local under `rfmna.parser.design_bundle`
  - conformance evidence: loader-backed check JSON compatibility and canonical RF port ordering tests
- Frozen artifact `#3`:
  - no semantic change
  - loader maps RF ports directly to existing `(p_plus, p_minus)` boundary semantics and forwards canonical `z0_ohm`
  - conformance evidence: loader-backed matched-port RF tests
- Frozen artifact `#11`:
  - no semantic change
  - loader feeds the existing sweep engine and retains fail-point sentinel handling unchanged
  - regression evidence: loader-backed sentinel/no-omission test
- Schema evolution reference:
  - `docs/dev/p3_02_design_bundle_schema_evolution.md`

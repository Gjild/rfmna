# Phase 1 Gate + Freeze Verification (`P1-00`)

This checklist is the Phase 1 baseline gate and freeze-boundary verification record.

## Baseline gate checks

- [ ] Re-run baseline tests with no semantic changes: `uv run pytest -m "unit or conformance"`.
- [ ] Confirm no unapproved normative/spec edits are included in task scope.
- [ ] For any approved normative/spec edit, confirm DR + migration note + conformance evidence are present.
- [ ] Confirm CI informational gate emits checklist status in logs and artifacts.

## Frozen artifact checklist (12)

1. Canonical element stamp equations (`docs/spec/stamp_appendix_v4_0_0.md`)
2. 2-port Y/Z block equations and stamping policy (`docs/spec/stamp_appendix_v4_0_0.md`)
3. Port current/voltage and wave conventions (`docs/spec/port_wave_conventions_v4_0_0.md`)
4. Residual formula and condition estimator definition (`docs/spec/v4_contract.md`, `docs/spec/thresholds_v4_0_0.yaml`)
5. Threshold table values and status bands (`docs/spec/thresholds_v4_0_0.yaml`)
6. Retry ladder order/defaults (`docs/spec/v4_contract.md`)
7. IR serialization/hash rules (`docs/spec/v4_contract.md`)
8. Frequency grammar and grid generation rules (`docs/spec/frequency_grid_and_sweep_rules_v4_0_0.md`)
9. CLI exit semantics and partial-sweep behavior (`docs/spec/v4_contract.md`)
10. Canonical API data shapes and ordering (`docs/spec/v4_contract.md`)
11. Fail-point sentinel policy (`docs/spec/v4_contract.md`)
12. Deterministic thread-control defaults (`.github/workflows/ci.yml`)

## Change-control requirements for any frozen-artifact change

No unapproved normative/spec edits are permitted in this gate.
Any approved change to the frozen artifact set requires all of the following:

- Semantic version bump.
- Decision record in `docs/spec/decision_records/`.
- Conformance updates.
- Migration note.
- Reproducibility-impact statement.

## Approved baseline publication (audited)

- Decision record: `docs/spec/decision_records/2026-02-27-phase1-freeze-baseline-v4-0-0.md`
- Migration note: `docs/spec/migration_notes/2026-02-27-v4-0-0-freeze-baseline.md`

## Evidence locations

- Checklist source: `docs/dev/phase1_gate.md`
- Process traceability artifact: `docs/dev/phase1_process_traceability.md`
- Informational CI gate: `.github/workflows/ci.yml` step `Phase 1 gate status (informational)`
- Executable verification: `tests/conformance/test_thread_controls_conformance.py::test_ci_workflow_always_surfaces_phase1_gate_status`
- Executable governance policy verification: `tests/conformance/test_phase1_governance_conformance.py`

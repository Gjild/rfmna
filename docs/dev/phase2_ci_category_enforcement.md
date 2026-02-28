# Phase 2 CI Category Enforcement Policy (`P2-09`)

This policy defines mandatory CI verification category selection for Phase 2.

## Mandatory categories

Phase 2 CI must execute all category selectors as independent, auditable lanes:

- `unit`
- `conformance`
- `property`
- `regression`
- `cross_check`

`cross_check` is an additional mandatory Phase 2 category supplementing the base set.

## Marker and strictness contract

- `pytest.ini` must keep `--strict-markers`.
- `pytest.ini` must declare markers for `unit`, `conformance`, `property`, `regression`, and `cross_check`.
- Unknown/undeclared markers are merge-blocking.

## Non-empty lane contract

Each category lane must include a non-empty collection guard:

- `uv run pytest -m <category> --collect-only -q`

Collection guard failure (zero collected tests) is merge-blocking.

## Thread-control conformance guard

CI must keep deterministic thread defaults fixed and run explicit conformance checks:

- `test_ci_workflow_declares_deterministic_thread_defaults`
- `test_envrc_declares_deterministic_thread_defaults`

## AGENTS alignment audit

CI audits that `AGENTS.md` still states `cross_check` as mandatory for Phase 2 robustness work.
If drift is detected, resolution must be tracked in a separate governance-labeled task/PR.
Do not bundle AGENTS policy edits into P2-09 implementation changes.

## Failure diagnostics artifacts

On CI failure, upload calibration/regression/cross-check diagnostics, including:

- regression and cross-check JUnit XML outputs
- calibration/tolerance policy evidence docs and YAML artifacts
- deterministic cross-check fixture references

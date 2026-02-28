<!-- docs/spec/decision_records/README.md -->
# Decision Records

This directory stores normative engineering decisions that affect frozen artifacts.

## When a decision record is required

Any change to frozen artifacts requires a record:

1. canonical element stamp equations,
2. 2-port Y/Z policy,
3. port/wave conventions,
4. residual formula or condition-indicator definition,
5. threshold values/status bands,
6. retry ladder order/defaults,
7. IR serialization/hash rules,
8. frequency grammar/grid algorithm,
9. CLI exit/partial-sweep behavior,
10. canonical API data shapes/ordering,
11. fail-point sentinel policy,
12. deterministic thread-control defaults.

## Required record sections

- Title
- Status (`proposed|accepted|superseded`)
- Date
- Context
- Decision
- Consequences
- Conformance impact
- Reproducibility impact
- Migration note (if applicable)
- Semver impact

## Filename convention

`YYYY-MM-DD-<short-slug>.md`

Example:

`2026-02-07-freeze-residual-threshold-bands.md`

## Active records

- `2026-02-27-phase1-freeze-baseline-v4-0-0.md`:
  establishes and governs the v4.0.0 frozen-artifact baseline used by Phase 1 work.
- `2026-02-28-p2-02-fallback-ladder-rf-hardening.md`:
  governs P2-02 fallback ladder and RF conversion hardening scope for v0.1.1.
- `2026-02-28-p2-06-regression-golden-tolerance-baseline-v0-1-2.md`:
  governs P2-06 regression golden/tolerance baseline promotion to normative merge-gating in v0.1.2.

# Codex rules for rfmna

1. Do not modify normative formulas/defaults/ordering without:
   - decision record in docs/spec/decision_records
   - semantic version bump
   - conformance test updates

2. No dense matrix path in solver core.
3. No symmetry/Hermitian assumptions anywhere.
4. Deterministic ordering required:
   nodes, aux vars, ports, diagnostics, output columns, witness payloads.
5. Failed-point sentinel policy is mandatory:
   - complex arrays: nan + 1j*nan
   - real arrays: nan
   - status: fail
   - diagnostics entry required
6. Every code change must include tests.
7. Math-sensitive changes require conformance tests.
8. Keep functions pure and side-effect free where specified in v4 contract.
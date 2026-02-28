<!-- docs/spec/frozen_artifacts_v4_0_0.md -->
# Frozen Artifacts v4.0.0 (Normative Index)

Version: **4.0.0**  
Status: **Normative index**

This index identifies artifacts that must not change silently.

## Frozen set

1. Canonical element stamp equations  
   - Source: `stamp_appendix_v4_0_0.md`
2. 2-port Y/Z block equations and stamping policy  
   - Source: `stamp_appendix_v4_0_0.md`
3. Port current/voltage and wave conventions  
   - Source: `port_wave_conventions_v4_0_0.md`
4. Residual formula and condition estimator definition  
   - Source: `v4_contract.md` + thresholds file
5. Threshold table values and status bands  
   - Source: `thresholds_v4_0_0.yaml`
6. Retry ladder order/defaults  
   - Source: `v4_contract.md` + solver config schema
7. IR serialization/hash rules  
   - Source: implementation + conformance references
8. Frequency grammar and grid generation rules  
   - Source: `frequency_grid_and_sweep_rules_v4_0_0.md`
9. CLI exit semantics and partial-sweep behavior  
   - Source: `v4_contract.md`
10. Canonical API data shapes and ordering  
    - Source: `v4_contract.md`
11. Fail-point sentinel policy  
    - Source: `v4_contract.md`
12. Deterministic thread-control defaults  
    - Source: repo env and CI config

## Change-control rule

Any change to this set requires:

- semantic version bump,
- decision record,
- conformance updates,
- migration note,
- reproducibility impact statement.

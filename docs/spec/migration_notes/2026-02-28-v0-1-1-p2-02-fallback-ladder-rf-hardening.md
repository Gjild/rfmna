<!-- docs/spec/migration_notes/2026-02-28-v0-1-1-p2-02-fallback-ladder-rf-hardening.md -->
# Migration Note: v0.1.1 P2-02 fallback ladder and RF hardening

Date: `2026-02-28`  
Related DR: `docs/spec/decision_records/2026-02-28-p2-02-fallback-ladder-rf-hardening.md`

## Summary

This release hardens fallback ladder execution across sweep and RF API paths and locks solver reproducibility snapshot defaults for manifest outputs.

## Consumer impact

- `solver_config_snapshot` remains present and now includes schema-stable retry controls, conversion-math controls, and attempt-trace summary defaults.
- Conversion-math APIs preserve explicit fail diagnostics for singular/ill-conditioned conversions and do not use gmin regularization by default conversion solver wiring.
- Standalone RF extraction APIs accept `node_voltage_count` for eligible MNA-system solves that include internal nodes not represented in port indices.
- `rfmna run` exit semantics are unchanged.

## Required action

1. If standalone RF extraction calls operate on matrices with internal node-voltage rows not present in port boundaries, pass explicit `node_voltage_count`.
2. If custom conversion solve hooks are used, ensure they do not introduce gmin regularization in conversion-math paths.

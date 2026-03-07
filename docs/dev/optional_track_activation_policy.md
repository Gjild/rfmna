# Optional-Track Activation Policy (`P3-00`)

This policy defines objective activation criteria for the usage-driven Phase 3 optional tracks.

## Tracks

- `p3_10_cccs_ccvs`
- `p3_11_mutual_inductance`

## Activation predicate

The optional-track gate is active if either condition is true:

- the track state in `docs/dev/optional_track_activation.yaml` is `activated`, or
- base-ref diff path detection against the reserved optional-track file set in `docs/dev/phase3_contract_surface_governance_rules.yaml` marks the track scope as touched, including token-matched shared integration files such as factories and shared parser/element modules.

Touching optional-track scope while the track remains deferred is merge-blocking.

## Deterministic time contract

- `usage_evidence_date` format: `YYYY-MM-DD`
- Time basis: UTC only
- Comparison anchor: the `base-ref` commit timestamp (`git show -s --format=%cI <base-ref>`) projected to its UTC calendar date
- Freshness window: `90` days inclusive
- Evidence dates after the base-ref UTC date are invalid

## Evidence and approval requirements

When a track is active, all of the following are mandatory:

- `usage_evidence_source.evidence_type` is one of the allowed types declared for the track in `docs/dev/phase3_contract_surface_governance_rules.yaml`
- `usage_evidence_source.reference` is non-empty
- `usage_evidence_date` is present and fresh under the 90-day UTC rule
- `activation_rationale` is non-empty
- `approval_record.status` is `approved`
- `approval_record.approved_by`, `approval_record.decision_date`, and `approval_record.decision_ref` are present
- `impacted_frozen_ids` is non-empty for activated tracks
- On the activation PR, `impacted_frozen_ids` exactly matches `docs/dev/change_scope.yaml`
- Once a track is already active at `base-ref`, its activation record is immutable for anti-tamper purposes unless a separate governed policy change updates the baseline artifacts first

## Enforcement entry point

- `python -m rfmna.governance.phase3_gate --sub-gate optional-track`

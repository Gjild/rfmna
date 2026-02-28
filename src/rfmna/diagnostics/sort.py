from __future__ import annotations

import json
from collections.abc import Iterable

from .models import DiagnosticEvent, Severity, SolverStage

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.ERROR: 0,
    Severity.WARNING: 1,
}

_STAGE_RANK: dict[SolverStage, int] = {
    SolverStage.PARSE: 0,
    SolverStage.PREFLIGHT: 1,
    SolverStage.ASSEMBLE: 2,
    SolverStage.SOLVE: 3,
    SolverStage.POSTPROCESS: 4,
}


def canonical_witness_json(witness: object | None) -> str:
    if witness is None:
        return ""
    return json.dumps(witness, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _element_id_sort_key(element_id: str | None) -> tuple[int, str]:
    if element_id is None:
        return (1, "")
    return (0, element_id)


def _point_index_sort_key(event: DiagnosticEvent) -> tuple[int, int, int, int]:
    freq_none_rank = 1 if event.frequency_index is None else 0
    freq_value_or_0 = event.frequency_index if event.frequency_index is not None else 0
    sweep_none_rank = 1 if event.sweep_index is None else 0
    sweep_value_or_0 = event.sweep_index if event.sweep_index is not None else 0
    return (freq_none_rank, freq_value_or_0, sweep_none_rank, sweep_value_or_0)


def diagnostic_sort_key(
    event: DiagnosticEvent,
) -> tuple[int, int, str, tuple[int, str], tuple[int, int, int, int], str, str]:
    return (
        _SEVERITY_RANK[event.severity],
        _STAGE_RANK[event.solver_stage],
        event.code,
        _element_id_sort_key(event.element_id),
        _point_index_sort_key(event),
        event.message,
        canonical_witness_json(event.witness),
    )


def sort_diagnostics(events: Iterable[DiagnosticEvent]) -> list[DiagnosticEvent]:
    return sorted(events, key=diagnostic_sort_key)

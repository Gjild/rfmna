from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]

from rfmna.diagnostics import canonical_witness_json
from rfmna.rf_metrics.boundary import (
    BoundaryRequest,
    PortBoundary,
    apply_boundary_conditions,
    apply_current_boundaries,
    apply_voltage_boundaries,
)

pytestmark = pytest.mark.unit

_PERMUTATION_REPEATS = 40


def _complex_vector_payload(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def _complex_matrix_payload(matrix: csc_matrix) -> list[list[list[float]]]:
    dense = np.asarray(matrix.toarray(), dtype=np.complex128)
    return [[_complex_vector_payload(row) for row in dense]][0]


def _result_json(result: object) -> str:
    payload = result
    assert hasattr(payload, "matrix")
    assert hasattr(payload, "rhs")
    assert hasattr(payload, "metadata")
    assert hasattr(payload, "diagnostics")
    matrix = payload.matrix
    rhs = payload.rhs
    metadata = payload.metadata
    diagnostics = payload.diagnostics

    serial = {
        "matrix": None
        if matrix is None
        else _complex_matrix_payload(csc_matrix(np.asarray(matrix.toarray(), dtype=np.complex128))),
        "rhs": None
        if rhs is None
        else _complex_vector_payload(np.asarray(rhs, dtype=np.complex128)),
        "metadata": {
            "canonical_port_ids": list(metadata.canonical_port_ids),
            "requested": [
                {
                    "kind": request.kind,
                    "port_id": request.port_id,
                    "source": request.source,
                    "value": [float(request.value.real), float(request.value.imag)],
                }
                for request in metadata.requested
            ],
            "applied": [
                {
                    "aux_unknown_index": applied.aux_unknown_index,
                    "current_into_dut_from_aux_sign": applied.current_into_dut_from_aux_sign,
                    "equation_row_index": applied.equation_row_index,
                    "kind": applied.kind,
                    "p_minus_index": applied.p_minus_index,
                    "p_plus_index": applied.p_plus_index,
                    "port_id": applied.port_id,
                    "source": applied.source,
                    "value": [float(applied.value.real), float(applied.value.imag)],
                }
                for applied in metadata.applied
            ],
        },
        "diagnostics": [
            {
                "code": diag.code,
                "message": diag.message,
                "severity": diag.severity.value,
                "solver_stage": diag.solver_stage.value,
                "witness": canonical_witness_json(diag.witness),
            }
            for diag in diagnostics
        ],
    }
    return json.dumps(serial, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_one_port_voltage_boundary_augmentation_and_current_sign_metadata() -> None:
    matrix = csc_matrix(np.asarray([[2.0 + 0.0j]], dtype=np.complex128))
    rhs = np.asarray([0.0 + 0.0j], dtype=np.complex128)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    result = apply_voltage_boundaries(
        matrix,
        rhs,
        ports,
        imposed_port_voltages=(("P1", 1.0 + 0.0j),),
    )

    assert result.diagnostics == ()
    assert result.matrix is not None
    assert result.rhs is not None
    assert result.matrix.shape == (2, 2)
    assert np.allclose(
        np.asarray(result.matrix.toarray(), dtype=np.complex128),
        np.asarray([[2.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128),
    )
    assert np.allclose(result.rhs, np.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128))

    solved = np.asarray(spsolve(result.matrix, result.rhs), dtype=np.complex128)
    assert len(result.metadata.applied) == 1
    applied = result.metadata.applied[0]
    assert applied.aux_unknown_index is not None
    assert applied.current_into_dut_from_aux_sign is not None
    port_current_into_dut = (
        applied.current_into_dut_from_aux_sign * solved[applied.aux_unknown_index]
    )
    assert np.isclose(port_current_into_dut.real, 2.0)


def test_two_port_current_boundary_with_inactive_open_boundary() -> None:
    matrix = csc_matrix(
        np.asarray([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 4.0 + 0.0j]], dtype=np.complex128)
    )
    rhs = np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    ports = (
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    )

    result = apply_current_boundaries(
        matrix,
        rhs,
        ports,
        imposed_port_currents=(("P1", 1.0 + 0.0j),),
        inactive_port_ids=("P2",),
    )

    assert result.diagnostics == ()
    assert result.matrix is not None
    assert result.rhs is not None
    assert result.matrix.shape == (2, 2)
    assert np.allclose(
        np.asarray(result.matrix.toarray(), dtype=np.complex128), np.asarray(matrix.toarray())
    )
    assert np.allclose(result.rhs, np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128))
    assert [entry.port_id for entry in result.metadata.applied] == ["P1", "P2"]
    assert [entry.kind for entry in result.metadata.applied] == ["current", "current"]


def test_inconsistent_boundary_values_emit_topology_diagnostic() -> None:
    matrix = csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128))
    rhs = np.asarray([0.0 + 0.0j], dtype=np.complex128)
    ports = (PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),)

    result = apply_voltage_boundaries(
        matrix,
        rhs,
        ports,
        imposed_port_voltages=(("P1", 1.0 + 0.0j),),
        inactive_port_ids=("P1",),
        inactive_voltage=0.0 + 0.0j,
    )

    assert result.matrix is None
    assert result.rhs is None
    assert [diag.code for diag in result.diagnostics] == ["E_TOPO_RF_BOUNDARY_INCONSISTENT"]


def test_singular_redundant_voltage_boundaries_emit_numeric_diagnostic() -> None:
    matrix = csc_matrix(np.asarray([[1.0 + 0.0j]], dtype=np.complex128))
    rhs = np.asarray([0.0 + 0.0j], dtype=np.complex128)
    ports = (
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        PortBoundary(port_id="P2", p_plus_index=0, p_minus_index=None),
    )

    result = apply_voltage_boundaries(
        matrix,
        rhs,
        ports,
        imposed_port_voltages=(("P1", 1.0 + 0.0j),),
        inactive_port_ids=("P2",),
        inactive_voltage=1.0 + 0.0j,
    )

    assert result.matrix is None
    assert result.rhs is None
    assert [diag.code for diag in result.diagnostics] == ["E_NUM_RF_BOUNDARY_SINGULAR"]


def test_boundary_injection_is_permutation_invariant() -> None:
    matrix = csc_matrix(
        np.asarray([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)
    )
    rhs = np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    base_ports = [
        PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
    ]
    base_imposed = [
        BoundaryRequest(port_id="P1", kind="current", value=1.0 + 0.0j, source="imposed"),
    ]
    base_inactive = [
        BoundaryRequest(port_id="P2", kind="current", value=0.0 + 0.0j, source="inactive"),
    ]
    baseline = apply_boundary_conditions(
        matrix,
        rhs,
        tuple(base_ports),
        imposed=tuple(base_imposed),
        inactive=tuple(base_inactive),
    )
    baseline_json = _result_json(baseline)

    rng = Random(0)
    for _ in range(_PERMUTATION_REPEATS):
        ports = list(base_ports)
        imposed = list(base_imposed)
        inactive = list(base_inactive)
        rng.shuffle(ports)
        rng.shuffle(imposed)
        rng.shuffle(inactive)
        current = apply_boundary_conditions(
            matrix,
            rhs,
            tuple(ports),
            imposed=tuple(imposed),
            inactive=tuple(inactive),
        )
        assert _result_json(current) == baseline_json

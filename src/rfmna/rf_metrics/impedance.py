from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfmna.diagnostics import DiagnosticEvent, PortContext, Severity, SolverStage, sort_diagnostics

from .boundary import PortBoundary
from .z_params import ZAssemblePointFn, ZSolvePointFn, extract_z_parameters

type ImpedancePointStatus = Literal["pass", "degraded", "fail"]

_IMPEDANCE_ELEMENT_ID = "rf_impedance"
_IMPEDANCE_PAYLOAD_RANK = 1
_IMPEDANCE_UNDEFINED_CODE = "E_NUM_IMPEDANCE_UNDEFINED"
_BOUNDARY_INCONSISTENT_CODE = "E_TOPO_RF_BOUNDARY_INCONSISTENT"


@dataclass(frozen=True, slots=True)
class ZinZoutResult:
    frequencies_hz: NDArray[np.float64]
    port_ids: tuple[str, ...]
    input_port_id: str
    output_port_id: str
    zin: NDArray[np.complex128]
    zout: NDArray[np.complex128]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...]

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        zin_values = np.asarray(self.zin, dtype=np.complex128)
        zout_values = np.asarray(self.zout, dtype=np.complex128)
        status_values = np.asarray(self.status, dtype=np.str_)

        if frequencies.ndim != _IMPEDANCE_PAYLOAD_RANK:
            raise ValueError("frequencies_hz must be one-dimensional")
        if zin_values.ndim != _IMPEDANCE_PAYLOAD_RANK:
            raise ValueError("zin must be one-dimensional")
        if zout_values.ndim != _IMPEDANCE_PAYLOAD_RANK:
            raise ValueError("zout must be one-dimensional")
        if status_values.ndim != _IMPEDANCE_PAYLOAD_RANK:
            raise ValueError("status must be one-dimensional")

        n_points = int(frequencies.shape[0])
        if zin_values.shape != (n_points,):
            raise ValueError("zin shape must match [n_points]")
        if zout_values.shape != (n_points,):
            raise ValueError("zout shape must match [n_points]")
        if status_values.shape != (n_points,):
            raise ValueError("status shape must match [n_points]")
        if len(self.diagnostics_by_point) != n_points:
            raise ValueError("diagnostics_by_point length must equal n_points")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "zin", zin_values)
        object.__setattr__(self, "zout", zout_values)
        object.__setattr__(self, "status", status_values)
        object.__setattr__(
            self,
            "diagnostics_by_point",
            tuple(tuple(sort_diagnostics(point)) for point in self.diagnostics_by_point),
        )


def extract_zin_zout(  # noqa: PLR0913
    freq_hz: Sequence[float] | NDArray[np.float64],
    ports: Sequence[PortBoundary],
    assemble_point: ZAssemblePointFn,
    *,
    solve_point: ZSolvePointFn | None = None,
    node_voltage_count: int | None = None,
    input_port_id: str | None = None,
    output_port_id: str | None = None,
) -> ZinZoutResult:
    z_result = extract_z_parameters(
        freq_hz,
        ports,
        assemble_point,
        solve_point=solve_point,
        node_voltage_count=node_voltage_count,
        extraction_mode="direct",
    )

    frequencies = np.asarray(z_result.frequencies_hz, dtype=np.float64)
    n_points = int(frequencies.shape[0])
    complex_nan = np.complex128(complex(float("nan"), float("nan")))
    zin = np.full(n_points, complex_nan, dtype=np.complex128)
    zout = np.full(n_points, complex_nan, dtype=np.complex128)
    status = np.full(n_points, "fail", dtype=np.dtype("<U8"))
    diagnostics_lists: list[list[DiagnosticEvent]] = [
        list(point) for point in z_result.diagnostics_by_point
    ]

    canonical_ports = tuple(z_result.port_ids)
    selected_input = canonical_ports[0] if input_port_id is None else input_port_id
    selected_output = canonical_ports[-1] if output_port_id is None else output_port_id
    selection_diagnostics = _selection_diagnostics(
        canonical_ports=canonical_ports,
        input_port_id=selected_input,
        output_port_id=selected_output,
    )

    if selection_diagnostics:
        for point_index, frequency in enumerate(frequencies):
            frequency_hz = float(frequency)
            for diagnostic in selection_diagnostics:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code=diagnostic.code,
                        message=diagnostic.message,
                        suggested_action=diagnostic.suggested_action,
                        stage=SolverStage.PREFLIGHT,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        port_id=diagnostic.port_id,
                        witness=diagnostic.witness,
                    )
                )
        return ZinZoutResult(
            frequencies_hz=frequencies,
            port_ids=canonical_ports,
            input_port_id=selected_input,
            output_port_id=selected_output,
            zin=zin,
            zout=zout,
            status=status,
            diagnostics_by_point=tuple(
                tuple(sort_diagnostics(point)) for point in diagnostics_lists
            ),
        )

    input_index = canonical_ports.index(selected_input)
    output_index = canonical_ports.index(selected_output)

    for point_index, frequency in enumerate(frequencies):
        frequency_hz = float(frequency)
        point_status = str(z_result.status[point_index])
        if point_status not in {"pass", "degraded", "fail"}:
            raise ValueError(f"status[{point_index}] must be one of: pass, degraded, fail")
        if point_status == "fail":
            status[point_index] = "fail"
            if not diagnostics_lists[point_index]:
                diagnostics_lists[point_index].append(
                    _point_error(
                        code="E_NUM_SOLVE_FAILED",
                        message="Zin/Zout fail sentinel applied for this point",
                        suggested_action="inspect upstream Z extraction diagnostics for this point",
                        stage=SolverStage.POSTPROCESS,
                        point_index=point_index,
                        frequency_hz=frequency_hz,
                        witness={
                            "input_port_id": selected_input,
                            "output_port_id": selected_output,
                            "reason": "upstream_point_failed",
                        },
                    )
                )
            continue

        zin_value = complex(z_result.z[point_index, input_index, input_index])
        zout_value = complex(z_result.z[point_index, output_index, output_index])
        if not _is_finite_complex(zin_value) or not _is_finite_complex(zout_value):
            diagnostics_lists[point_index].append(
                _point_error(
                    code=_IMPEDANCE_UNDEFINED_CODE,
                    message="Zin/Zout undefined due non-finite impedance extraction result",
                    suggested_action="inspect boundary constraints and conditioning for selected ports",
                    stage=SolverStage.POSTPROCESS,
                    point_index=point_index,
                    frequency_hz=frequency_hz,
                    witness={
                        "input_port_id": selected_input,
                        "output_port_id": selected_output,
                        "zin": _complex_witness(zin_value),
                        "zout": _complex_witness(zout_value),
                    },
                )
            )
            status[point_index] = "fail"
            continue

        zin[point_index] = zin_value
        zout[point_index] = zout_value
        status[point_index] = point_status

    return ZinZoutResult(
        frequencies_hz=frequencies,
        port_ids=canonical_ports,
        input_port_id=selected_input,
        output_port_id=selected_output,
        zin=zin,
        zout=zout,
        status=status,
        diagnostics_by_point=tuple(tuple(sort_diagnostics(point)) for point in diagnostics_lists),
    )


@dataclass(frozen=True, slots=True)
class _SelectionDiagnostic:
    code: str
    message: str
    suggested_action: str
    port_id: str | None
    witness: dict[str, object]


def _selection_diagnostics(
    *,
    canonical_ports: tuple[str, ...],
    input_port_id: str,
    output_port_id: str,
) -> tuple[_SelectionDiagnostic, ...]:
    diagnostics: list[_SelectionDiagnostic] = []
    available_port_ids = list(canonical_ports)

    if input_port_id not in canonical_ports:
        diagnostics.append(
            _SelectionDiagnostic(
                code=_BOUNDARY_INCONSISTENT_CODE,
                message=f"input_port_id '{input_port_id}' is not a declared canonical port",
                suggested_action="select input_port_id from declared canonical ports",
                port_id=input_port_id if input_port_id else None,
                witness={
                    "available_port_ids": available_port_ids,
                    "issue": "unknown_input_port_id",
                    "requested_input_port_id": input_port_id,
                },
            )
        )
    if output_port_id not in canonical_ports:
        diagnostics.append(
            _SelectionDiagnostic(
                code=_BOUNDARY_INCONSISTENT_CODE,
                message=f"output_port_id '{output_port_id}' is not a declared canonical port",
                suggested_action="select output_port_id from declared canonical ports",
                port_id=output_port_id if output_port_id else None,
                witness={
                    "available_port_ids": available_port_ids,
                    "issue": "unknown_output_port_id",
                    "requested_output_port_id": output_port_id,
                },
            )
        )
    return tuple(diagnostics)


def _is_finite_complex(value: complex) -> bool:
    return isfinite(value.real) and isfinite(value.imag)


def _complex_witness(value: complex) -> dict[str, float]:
    return {
        "imag": float(value.imag),
        "real": float(value.real),
    }


def _point_error(  # noqa: PLR0913
    *,
    code: str,
    message: str,
    suggested_action: str,
    stage: SolverStage,
    point_index: int,
    frequency_hz: float,
    port_id: str | None = None,
    witness: object | None = None,
) -> DiagnosticEvent:
    return DiagnosticEvent(
        code=code,
        severity=Severity.ERROR,
        message=message,
        suggested_action=suggested_action,
        solver_stage=stage,
        element_id=_IMPEDANCE_ELEMENT_ID,
        port_context=PortContext(port_id=port_id) if port_id is not None else None,
        frequency_hz=frequency_hz,
        frequency_index=point_index,
        witness=witness,
    )

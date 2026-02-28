from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfmna.diagnostics import DiagnosticEvent, sort_diagnostics
from rfmna.rf_metrics.boundary import PortBoundary
from rfmna.rf_metrics.s_params import SParameterResult
from rfmna.rf_metrics.y_params import YParameterResult
from rfmna.rf_metrics.z_params import ZParameterResult

type RFMetricName = Literal["y", "z", "s", "zin", "zout"]
type SConversionSource = Literal["from_z", "from_y"]

_RF_METRIC_ORDER: dict[RFMetricName, int] = {
    "y": 0,
    "z": 1,
    "s": 2,
    "zin": 3,
    "zout": 4,
}


def _canonical_metric_names(metrics: Sequence[RFMetricName]) -> tuple[RFMetricName, ...]:
    if not metrics:
        raise ValueError("rf metrics must be non-empty")
    unique = dict.fromkeys(metrics)
    metric_names = tuple(unique)
    for metric_name in metric_names:
        if metric_name not in _RF_METRIC_ORDER:
            raise ValueError("rf metrics must be chosen from: y,z,s,zin,zout")
    return tuple(sorted(metric_names, key=lambda metric_name: _RF_METRIC_ORDER[metric_name]))


@dataclass(frozen=True, slots=True)
class SweepRFRequest:
    ports: tuple[PortBoundary, ...]
    metrics: tuple[RFMetricName, ...]
    z0_ohm: object = 50.0
    s_conversion_source: SConversionSource = "from_z"
    input_port_id: str | None = None
    output_port_id: str | None = None

    def __post_init__(self) -> None:
        canonical_ports = tuple(sorted(self.ports, key=lambda port: port.port_id))
        if not canonical_ports:
            raise ValueError("rf ports must be non-empty")
        object.__setattr__(self, "ports", canonical_ports)
        object.__setattr__(self, "metrics", _canonical_metric_names(self.metrics))
        if self.s_conversion_source not in {"from_z", "from_y"}:
            raise ValueError("s_conversion_source must be 'from_z' or 'from_y'")


@dataclass(frozen=True, slots=True)
class SweepRFScalarResult:
    frequencies_hz: NDArray[np.float64]
    port_ids: tuple[str, ...]
    port_id: str
    metric_name: Literal["zin", "zout"]
    values: NDArray[np.complex128]
    status: NDArray[np.str_]
    diagnostics_by_point: tuple[tuple[DiagnosticEvent, ...], ...]

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.complex128)
        status_values = np.asarray(self.status, dtype=np.str_)

        if frequencies.ndim != 1:
            raise ValueError("frequencies_hz must be one-dimensional")
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if status_values.ndim != 1:
            raise ValueError("status must be one-dimensional")

        n_points = int(frequencies.shape[0])
        if values.shape != (n_points,):
            raise ValueError("values shape must match [n_points]")
        if status_values.shape != (n_points,):
            raise ValueError("status shape must match [n_points]")
        if len(self.diagnostics_by_point) != n_points:
            raise ValueError("diagnostics_by_point length must equal n_points")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "status", status_values)
        object.__setattr__(
            self,
            "diagnostics_by_point",
            tuple(tuple(sort_diagnostics(point)) for point in self.diagnostics_by_point),
        )


type SweepRFMetricPayload = (
    YParameterResult | ZParameterResult | SParameterResult | SweepRFScalarResult
)


@dataclass(frozen=True, slots=True)
class SweepRFPayloads:
    by_metric: tuple[tuple[RFMetricName, SweepRFMetricPayload], ...]

    def __post_init__(self) -> None:
        if not self.by_metric:
            raise ValueError("rf payloads must be non-empty")

        keys = [metric_name for metric_name, _ in self.by_metric]
        if len(set(keys)) != len(keys):
            raise ValueError("rf payload metric keys must be unique")

        ordered = tuple(
            sorted(
                self.by_metric,
                key=lambda pair: _RF_METRIC_ORDER[pair[0]],
            )
        )
        object.__setattr__(self, "by_metric", ordered)

    @property
    def metric_names(self) -> tuple[RFMetricName, ...]:
        return tuple(metric_name for metric_name, _ in self.by_metric)

    def get(self, metric_name: RFMetricName) -> SweepRFMetricPayload | None:
        for current_name, payload in self.by_metric:
            if current_name == metric_name:
                return payload
        return None

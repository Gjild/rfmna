from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rfmna.rf_metrics.s_params import SParameterResult
from rfmna.rf_metrics.y_params import YParameterResult
from rfmna.rf_metrics.z_params import ZParameterResult
from rfmna.sweep_engine.frequency_grid import hash_frequency_grid
from rfmna.sweep_engine.run import SweepResult
from rfmna.sweep_engine.types import (
    SweepRFMetricPayload,
    SweepRFPayloads,
    SweepRFRequest,
    SweepRFScalarResult,
)

RF_EXPORT_SCHEMA_VERSION = "rf_export_v1"
_CSV_FLOAT_FORMAT = ".12e"
_CSV_METADATA_PREFIX = "# rf_export_metadata="
_CSV_NEWLINE = "\n"
_MATRIX_PAYLOAD_RANK = 3


def build_rf_export_metadata(
    *,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    rf_request: SweepRFRequest,
    convention_tag: str,
    schema_version: str = RF_EXPORT_SCHEMA_VERSION,
) -> Mapping[str, object]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional")
    if not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz entries must be finite")
    if not convention_tag:
        raise ValueError("convention_tag must be non-empty")
    if not schema_version:
        raise ValueError("schema_version must be non-empty")

    port_order = tuple(port.port_id for port in rf_request.ports)
    z0_values = _canonical_z0_values(z0_ohm=rf_request.z0_ohm, n_ports=len(port_order))
    metadata = {
        "schema_version": schema_version,
        "port_order": list(port_order),
        "z0": z0_values,
        "convention_tag": convention_tag,
        "grid_hash": hash_frequency_grid([float(value) for value in frequencies.tolist()]),
    }
    return metadata


def export_rf_npz_bytes(
    *,
    sweep_result: SweepResult,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    rf_request: SweepRFRequest,
    convention_tag: str,
    schema_version: str = RF_EXPORT_SCHEMA_VERSION,
) -> bytes:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    payloads = _require_rf_payloads(sweep_result)
    _validate_point_alignment(n_points=sweep_result.n_points, frequencies=frequencies)
    metadata = build_rf_export_metadata(
        frequencies_hz=frequencies,
        rf_request=rf_request,
        convention_tag=convention_tag,
        schema_version=schema_version,
    )
    metadata_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)

    npz_items: list[tuple[str, NDArray[np.generic]]] = [
        ("schema_version", np.asarray(schema_version)),
        ("rf_metadata_json", np.asarray(metadata_json)),
        ("frequencies_hz", frequencies),
    ]
    for metric_name in payloads.metric_names:
        payload = payloads.get(metric_name)
        assert payload is not None
        value_array, status_array = _payload_arrays(
            metric_name=metric_name, payload=payload, n_points=sweep_result.n_points
        )
        npz_items.append((f"rf_{metric_name}_values", value_array))
        npz_items.append((f"rf_{metric_name}_status", status_array))

    output = io.BytesIO()
    npz_payload: dict[str, object] = {name: value for name, value in npz_items}
    savez = cast(Any, np.savez)
    savez(output, **npz_payload)
    return output.getvalue()


def export_rf_csv_text(
    *,
    sweep_result: SweepResult,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    rf_request: SweepRFRequest,
    convention_tag: str,
    schema_version: str = RF_EXPORT_SCHEMA_VERSION,
) -> str:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    payloads = _require_rf_payloads(sweep_result)
    _validate_point_alignment(n_points=sweep_result.n_points, frequencies=frequencies)
    metadata = build_rf_export_metadata(
        frequencies_hz=frequencies,
        rf_request=rf_request,
        convention_tag=convention_tag,
        schema_version=schema_version,
    )
    metadata_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)

    header = ["point_index", "frequency_hz"]
    status_columns: list[str] = []
    value_columns: list[str] = []
    extractors: list[tuple[str, Callable[[int], complex]]] = []
    for metric_name in payloads.metric_names:
        payload = payloads.get(metric_name)
        assert payload is not None
        status_columns.append(f"status_{metric_name}")
        metric_extractors = _metric_extractors(
            metric_name=metric_name,
            payload=payload,
            port_order=tuple(cast(list[str], metadata["port_order"])),
            n_points=sweep_result.n_points,
        )
        for column_name, extractor in metric_extractors:
            value_columns.extend((f"{column_name}_real", f"{column_name}_imag"))
            extractors.append((metric_name, extractor))

    header.extend(status_columns)
    header.extend(value_columns)

    lines = [f"{_CSV_METADATA_PREFIX}{metadata_json}", ",".join(header)]
    for point_index in range(sweep_result.n_points):
        row = [str(point_index), _format_float(float(frequencies[point_index]))]
        for metric_name in payloads.metric_names:
            payload = payloads.get(metric_name)
            assert payload is not None
            status_array = _payload_status_array(
                metric_name=metric_name, payload=payload, n_points=sweep_result.n_points
            )
            row.append(str(status_array[point_index]))
        for _, extractor in extractors:
            value = extractor(point_index)
            row.append(_format_float(value.real))
            row.append(_format_float(value.imag))
        lines.append(",".join(row))
    return _CSV_NEWLINE.join(lines) + _CSV_NEWLINE


def export_rf_csv_bytes(
    *,
    sweep_result: SweepResult,
    frequencies_hz: Sequence[float] | NDArray[np.float64],
    rf_request: SweepRFRequest,
    convention_tag: str,
    schema_version: str = RF_EXPORT_SCHEMA_VERSION,
) -> bytes:
    text = export_rf_csv_text(
        sweep_result=sweep_result,
        frequencies_hz=frequencies_hz,
        rf_request=rf_request,
        convention_tag=convention_tag,
        schema_version=schema_version,
    )
    return text.encode("utf-8")


def _require_rf_payloads(sweep_result: SweepResult) -> SweepRFPayloads:
    if sweep_result.rf_payloads is None:
        raise ValueError("sweep_result.rf_payloads must be present for RF export")
    return sweep_result.rf_payloads


def _validate_point_alignment(*, n_points: int, frequencies: NDArray[np.float64]) -> None:
    if n_points != int(frequencies.shape[0]):
        raise ValueError("n_points must match frequencies_hz length")


def _canonical_z0_values(*, z0_ohm: object, n_ports: int) -> list[float]:
    if np.isscalar(z0_ohm):
        scalar_value = _real_positive_z0(z0_ohm)
        return [scalar_value] * n_ports

    vector = np.asarray(z0_ohm)
    if vector.ndim == 0:
        scalar_value = _real_positive_z0(vector.item())
        return [scalar_value] * n_ports
    if vector.ndim != 1:
        raise ValueError("z0_ohm must be scalar or one-dimensional")
    if int(vector.shape[0]) != n_ports:
        raise ValueError("z0_ohm vector length must match port count")
    return [_real_positive_z0(value) for value in vector.tolist()]


def _real_positive_z0(value: object) -> float:
    try:
        complex_value = complex(cast(complex | float | int | str, value))
    except TypeError, ValueError:
        raise ValueError("z0_ohm values must be numeric") from None
    if complex_value.imag != 0.0:
        raise ValueError("z0_ohm values must be real")
    numeric = float(complex_value.real)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("z0_ohm values must be finite and > 0")
    return numeric


def _payload_arrays(
    *,
    metric_name: str,
    payload: SweepRFMetricPayload,
    n_points: int,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    status_array = _payload_status_array(
        metric_name=metric_name, payload=payload, n_points=n_points
    )
    if metric_name == "y":
        if not isinstance(payload, YParameterResult):
            raise TypeError("y payload must be YParameterResult")
        values = np.asarray(payload.y, dtype=np.complex128)
    elif metric_name == "z":
        if not isinstance(payload, ZParameterResult):
            raise TypeError("z payload must be ZParameterResult")
        values = np.asarray(payload.z, dtype=np.complex128)
    elif metric_name == "s":
        if not isinstance(payload, SParameterResult):
            raise TypeError("s payload must be SParameterResult")
        values = np.asarray(payload.s, dtype=np.complex128)
    elif metric_name in {"zin", "zout"}:
        if not isinstance(payload, SweepRFScalarResult):
            raise TypeError("scalar RF payload must be SweepRFScalarResult")
        values = np.asarray(payload.values, dtype=np.complex128)
    else:
        raise ValueError(f"unsupported rf metric '{metric_name}'")
    if values.shape[0] != n_points:
        raise ValueError("payload values first dimension must match sweep n_points")
    return (values, status_array)


def _payload_status_array(
    *,
    metric_name: str,
    payload: SweepRFMetricPayload,
    n_points: int,
) -> NDArray[np.str_]:
    if metric_name == "y":
        if not isinstance(payload, YParameterResult):
            raise TypeError("y payload must be YParameterResult")
        status = np.asarray(payload.status, dtype=np.str_)
    elif metric_name == "z":
        if not isinstance(payload, ZParameterResult):
            raise TypeError("z payload must be ZParameterResult")
        status = np.asarray(payload.status, dtype=np.str_)
    elif metric_name == "s":
        if not isinstance(payload, SParameterResult):
            raise TypeError("s payload must be SParameterResult")
        status = np.asarray(payload.status, dtype=np.str_)
    elif metric_name in {"zin", "zout"}:
        if not isinstance(payload, SweepRFScalarResult):
            raise TypeError("scalar RF payload must be SweepRFScalarResult")
        status = np.asarray(payload.status, dtype=np.str_)
    else:
        raise ValueError(f"unsupported rf metric '{metric_name}'")
    if status.shape != (n_points,):
        raise ValueError("payload status shape must match [n_points]")
    return status


def _metric_extractors(
    *,
    metric_name: str,
    payload: SweepRFMetricPayload,
    port_order: tuple[str, ...],
    n_points: int,
) -> tuple[tuple[str, Callable[[int], complex]], ...]:
    if metric_name == "y":
        if not isinstance(payload, YParameterResult):
            raise TypeError("y payload must be YParameterResult")
        values = np.asarray(payload.y, dtype=np.complex128)
        port_ids = tuple(payload.port_ids)
        return _matrix_extractors(
            metric_name=metric_name,
            values=values,
            port_ids=port_ids,
            port_order=port_order,
            n_points=n_points,
        )
    if metric_name == "z":
        if not isinstance(payload, ZParameterResult):
            raise TypeError("z payload must be ZParameterResult")
        values = np.asarray(payload.z, dtype=np.complex128)
        port_ids = tuple(payload.port_ids)
        return _matrix_extractors(
            metric_name=metric_name,
            values=values,
            port_ids=port_ids,
            port_order=port_order,
            n_points=n_points,
        )
    if metric_name == "s":
        if not isinstance(payload, SParameterResult):
            raise TypeError("s payload must be SParameterResult")
        values = np.asarray(payload.s, dtype=np.complex128)
        port_ids = tuple(payload.port_ids)
        return _matrix_extractors(
            metric_name=metric_name,
            values=values,
            port_ids=port_ids,
            port_order=port_order,
            n_points=n_points,
        )
    if metric_name in {"zin", "zout"}:
        if not isinstance(payload, SweepRFScalarResult):
            raise TypeError("scalar RF payload must be SweepRFScalarResult")
        values = np.asarray(payload.values, dtype=np.complex128)
        if values.shape != (n_points,):
            raise ValueError("scalar RF payload values shape must match [n_points]")

        def _extract(point_index: int) -> complex:
            return complex(values[point_index])

        return ((f"{metric_name}_{payload.port_id}", _extract),)
    raise ValueError(f"unsupported rf metric '{metric_name}'")


def _matrix_extractors(
    *,
    metric_name: str,
    values: NDArray[np.complex128],
    port_ids: tuple[str, ...],
    port_order: tuple[str, ...],
    n_points: int,
) -> tuple[tuple[str, Callable[[int], complex]], ...]:
    if values.ndim != _MATRIX_PAYLOAD_RANK:
        raise ValueError("matrix RF payload values must be rank-3")
    if values.shape[0] != n_points:
        raise ValueError("matrix RF payload point dimension must match n_points")

    index_by_port = {port_id: idx for idx, port_id in enumerate(port_ids)}
    for port_id in port_order:
        if port_id not in index_by_port:
            raise ValueError("port_order must be a subset of payload port_ids")

    extractors: list[tuple[str, Callable[[int], complex]]] = []
    for row_port in port_order:
        for col_port in port_order:
            row_index = index_by_port[row_port]
            col_index = index_by_port[col_port]
            column_name = f"{metric_name}_{row_port}_{col_port}"

            def _extract(point_index: int, ri: int = row_index, ci: int = col_index) -> complex:
                return complex(values[point_index, ri, ci])

            extractors.append((column_name, _extract))
    return tuple(extractors)


def _format_float(value: float) -> str:
    numeric = float(value)
    if numeric == 0.0:
        numeric = 0.0
    return format(numeric, _CSV_FLOAT_FORMAT)

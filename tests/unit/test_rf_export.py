from __future__ import annotations

import io
import json

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.rf_metrics import PortBoundary
from rfmna.sweep_engine import SweepLayout, SweepRFRequest, run_sweep
from rfmna.viz_io.rf_export import (
    RF_EXPORT_SCHEMA_VERSION,
    build_rf_export_metadata,
    export_rf_csv_bytes,
    export_rf_csv_text,
    export_rf_npz_bytes,
)

pytestmark = pytest.mark.unit

_CONVENTION_TAG = "port_wave_v4_s_from_z"
_EXPECTED_METADATA_KEYS = ["schema_version", "port_order", "z0", "convention_tag", "grid_hash"]


def _assemble_pass(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
    y_block = np.asarray([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.25 + 0.0j]], dtype=np.complex128)
    return (
        csc_matrix(y_block),
        np.asarray([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
    )


def _assemble_with_fail(point_index: int, _: float) -> tuple[csc_matrix, np.ndarray]:
    if point_index == 1:
        raise RuntimeError("intentional RF export point failure")
    return _assemble_pass(point_index, 0.0)


def _run_with_rf_request(
    frequencies: np.ndarray,
    request: SweepRFRequest,
    *,
    with_fail: bool = False,
):
    layout = SweepLayout(n_nodes=2, n_aux=0)
    assemble = _assemble_with_fail if with_fail else _assemble_pass
    return run_sweep(frequencies, layout, assemble, rf_request=request)


def _npz_contents(npz_bytes: bytes) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz_file:
        files = tuple(npz_file.files)
        arrays = {key: np.asarray(npz_file[key]) for key in files}
    return (files, arrays)


def test_metadata_contains_required_keys_and_stable_order() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        ),
        metrics=("zout", "y", "zin"),
        z0_ohm=50.0,
    )
    metadata = build_rf_export_metadata(
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )

    assert list(metadata.keys()) == _EXPECTED_METADATA_KEYS
    assert metadata["schema_version"] == RF_EXPORT_SCHEMA_VERSION
    assert metadata["port_order"] == ["P1", "P2"]
    assert metadata["z0"] == [50.0, 50.0]
    assert metadata["convention_tag"] == _CONVENTION_TAG
    assert isinstance(metadata["grid_hash"], str) and metadata["grid_hash"]


def test_npz_export_keys_are_deterministic_and_include_requested_payloads() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        ),
        metrics=("zout", "y", "zin"),
    )
    result = _run_with_rf_request(frequencies, request)

    first_bytes = export_rf_npz_bytes(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )
    second_bytes = export_rf_npz_bytes(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )
    first_keys, first_arrays = _npz_contents(first_bytes)
    second_keys, second_arrays = _npz_contents(second_bytes)

    expected_keys = (
        "schema_version",
        "rf_metadata_json",
        "frequencies_hz",
        "rf_y_values",
        "rf_y_status",
        "rf_zin_values",
        "rf_zin_status",
        "rf_zout_values",
        "rf_zout_status",
    )
    assert first_keys == expected_keys
    assert second_keys == expected_keys
    for key in expected_keys:
        if first_arrays[key].dtype.kind in {"U", "S", "O"} and first_arrays[key].ndim == 0:
            assert first_arrays[key].item() == second_arrays[key].item()
        elif first_arrays[key].dtype.kind in {"U", "S", "O"}:
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        else:
            np.testing.assert_allclose(first_arrays[key], second_arrays[key], equal_nan=True)

    metadata_json = str(first_arrays["rf_metadata_json"].item())
    parsed_pairs = json.loads(metadata_json, object_pairs_hook=list)
    assert [pair[0] for pair in parsed_pairs] == _EXPECTED_METADATA_KEYS


def test_csv_format_contract_header_order_precision_encoding_and_newline() -> None:
    frequencies = np.asarray([1.0, 2.0], dtype=np.float64)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("y", "zin"),
    )
    result = _run_with_rf_request(frequencies, request)

    csv_text = export_rf_csv_text(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )
    csv_bytes = export_rf_csv_bytes(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )

    decoded = csv_bytes.decode("utf-8")
    assert decoded == csv_text
    assert "\r\n" not in decoded
    assert decoded.endswith("\n")

    lines = decoded.split("\n")
    assert lines[0].startswith("# rf_export_metadata=")
    assert lines[1] == (
        "point_index,frequency_hz,status_y,status_zin,"
        "y_P1_P1_real,y_P1_P1_imag,y_P1_P2_real,y_P1_P2_imag,"
        "y_P2_P1_real,y_P2_P1_imag,y_P2_P2_real,y_P2_P2_imag,"
        "zin_P1_real,zin_P1_imag"
    )
    assert lines[2] == (
        "0,1.000000000000e+00,pass,pass,"
        "5.000000000000e-01,0.000000000000e+00,0.000000000000e+00,0.000000000000e+00,"
        "0.000000000000e+00,0.000000000000e+00,2.500000000000e-01,0.000000000000e+00,"
        "2.000000000000e+00,0.000000000000e+00"
    )


def test_export_is_deterministic_under_equivalent_request_permutations() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    request_one = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
        ),
        metrics=("zout", "s", "y", "zin", "z"),
        z0_ohm=50.0,
    )
    request_two = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("zin", "z", "y", "s", "zout"),
        z0_ohm=np.asarray([50.0, 50.0], dtype=np.float64),
    )
    first = _run_with_rf_request(frequencies, request_one, with_fail=True)
    second = _run_with_rf_request(frequencies, request_two, with_fail=True)

    first_csv = export_rf_csv_text(
        sweep_result=first,
        frequencies_hz=frequencies,
        rf_request=request_one,
        convention_tag=_CONVENTION_TAG,
    )
    second_csv = export_rf_csv_text(
        sweep_result=second,
        frequencies_hz=frequencies,
        rf_request=request_two,
        convention_tag=_CONVENTION_TAG,
    )
    assert first_csv == second_csv

    first_npz = export_rf_npz_bytes(
        sweep_result=first,
        frequencies_hz=frequencies,
        rf_request=request_one,
        convention_tag=_CONVENTION_TAG,
    )
    second_npz = export_rf_npz_bytes(
        sweep_result=second,
        frequencies_hz=frequencies,
        rf_request=request_two,
        convention_tag=_CONVENTION_TAG,
    )
    first_keys, first_arrays = _npz_contents(first_npz)
    second_keys, second_arrays = _npz_contents(second_npz)
    assert first_keys == second_keys
    for key in first_keys:
        if first_arrays[key].dtype.kind in {"U", "S", "O"} and first_arrays[key].ndim == 0:
            assert first_arrays[key].item() == second_arrays[key].item()
        elif first_arrays[key].dtype.kind in {"U", "S", "O"}:
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        else:
            np.testing.assert_allclose(first_arrays[key], second_arrays[key], equal_nan=True)


def test_failed_points_remain_in_export_with_nan_sentinels_and_status() -> None:
    frequencies = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    request = SweepRFRequest(
        ports=(
            PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
            PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
        ),
        metrics=("y", "s", "zin"),
    )
    result = _run_with_rf_request(frequencies, request, with_fail=True)
    fail_index = 1

    npz_bytes = export_rf_npz_bytes(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )
    keys, arrays = _npz_contents(npz_bytes)
    assert keys[:3] == ("schema_version", "rf_metadata_json", "frequencies_hz")
    assert np.isnan(arrays["rf_y_values"][fail_index].real).all()
    assert np.isnan(arrays["rf_y_values"][fail_index].imag).all()
    assert np.isnan(arrays["rf_s_values"][fail_index].real).all()
    assert np.isnan(arrays["rf_s_values"][fail_index].imag).all()
    assert np.isnan(arrays["rf_zin_values"][fail_index].real)
    assert np.isnan(arrays["rf_zin_values"][fail_index].imag)
    assert arrays["rf_y_status"][fail_index] == "fail"
    assert arrays["rf_s_status"][fail_index] == "fail"
    assert arrays["rf_zin_status"][fail_index] == "fail"

    csv_text = export_rf_csv_text(
        sweep_result=result,
        frequencies_hz=frequencies,
        rf_request=request,
        convention_tag=_CONVENTION_TAG,
    )
    fail_row = csv_text.split("\n")[3]
    assert ",fail,fail,fail," in fail_row
    assert ",nan,nan," in fail_row

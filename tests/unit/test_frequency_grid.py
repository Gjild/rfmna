from __future__ import annotations

import hashlib

import numpy as np
import pytest
from numpy.typing import NDArray

from rfmna.parser import ParseError, ParseErrorCode
from rfmna.sweep_engine import frequency_grid, frequency_grid_canonical_bytes, hash_frequency_grid

pytestmark = pytest.mark.unit

_POINTS = 5
_REPEAT_COUNT = 20
_ONE_POINT = 1


def _assert_array_equal(left: NDArray[np.float64], right: NDArray[np.float64]) -> None:
    assert left.dtype == np.float64
    assert right.dtype == np.float64
    np.testing.assert_array_equal(left, right)


def test_linear_grid_matches_formula_and_endpoints() -> None:
    out = frequency_grid("linear", 1.0, 9.0, _POINTS)
    expected = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64)

    _assert_array_equal(out, expected)
    assert out[0] == expected[0]
    assert out[-1] == expected[-1]


def test_linear_n_equals_one_valid_case() -> None:
    out = frequency_grid("linear", 7.0, 7.0, _ONE_POINT)
    _assert_array_equal(out, np.array([7.0], dtype=np.float64))


def test_linear_n_equals_one_mismatch_rejected() -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid("linear", 1.0, 2.0, _ONE_POINT)
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID.value


def test_log_grid_matches_formula_and_endpoints() -> None:
    out = frequency_grid("log", 1.0, 1000.0, _POINTS)
    expected_log = np.array([0.0, 0.75, 1.5, 2.25, 3.0], dtype=np.float64)
    np.testing.assert_allclose(np.log10(out), expected_log)
    assert out[0] == 10.0 ** expected_log[0]
    assert out[-1] == 10.0 ** expected_log[-1]


def test_log_n_equals_one_valid_case() -> None:
    out = frequency_grid("log", 10.0, 10.0, _ONE_POINT)
    _assert_array_equal(out, np.array([10.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("f_min_hz", "f_max_hz"),
    [
        (0.0, 0.0),
        (-1.0, -1.0),
    ],
)
def test_log_n_equals_one_nonpositive_domain_rejected(f_min_hz: float, f_max_hz: float) -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid("log", f_min_hz, f_max_hz, _ONE_POINT)
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN.value


def test_log_domain_rejected() -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid("log", 0.0, 10.0, _POINTS)
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN.value


@pytest.mark.parametrize(
    ("mode", "f_min_hz", "f_max_hz", "n_points"),
    [
        ("linear", 1.0, 10.0, 0),
        ("linear", 10.0, 1.0, _POINTS),
        ("linear", float("nan"), 10.0, _POINTS),
        ("linear", 1.0, float("inf"), _POINTS),
        ("LINEAR", 1.0, 10.0, _POINTS),
    ],
)
def test_frequency_grid_validation_failures(
    mode: str, f_min_hz: float, f_max_hz: float, n_points: int
) -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid(mode, f_min_hz, f_max_hz, n_points)  # type: ignore[arg-type]
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID.value


def test_nonfinite_generated_output_rejected() -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid("linear", -1.0e308, 1.0e308, 3)
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID.value


def test_determinism_repeated_calls_values_bytes_hash() -> None:
    baseline_grid = frequency_grid("log", 1.0, 1.0e6, _POINTS)
    baseline_bytes = frequency_grid_canonical_bytes(baseline_grid)
    baseline_hash = hash_frequency_grid(baseline_grid)

    for _ in range(_REPEAT_COUNT):
        grid = frequency_grid("log", 1.0, 1.0e6, _POINTS)
        assert np.array_equal(grid, baseline_grid)
        assert frequency_grid_canonical_bytes(grid) == baseline_bytes
        assert hash_frequency_grid(grid) == baseline_hash


def test_canonical_bytes_match_little_endian_float64() -> None:
    values = np.array([1.0, 2.0, 3.5], dtype=np.float64)
    expected = values.astype(np.dtype("<f8"), copy=False).tobytes(order="C")
    assert frequency_grid_canonical_bytes(values) == expected


def test_canonical_bytes_identical_across_input_containers() -> None:
    list_values = [1.0, 2.0, 3.0]
    tuple_values = (1.0, 2.0, 3.0)
    array_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    left = frequency_grid_canonical_bytes(list_values)
    center = frequency_grid_canonical_bytes(tuple_values)
    right = frequency_grid_canonical_bytes(array_values)
    assert left == center
    assert center == right


def test_canonical_bytes_normalize_big_endian_input() -> None:
    big_endian_values = np.array([1.0, 2.0, 3.0], dtype=np.dtype(">f8"))
    expected = np.array([1.0, 2.0, 3.0], dtype=np.dtype("<f8")).tobytes(order="C")
    assert frequency_grid_canonical_bytes(big_endian_values) == expected


def test_canonical_bytes_validation_failures() -> None:
    with pytest.raises(ParseError) as shape_error:
        frequency_grid_canonical_bytes(np.array([[1.0, 2.0]], dtype=np.float64))
    assert shape_error.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID.value

    with pytest.raises(ParseError) as finite_error:
        frequency_grid_canonical_bytes(np.array([1.0, float("nan")], dtype=np.float64))
    assert finite_error.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID.value


def test_hash_frequency_grid_matches_sha256_of_canonical_bytes() -> None:
    freq = frequency_grid("linear", 1.0, 4.0, 4)
    expected = hashlib.sha256(frequency_grid_canonical_bytes(freq)).hexdigest()
    assert hash_frequency_grid(freq) == expected

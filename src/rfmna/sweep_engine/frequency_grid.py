from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfmna.parser.errors import ParseErrorCode, build_parse_error

_MODE_LINEAR = "linear"
_MODE_LOG = "log"
_MIN_POINTS = 1


def _error_input_text(mode: str, f_min_hz: float, f_max_hz: float, n_points: int) -> str:
    return f"mode={mode},f_min_hz={f_min_hz!r},f_max_hz={f_max_hz!r},n_points={n_points}"


def _validate_common(mode: str, f_min_hz: float, f_max_hz: float, n_points: int) -> None:
    input_text = _error_input_text(mode, f_min_hz, f_max_hz, n_points)
    if mode not in {_MODE_LINEAR, _MODE_LOG}:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "frequency grid mode must be 'linear' or 'log'",
            input_text,
        )
    if n_points < _MIN_POINTS:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "n_points must be >= 1",
            input_text,
        )
    if not np.isfinite(f_min_hz) or not np.isfinite(f_max_hz):
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "frequency bounds must be finite",
            input_text,
        )
    if f_min_hz > f_max_hz:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "f_min_hz must be <= f_max_hz",
            input_text,
        )
    if n_points == _MIN_POINTS and f_min_hz != f_max_hz:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "for n_points == 1, f_min_hz must equal f_max_hz",
            input_text,
        )


def _linear_grid(f_min_hz: float, f_max_hz: float, n_points: int) -> NDArray[np.float64]:
    index = np.arange(n_points, dtype=np.float64)
    step = (f_max_hz - f_min_hz) / float(n_points - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        out = f_min_hz + index * step
    out[0] = f_min_hz
    out[-1] = f_max_hz
    return out


def _validate_log_domain(f_min_hz: float, f_max_hz: float, input_text: str) -> None:
    if f_min_hz <= 0.0 or f_max_hz <= 0.0:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN,
            "log mode requires f_min_hz > 0 and f_max_hz > 0",
            input_text,
        )


def _log_grid(
    f_min_hz: float, f_max_hz: float, n_points: int, input_text: str
) -> NDArray[np.float64]:
    _validate_log_domain(f_min_hz, f_max_hz, input_text)

    index = np.arange(n_points, dtype=np.float64)
    log_min = np.log10(f_min_hz)
    log_max = np.log10(f_max_hz)
    step = (log_max - log_min) / float(n_points - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(np.power(10.0, log_min + index * step), dtype=np.float64)
    out[0] = f_min_hz
    out[-1] = f_max_hz
    return out


def frequency_grid(
    mode: Literal["linear", "log"],
    f_min_hz: float,
    f_max_hz: float,
    n_points: int,
) -> NDArray[np.float64]:
    _validate_common(mode, f_min_hz, f_max_hz, n_points)
    input_text = _error_input_text(mode, f_min_hz, f_max_hz, n_points)
    if mode == _MODE_LOG:
        _validate_log_domain(f_min_hz, f_max_hz, input_text)
    if n_points == _MIN_POINTS:
        return np.array([f_min_hz], dtype=np.float64)

    if mode == _MODE_LINEAR:
        out = _linear_grid(f_min_hz, f_max_hz, n_points)
    else:
        out = _log_grid(f_min_hz, f_max_hz, n_points, input_text)

    if not np.all(np.isfinite(out)):
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "generated frequency grid contains non-finite values",
            input_text,
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def frequency_grid_canonical_bytes(freq_hz: Sequence[float]) -> bytes:
    arr = np.asarray(freq_hz, dtype=np.float64)
    if arr.ndim != 1:
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "frequency vector must be 1D",
            f"shape={arr.shape!r}",
        )
    if not np.all(np.isfinite(arr)):
        raise build_parse_error(
            ParseErrorCode.E_MODEL_FREQ_GRID_INVALID,
            "frequency vector must contain finite values",
            f"shape={arr.shape!r}",
        )

    little_endian = arr.astype(np.dtype("<f8"), copy=False)
    contiguous = np.ascontiguousarray(little_endian, dtype=np.dtype("<f8"))
    return contiguous.tobytes(order="C")


def hash_frequency_grid(freq_hz: Sequence[float], algo: str = "sha256") -> str:
    hasher = hashlib.new(algo)
    hasher.update(frequency_grid_canonical_bytes(freq_hz))
    return hasher.hexdigest()

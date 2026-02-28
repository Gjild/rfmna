from __future__ import annotations

import numpy as np
import pytest

from rfmna.parser import ParseError, ParseErrorCode
from rfmna.sweep_engine import frequency_grid

pytestmark = pytest.mark.conformance

_LINEAR_START = 1.0
_LINEAR_STOP = 9.0
_LOG_START = 1.0
_LOG_STOP = 1000.0


def test_frequency_grid_linear_log_formulas_endpoints_and_n_equals_one() -> None:
    linear = frequency_grid("linear", _LINEAR_START, _LINEAR_STOP, 5)
    np.testing.assert_array_equal(linear, np.asarray([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64))
    assert linear[0] == _LINEAR_START
    assert linear[-1] == _LINEAR_STOP

    log = frequency_grid("log", _LOG_START, _LOG_STOP, 5)
    expected_log10 = np.asarray([0.0, 0.75, 1.5, 2.25, 3.0], dtype=np.float64)
    np.testing.assert_allclose(np.log10(log), expected_log10)
    assert log[0] == _LOG_START
    assert log[-1] == _LOG_STOP

    linear_single = frequency_grid("linear", 7.0, 7.0, 1)
    log_single = frequency_grid("log", 10.0, 10.0, 1)
    np.testing.assert_array_equal(linear_single, np.asarray([7.0], dtype=np.float64))
    np.testing.assert_array_equal(log_single, np.asarray([10.0], dtype=np.float64))


def test_frequency_grid_invalid_domain_and_deterministic_order() -> None:
    with pytest.raises(ParseError) as exc_info:
        frequency_grid("log", 0.0, 10.0, 3)
    assert exc_info.value.detail.code == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN.value
    with pytest.raises(ParseError) as single_point_exc:
        frequency_grid("log", 0.0, 0.0, 1)
    assert (
        single_point_exc.value.detail.code
        == ParseErrorCode.E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN.value
    )

    first = frequency_grid("log", 1.0, 1.0e6, 7)
    second = frequency_grid("log", 1.0, 1.0e6, 7)
    np.testing.assert_array_equal(first, second)
    assert np.array_equal(first, np.sort(first))

from __future__ import annotations

import pytest

from rfmna.parser import ParseError, ParseErrorCode, parse_frequency_unit, parse_scalar_number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", 1.0),
        ("1.0", 1.0),
        (".5", 0.5),
        ("1e6", 1.0e6),
        ("-2.5e-3", -2.5e-3),
        ("+3.0", 3.0),
        ("\t +3.0 \n", 3.0),
    ],
)
def test_parse_scalar_number_valid(text: str, expected: float) -> None:
    assert parse_scalar_number(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1,23",
        "1_000",
        "1 000",
        "sin(1)",
        "rand()",
        "1+2",
        "a*b",
        "",
        " ",
    ],
)
def test_parse_scalar_number_invalid_tokens(text: str) -> None:
    with pytest.raises(ParseError) as exc_info:
        parse_scalar_number(text)

    assert exc_info.value.detail.code == ParseErrorCode.E_PARSE_NUMBER_INVALID.value
    assert exc_info.value.detail.input_text == text


@pytest.mark.parametrize(
    "text",
    [
        "nan",
        "NaN",
        "inf",
        "-inf",
        "+Infinity",
        "1e309",
    ],
)
def test_parse_scalar_number_nonfinite_rejected(text: str) -> None:
    with pytest.raises(ParseError) as exc_info:
        parse_scalar_number(text)

    assert exc_info.value.detail.code == ParseErrorCode.E_PARSE_NUMBER_NONFINITE.value
    assert exc_info.value.detail.input_text == text


def test_parse_scalar_number_error_payload_is_deterministic() -> None:
    payloads: list[tuple[str, str, str]] = []
    for _ in range(10):
        with pytest.raises(ParseError) as exc_info:
            parse_scalar_number("1,23")
        detail = exc_info.value.detail
        payloads.append((detail.code, detail.message, detail.input_text))

    assert all(payload == payloads[0] for payload in payloads)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("Hz", 1.0),
        ("kHz", 1.0e3),
        ("MHz", 1.0e6),
        ("GHz", 1.0e9),
    ],
)
def test_parse_frequency_unit_valid(unit: str, expected: float) -> None:
    assert parse_frequency_unit(unit) == expected


@pytest.mark.parametrize(
    "unit",
    [
        "hz",
        "KHz",
        "mhz",
        "ghz",
        "THz",
        " Hz",
        "Hz ",
        "",
    ],
)
def test_parse_frequency_unit_invalid(unit: str) -> None:
    with pytest.raises(ParseError) as exc_info:
        parse_frequency_unit(unit)

    assert exc_info.value.detail.code == ParseErrorCode.E_PARSE_UNIT_INVALID.value
    assert exc_info.value.detail.input_text == unit


def test_parse_frequency_unit_error_payload_is_deterministic() -> None:
    payloads: list[tuple[str, str, str]] = []
    for _ in range(10):
        with pytest.raises(ParseError) as exc_info:
            parse_frequency_unit("hz")
        detail = exc_info.value.detail
        payloads.append((detail.code, detail.message, detail.input_text))

    assert all(payload == payloads[0] for payload in payloads)

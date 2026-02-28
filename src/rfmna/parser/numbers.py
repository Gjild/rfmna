from __future__ import annotations

import math
import re

from .errors import ParseErrorCode, build_parse_error

_ASCII_WS = " \t\n\r\f\v"
_NUMBER_PATTERN = re.compile(
    r"^[ \t\n\r\f\v]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[ \t\n\r\f\v]*$",
    flags=re.ASCII,
)
_NONFINITE_SPELLINGS = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def parse_scalar_number(input_text: str) -> float:
    stripped = input_text.strip(_ASCII_WS)
    if stripped.lower() in _NONFINITE_SPELLINGS:
        raise build_parse_error(
            ParseErrorCode.E_PARSE_NUMBER_NONFINITE,
            "numeric literal must be finite",
            input_text,
        )

    match = _NUMBER_PATTERN.fullmatch(input_text)
    if match is None:
        raise build_parse_error(
            ParseErrorCode.E_PARSE_NUMBER_INVALID,
            "invalid numeric literal",
            input_text,
        )

    value = float(match.group(1))
    if not math.isfinite(value):
        raise build_parse_error(
            ParseErrorCode.E_PARSE_NUMBER_NONFINITE,
            "numeric literal must be finite",
            input_text,
        )
    return value

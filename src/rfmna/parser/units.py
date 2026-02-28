from __future__ import annotations

from .errors import ParseErrorCode, build_parse_error

FREQUENCY_UNIT_SCALE: dict[str, float] = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}


def parse_frequency_unit(input_text: str) -> float:
    scale = FREQUENCY_UNIT_SCALE.get(input_text)
    if scale is None:
        raise build_parse_error(
            ParseErrorCode.E_PARSE_UNIT_INVALID,
            "invalid frequency unit suffix",
            input_text,
        )
    return scale

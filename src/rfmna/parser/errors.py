from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ParseErrorCode(StrEnum):
    E_PARSE_NUMBER_INVALID = "E_PARSE_NUMBER_INVALID"
    E_PARSE_NUMBER_NONFINITE = "E_PARSE_NUMBER_NONFINITE"
    E_PARSE_UNIT_INVALID = "E_PARSE_UNIT_INVALID"
    E_PARSE_EXPR_INVALID = "E_PARSE_EXPR_INVALID"
    E_PARSE_PARAM_UNDEFINED = "E_PARSE_PARAM_UNDEFINED"
    E_PARSE_PARAM_NONFINITE = "E_PARSE_PARAM_NONFINITE"
    E_PARSE_PARAM_VALUE_INVALID = "E_PARSE_PARAM_VALUE_INVALID"
    E_MODEL_PARAM_CYCLE = "E_MODEL_PARAM_CYCLE"
    E_MODEL_FREQ_GRID_INVALID = "E_MODEL_FREQ_GRID_INVALID"
    E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN = "E_MODEL_FREQ_GRID_INVALID_LOG_DOMAIN"


@dataclass(frozen=True, slots=True)
class ParseErrorDetail:
    code: str
    message: str
    input_text: str
    witness: tuple[str, ...] | None = None


class ParseError(ValueError):
    def __init__(self, detail: ParseErrorDetail) -> None:
        super().__init__(f"{detail.code}: {detail.message}")
        self.detail = detail


def build_parse_error(
    code: ParseErrorCode,
    message: str,
    input_text: str,
    witness: tuple[str, ...] | None = None,
) -> ParseError:
    return ParseError(
        ParseErrorDetail(
            code=code.value,
            message=message,
            input_text=input_text,
            witness=witness,
        )
    )

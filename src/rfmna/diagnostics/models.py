from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class SolverStage(StrEnum):
    PARSE = "parse"
    PREFLIGHT = "preflight"
    ASSEMBLE = "assemble"
    SOLVE = "solve"
    POSTPROCESS = "postprocess"


class NodeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)


class PortContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port_id: str = Field(min_length=1)


def _normalize_json(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, dict):
        raw_dict = cast(dict[object, object], value)
        string_keys: list[str] = []
        for key in raw_dict:
            if not isinstance(key, str):
                raise ValueError("witness object keys must be strings")
            string_keys.append(key)
        normalized: dict[str, object] = {}
        for key in sorted(string_keys):
            normalized[key] = _normalize_json(raw_dict[key])
        return normalized
    raise ValueError("witness must be JSON-serializable")


class DiagnosticEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: Severity
    message: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    solver_stage: SolverStage

    element_id: str | None = None
    node_context: NodeContext | None = None
    port_context: PortContext | None = None

    frequency_hz: float | None = None
    frequency_index: int | None = Field(default=None, ge=0)
    sweep_index: int | None = Field(default=None, ge=0)

    witness: object | None = None

    @field_validator("witness", mode="before")
    @classmethod
    def _validate_and_normalize_witness(cls, witness: object) -> object:
        if witness is None:
            return None
        return _normalize_json(witness)

    @model_validator(mode="after")
    def _validate_context_presence(self) -> DiagnosticEvent:
        if self.element_id is None and self.node_context is None and self.port_context is None:
            raise ValueError(
                "at least one context value is required: element_id, node_context, or port_context"
            )
        return self

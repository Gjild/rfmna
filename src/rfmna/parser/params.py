from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import ParseErrorCode, build_parse_error
from .expressions import evaluate_expression, extract_dependencies

ParamValue = str | float
_STATE_UNVISITED = 0
_STATE_ACTIVE = 1
_STATE_DONE = 2


@dataclass(frozen=True, slots=True)
class ResolvedParameters:
    items: tuple[tuple[str, float], ...]

    def __getitem__(self, key: str) -> float:
        for item_key, value in self.items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(item_key == key for item_key, _ in self.items)

    def get(self, key: str, default: float | None = None) -> float | None:
        for item_key, value in self.items:
            if item_key == key:
                return value
        return default

    def as_dict(self) -> dict[str, float]:
        return dict(self.items)

    def to_canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            separators=(",", ":"),
            ensure_ascii=True,
        )


def _merge_effective_params(
    file_params: Mapping[str, ParamValue],
    overrides: Mapping[str, ParamValue] | None,
) -> dict[str, ParamValue]:
    merged: dict[str, ParamValue] = {}
    for key in sorted(file_params):
        merged[key] = file_params[key]
    if overrides is not None:
        for key in sorted(overrides):
            merged[key] = overrides[key]
    return merged


def _canonical_cycle_witness(cycle_nodes: list[str]) -> tuple[str, ...]:
    return tuple(sorted(cycle_nodes))


def _parse_literal_value(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise build_parse_error(
            ParseErrorCode.E_PARSE_PARAM_NONFINITE,
            "parameter value must be finite",
            name,
        )
    return float(value)


def _build_dependency_map(
    effective_params: Mapping[str, ParamValue],
) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for name in sorted(effective_params):
        raw_value = effective_params[name]
        if isinstance(raw_value, str):
            extracted = extract_dependencies(raw_value)
            for dependency in extracted:
                if dependency not in effective_params:
                    raise build_parse_error(
                        ParseErrorCode.E_PARSE_PARAM_UNDEFINED,
                        f"undefined parameter reference: {dependency}",
                        name,
                        witness=(dependency,),
                    )
            dependencies[name] = extracted
        elif isinstance(raw_value, float):
            _parse_literal_value(name, raw_value)
            dependencies[name] = ()
        else:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_VALUE_INVALID,
                "parameter value must be a str expression or float literal",
                name,
            )
    return dependencies


def resolve_parameters(
    file_params: Mapping[str, ParamValue],
    overrides: Mapping[str, ParamValue] | None = None,
) -> ResolvedParameters:
    effective_params = _merge_effective_params(file_params, overrides)
    dependencies = _build_dependency_map(effective_params)
    order = tuple(sorted(effective_params))
    resolved: dict[str, float] = {}
    state: dict[str, int] = {}
    stack: list[str] = []

    def resolve_one(name: str) -> None:
        current_state = state.get(name, _STATE_UNVISITED)
        if current_state == _STATE_DONE:
            return
        if current_state == _STATE_ACTIVE:
            cycle_start = stack.index(name)
            witness = _canonical_cycle_witness(stack[cycle_start:])
            cycle_path = " -> ".join(witness + (witness[0],))
            raise build_parse_error(
                ParseErrorCode.E_MODEL_PARAM_CYCLE,
                f"parameter dependency cycle detected: {cycle_path}",
                name,
                witness=witness,
            )

        state[name] = _STATE_ACTIVE
        stack.append(name)
        for dependency in dependencies[name]:
            resolve_one(dependency)

        raw_value = effective_params[name]
        if isinstance(raw_value, float):
            resolved_value = _parse_literal_value(name, raw_value)
        else:
            resolved_value = evaluate_expression(raw_value, resolved)
            if not math.isfinite(resolved_value):
                raise build_parse_error(
                    ParseErrorCode.E_PARSE_PARAM_NONFINITE,
                    "parameter value must be finite",
                    name,
                )
        resolved[name] = float(resolved_value)
        stack.pop()
        state[name] = _STATE_DONE

    for name in order:
        resolve_one(name)

    return ResolvedParameters(tuple((name, resolved[name]) for name in order))

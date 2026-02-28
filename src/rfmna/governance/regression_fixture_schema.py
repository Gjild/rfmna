from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Protocol

_SCHEMA_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "$schema",
    "$id",
    "title",
    "type",
    "required",
    "properties",
)
_VALID_SCHEMA_TYPES: Final[set[str]] = {
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
}


class _ValidatorFn(Protocol):
    def __call__(
        self,
        *,
        instance: object,
        schema: Mapping[str, object],
        path: str,
        errors: list[str],
    ) -> None: ...


def load_json_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.as_posix()}")
    return payload


def validate_fixture_schema_document(schema: Mapping[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    for key in _SCHEMA_REQUIRED_KEYS:
        if key not in schema:
            errors.append(f"schema missing required key: {key}")
    schema_type = schema.get("type")
    if schema_type != "object":
        errors.append("schema root type must be object")
    required = schema.get("required")
    if not isinstance(required, list) or not required:
        errors.append("schema required must be a non-empty list")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        errors.append("schema properties must be a non-empty mapping")
    return tuple(errors)


def validate_json_against_schema(
    instance: object,
    schema: Mapping[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    _validate_value(instance=instance, schema=schema, path="$", errors=errors)
    return tuple(errors)


def _validate_value(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
        return

    enum_values = schema.get("enum")
    if enum_values is not None:
        if not isinstance(enum_values, list) or not enum_values:
            errors.append(f"{path}: schema enum must be a non-empty list")
            return
        if instance not in enum_values:
            errors.append(f"{path}: expected one of {tuple(enum_values)!r}")
            return

    schema_types = _coerce_schema_types(schema=schema, path=path, errors=errors)
    if schema_types is None:
        return
    matching_type = _pick_matching_type(instance=instance, allowed_types=schema_types)
    if matching_type is None:
        errors.append(f"{path}: expected type in {schema_types!r}")
        return

    _validate_by_type(
        schema_type=matching_type,
        instance=instance,
        schema=schema,
        path=path,
        errors=errors,
    )


def _coerce_schema_types(
    *,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> tuple[str, ...] | None:
    schema_type_raw = schema.get("type")
    parsed: tuple[str, ...] | None = None
    if schema_type_raw is None:
        parsed = None
    elif isinstance(schema_type_raw, str):
        if schema_type_raw not in _VALID_SCHEMA_TYPES:
            errors.append(f"{path}: unsupported schema type {schema_type_raw!r}")
        else:
            parsed = (schema_type_raw,)
    elif isinstance(schema_type_raw, list) and schema_type_raw:
        collected: list[str] = []
        for item in schema_type_raw:
            if not isinstance(item, str):
                errors.append(f"{path}: schema type list entries must be strings")
                return None
            if item not in _VALID_SCHEMA_TYPES:
                errors.append(f"{path}: unsupported schema type {item!r}")
                return None
            collected.append(item)
        parsed = tuple(dict.fromkeys(collected))
    else:
        errors.append(f"{path}: schema type must be a string or non-empty list of strings")
    return parsed


def _pick_matching_type(*, instance: object, allowed_types: tuple[str, ...]) -> str | None:
    for schema_type in allowed_types:
        if _matches_type(instance=instance, schema_type=schema_type):
            return schema_type
    return None


def _matches_type(*, instance: object, schema_type: str) -> bool:
    matchers: dict[str, Callable[[object], bool]] = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "number": _is_json_number,
        "integer": _is_json_integer,
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    matcher = matchers.get(schema_type)
    return bool(matcher(instance)) if matcher is not None else False


def _validate_by_type(
    *,
    schema_type: str,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    validator = _TYPE_VALIDATORS[schema_type]
    validator(instance=instance, schema=schema, path=path, errors=errors)


def _validate_object(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(instance, dict):
        errors.append(f"{path}: expected object")
        return

    required = schema.get("required", [])
    if not isinstance(required, list):
        errors.append(f"{path}: schema required must be a list")
        return
    for required_key in required:
        if not isinstance(required_key, str):
            errors.append(f"{path}: schema required entry must be a string")
            continue
        if required_key not in instance:
            errors.append(f"{path}: missing required key {required_key!r}")

    properties_raw = schema.get("properties", {})
    if not isinstance(properties_raw, dict):
        errors.append(f"{path}: schema properties must be a mapping")
        return
    properties = {
        key: value
        for key, value in properties_raw.items()
        if isinstance(key, str) and isinstance(value, dict)
    }

    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        errors.append(f"{path}: schema additionalProperties must be bool or object")
        return

    for key, value in instance.items():
        if not isinstance(key, str):
            errors.append(f"{path}: object keys must be strings")
            continue
        child_path = f"{path}.{key}"
        prop_schema = properties.get(key)
        if prop_schema is not None:
            _validate_value(instance=value, schema=prop_schema, path=child_path, errors=errors)
            continue
        if additional is False:
            errors.append(f"{path}: additional key not allowed: {key!r}")
            continue
        if isinstance(additional, dict):
            _validate_value(instance=value, schema=additional, path=child_path, errors=errors)


def _validate_array(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(instance, list):
        errors.append(f"{path}: expected array")
        return

    min_items = schema.get("minItems")
    if min_items is not None:
        if not isinstance(min_items, int) or isinstance(min_items, bool):
            errors.append(f"{path}: schema minItems must be an integer")
        elif len(instance) < min_items:
            errors.append(f"{path}: expected at least {min_items} items")

    max_items = schema.get("maxItems")
    if max_items is not None:
        if not isinstance(max_items, int) or isinstance(max_items, bool):
            errors.append(f"{path}: schema maxItems must be an integer")
        elif len(instance) > max_items:
            errors.append(f"{path}: expected at most {max_items} items")

    items_schema = schema.get("items")
    if items_schema is None:
        return
    if not isinstance(items_schema, dict):
        errors.append(f"{path}: schema items must be an object")
        return
    for index, value in enumerate(instance):
        _validate_value(
            instance=value,
            schema=items_schema,
            path=f"{path}[{index}]",
            errors=errors,
        )


def _validate_string(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    del schema
    if not isinstance(instance, str):
        errors.append(f"{path}: expected string")


def _validate_number(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    del schema
    if not _is_json_number(instance):
        errors.append(f"{path}: expected number")


def _validate_integer(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    del schema
    if not _is_json_integer(instance):
        errors.append(f"{path}: expected integer")


def _validate_boolean(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    del schema
    if not isinstance(instance, bool):
        errors.append(f"{path}: expected boolean")


def _validate_null(
    *,
    instance: object,
    schema: Mapping[str, object],
    path: str,
    errors: list[str],
) -> None:
    del schema
    if instance is not None:
        errors.append(f"{path}: expected null")


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_json_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_TYPE_VALIDATORS: Final[dict[str, _ValidatorFn]] = {
    "object": _validate_object,
    "array": _validate_array,
    "string": _validate_string,
    "number": _validate_number,
    "integer": _validate_integer,
    "boolean": _validate_boolean,
    "null": _validate_null,
}

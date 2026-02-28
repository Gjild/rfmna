from __future__ import annotations

import json
import os
import platform as py_platform
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import cast

import numpy as np
import scipy  # type: ignore[import-untyped]

from rfmna import __version__

CANONICAL_JSON_KWARGS = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
    "allow_nan": False,
}
_CANONICAL_SORT_KEYS = True
_CANONICAL_SEPARATORS = (",", ":")
_CANONICAL_ENSURE_ASCII = False
_CANONICAL_ALLOW_NAN = False
VOLATILE_STABLE_EXCLUDE_KEYS: frozenset[str] = frozenset(("timestamp", "timezone"))
CORE_DEPENDENCIES: tuple[str, ...] = (
    "matplotlib",
    "numpy",
    "pandas",
    "pydantic",
    "scipy",
    "typer",
)
THREAD_ENV_KEYS: tuple[str, ...] = (
    "PYTHONHASHSEED",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class ManifestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RunManifest:
    tool_version: str | None
    git_commit_sha: str | None
    source_hash_fallback: str | None
    python_version: str | None
    dependency_versions: Mapping[str, str | None]
    platform: Mapping[str, str | None]
    input_hash: str
    resolved_params_hash: str
    timestamp: str | None
    timezone: str | None
    solver_config_snapshot: Mapping[str, object] | None
    thread_runtime_fingerprint: Mapping[str, str | None]
    numeric_backend_fingerprint: Mapping[str, object | None]
    frequency_grid_metadata: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class RunArtifactWithManifest[T]:
    run_payload: T
    manifest: RunManifest


def build_manifest(  # noqa: PLR0913
    *,
    input_payload: object,
    resolved_params_payload: object,
    solver_config_snapshot: Mapping[str, object] | None = None,
    frequency_grid_metadata: Mapping[str, object] | None = None,
    tool_version: str | None = None,
    git_commit_sha: str | None = None,
    source_hash_fallback: str | None = None,
    dependency_versions: Mapping[str, str | None] | None = None,
    platform_fingerprint: Mapping[str, str | None] | None = None,
    thread_runtime_fingerprint: Mapping[str, str | None] | None = None,
    numeric_backend_fingerprint: Mapping[str, object | None] | None = None,
    timestamp: str | None = None,
    timezone: str | None = None,
) -> RunManifest:
    input_hash = _sha256_hex(_canonical_payload_bytes(input_payload, payload_name="input_payload"))
    resolved_params_hash = _sha256_hex(
        _canonical_payload_bytes(resolved_params_payload, payload_name="resolved_params_payload")
    )

    manifest = RunManifest(
        tool_version=tool_version if tool_version is not None else __version__,
        git_commit_sha=git_commit_sha
        if git_commit_sha is not None
        else _env_or_none("GIT_COMMIT_SHA"),
        source_hash_fallback=(
            source_hash_fallback
            if source_hash_fallback is not None
            else _env_or_none("RFMNA_SOURCE_HASH_FALLBACK")
        ),
        python_version=py_platform.python_version(),
        dependency_versions=_freeze_string_mapping(
            dependency_versions
            if dependency_versions is not None
            else _collect_dependency_versions()
        ),
        platform=_freeze_string_mapping(
            platform_fingerprint
            if platform_fingerprint is not None
            else _collect_platform_fingerprint()
        ),
        input_hash=input_hash,
        resolved_params_hash=resolved_params_hash,
        timestamp=timestamp if timestamp is not None else _default_timestamp(),
        timezone=timezone if timezone is not None else _default_timezone(),
        solver_config_snapshot=_freeze_object_mapping(solver_config_snapshot),
        thread_runtime_fingerprint=_freeze_string_mapping(
            thread_runtime_fingerprint
            if thread_runtime_fingerprint is not None
            else _collect_thread_runtime_fingerprint()
        ),
        numeric_backend_fingerprint=_freeze_object_mapping(
            numeric_backend_fingerprint
            if numeric_backend_fingerprint is not None
            else _collect_numeric_backend_fingerprint()
        )
        or MappingProxyType({}),
        frequency_grid_metadata=_freeze_object_mapping(frequency_grid_metadata),
    )
    return manifest


def to_canonical_dict(manifest: RunManifest) -> Mapping[str, object]:
    raw = {field.name: getattr(manifest, field.name) for field in fields(RunManifest)}
    normalized = _normalize_json(raw, payload_name="manifest")
    if not isinstance(normalized, dict):
        raise ManifestError(
            "E_MANIFEST_SERIALIZE_FAILED", "manifest normalization did not produce mapping"
        )
    return cast(dict[str, object], normalized)


def to_canonical_json(manifest: RunManifest) -> str:
    canonical = to_canonical_dict(manifest)
    return _canonical_json(canonical, payload_name="manifest")


def stable_projection(manifest: RunManifest) -> Mapping[str, object]:
    canonical = dict(to_canonical_dict(manifest))
    for key in VOLATILE_STABLE_EXCLUDE_KEYS:
        canonical.pop(key, None)
    normalized = _normalize_json(canonical, payload_name="stable_projection")
    if not isinstance(normalized, dict):
        raise ManifestError(
            "E_MANIFEST_SERIALIZE_FAILED",
            "stable projection normalization did not produce mapping",
        )
    return cast(dict[str, object], normalized)


def stable_manifest_hash(manifest: RunManifest) -> str:
    payload = _canonical_json(stable_projection(manifest), payload_name="stable_projection")
    return _sha256_hex(payload.encode("utf-8"))


def attach_manifest_to_run_payload[T](
    run_payload: T,
    manifest: RunManifest,
) -> RunArtifactWithManifest[T]:
    return RunArtifactWithManifest(run_payload=run_payload, manifest=manifest)


def _canonical_payload_bytes(payload: object, *, payload_name: str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray | memoryview):
        return bytes(payload)
    normalized = _normalize_json(payload, payload_name=payload_name)
    text = _canonical_json(normalized, payload_name=payload_name)
    return text.encode("utf-8")


def _canonical_json(payload: object, *, payload_name: str) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=_CANONICAL_SORT_KEYS,
            separators=_CANONICAL_SEPARATORS,
            ensure_ascii=_CANONICAL_ENSURE_ASCII,
            allow_nan=_CANONICAL_ALLOW_NAN,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            "E_MANIFEST_SERIALIZE_FAILED",
            f"{payload_name} is not serializable via canonical JSON path",
        ) from exc


def _normalize_json(payload: object, *, payload_name: str) -> object:
    if payload is None or isinstance(payload, bool | int | float | str):
        return payload
    if isinstance(payload, np.ndarray):
        return _normalize_json(payload.tolist(), payload_name=payload_name)
    if isinstance(payload, list | tuple):
        return [_normalize_json(item, payload_name=payload_name) for item in payload]
    if isinstance(payload, Mapping):
        normalized: dict[str, object] = {}
        keys: list[str] = []
        for key in payload:
            if not isinstance(key, str):
                raise ManifestError(
                    "E_MANIFEST_PAYLOAD_INVALID",
                    f"{payload_name} contains non-string mapping key",
                )
            keys.append(key)
        for key in sorted(keys):
            normalized[key] = _normalize_json(payload[key], payload_name=payload_name)
        return normalized
    raise ManifestError(
        "E_MANIFEST_PAYLOAD_INVALID",
        f"{payload_name} contains unsupported value type: {type(payload)!r}",
    )


def _sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _collect_dependency_versions() -> Mapping[str, str | None]:
    out: dict[str, str | None] = {}
    for name in CORE_DEPENDENCIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return MappingProxyType(out)


def _collect_platform_fingerprint() -> Mapping[str, str | None]:
    data = {
        "system": py_platform.system() or None,
        "release": py_platform.release() or None,
        "version": py_platform.version() or None,
        "machine": py_platform.machine() or None,
        "processor": py_platform.processor() or None,
        "python_implementation": py_platform.python_implementation() or None,
    }
    return MappingProxyType(data)


def _collect_thread_runtime_fingerprint() -> Mapping[str, str | None]:
    return MappingProxyType({key: os.environ.get(key) for key in THREAD_ENV_KEYS})


def _collect_numeric_backend_fingerprint() -> Mapping[str, object | None]:
    return MappingProxyType(
        {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "numpy_blas_opt_info": _collect_module_info(np, info_name="blas_opt_info"),
            "scipy_blas_opt_info": _collect_module_info(scipy, info_name="blas_opt_info"),
        }
    )


def _collect_module_info(module: object, *, info_name: str) -> Mapping[str, object] | None:
    config = getattr(module, "__config__", None)
    getter = getattr(config, "get_info", None)
    if not callable(getter):
        return None
    try:
        info = getter(info_name)
    except Exception:
        return None
    if isinstance(info, Mapping):
        normalized = _normalize_json(
            dict(info), payload_name=f"{module.__class__.__name__}.{info_name}"
        )
        if isinstance(normalized, dict):
            return MappingProxyType(cast(dict[str, object], normalized))
    return None


def _default_timestamp() -> str | None:
    return datetime.now().astimezone().isoformat()


def _default_timezone() -> str | None:
    now = datetime.now().astimezone()
    tz_name = now.tzname()
    return tz_name if tz_name else None


def _env_or_none(key: str) -> str | None:
    value = os.environ.get(key)
    return value if value else None


def _freeze_string_mapping(data: Mapping[str, str | None]) -> Mapping[str, str | None]:
    copied = {key: data[key] for key in sorted(data)}
    return MappingProxyType(copied)


def _freeze_object_mapping(data: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if data is None:
        return None
    normalized = _normalize_json(data, payload_name="mapping")
    if not isinstance(normalized, dict):
        raise ManifestError(
            "E_MANIFEST_PAYLOAD_INVALID", "mapping payload must normalize to mapping"
        )
    return MappingProxyType(cast(dict[str, object], normalized))

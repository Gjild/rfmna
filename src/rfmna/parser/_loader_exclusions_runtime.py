from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Final

_GOVERNED_ARTIFACT_RELATIVE_PATH: Final[Path] = Path("docs/dev/p3_loader_temporary_exclusions.yaml")
_PACKAGED_ARTIFACT_RESOURCE: Final[str] = "resources/p3_loader_temporary_exclusions.yaml"
_CLOSURE_PAYLOAD_TEXT: Final[str] = "schema_version: 1\nexclusions: []\n"

LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE: Final[str] = _GOVERNED_ARTIFACT_RELATIVE_PATH.as_posix()


def repo_loader_temp_exclusions_artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / _GOVERNED_ARTIFACT_RELATIVE_PATH


def source_tree_loader_temp_exclusions_resource_path() -> Path:
    return Path(__file__).resolve().parent / _PACKAGED_ARTIFACT_RESOURCE


def load_packaged_loader_temp_exclusions_payload_text() -> str:
    source_tree_resource_path = source_tree_loader_temp_exclusions_resource_path()
    if source_tree_resource_path.is_file():
        return source_tree_resource_path.read_text(encoding="utf-8")

    packaged_resource = files("rfmna.parser").joinpath(_PACKAGED_ARTIFACT_RESOURCE)
    try:
        with as_file(packaged_resource) as packaged_path:
            if packaged_path.is_file():
                return packaged_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "loader temporary exclusions packaged resource is unavailable"
    )


def load_loader_temp_exclusions_payload_text() -> str:
    repo_artifact_path = repo_loader_temp_exclusions_artifact_path()
    if repo_artifact_path.is_file():
        return repo_artifact_path.read_text(encoding="utf-8")
    return load_packaged_loader_temp_exclusions_payload_text()

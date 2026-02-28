from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from rfmna.governance.regression_fixture_schema import (
    load_json_mapping,
    validate_fixture_schema_document,
    validate_json_against_schema,
)

_APPROVED_HASHES_FILE: Final[str] = "approved_hashes_v1.json"
_FIXTURE_SCHEMA_FILE: Final[str] = "tests/regression/schemas/rf_regression_fixture_v1.schema.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_dir() -> Path:
    return _repo_root() / "tests/fixtures/regression"


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _load_fixture(path: Path) -> dict[str, object]:
    return load_json_mapping(path)


def _load_fixture_schema() -> dict[str, object]:
    schema = load_json_mapping(_repo_root() / _FIXTURE_SCHEMA_FILE)
    schema_errors = validate_fixture_schema_document(schema)
    if schema_errors:
        raise ValueError(
            f"fixture schema validation failed ({_FIXTURE_SCHEMA_FILE}): "
            + "; ".join(schema_errors)
        )
    return schema


def _iter_fixture_paths() -> tuple[Path, ...]:
    fixture_dir = _fixture_dir()
    paths = [
        path for path in sorted(fixture_dir.glob("*.json")) if path.name != _APPROVED_HASHES_FILE
    ]
    if not paths:
        raise ValueError("no regression fixtures found under tests/fixtures/regression")
    return tuple(paths)


def _build_hash_payload(paths: tuple[Path, ...]) -> dict[str, object]:
    root = _repo_root()
    schema = _load_fixture_schema()
    hashes: list[dict[str, str]] = []
    for path in paths:
        payload = _load_fixture(path)
        fixture_errors = validate_json_against_schema(payload, schema)
        if fixture_errors:
            raise ValueError(
                f"fixture schema validation failed ({path.relative_to(root).as_posix()}): "
                + "; ".join(fixture_errors)
            )
        hashes.append(
            {
                "fixture": path.relative_to(root).as_posix(),
                "canonical_sha256": _canonical_sha256(payload),
            }
        )
    return {
        "schema_version": 1,
        "policy": "canonical_json_sha256",
        "hashes": hashes,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approve regression fixture hashes")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="required explicit acknowledgment to rewrite approved hash lock",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.approve:
        raise SystemExit(
            "refusing to update approved regression hashes without --approve "
            "(explicit workflow gate)"
        )

    fixture_paths = _iter_fixture_paths()
    payload = _build_hash_payload(fixture_paths)
    target = _fixture_dir() / _APPROVED_HASHES_FILE
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

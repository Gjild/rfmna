from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml  # type: ignore[import-untyped]

_TYPED_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(?:E_[A-Z0-9_]+|FACTORY_[A-Z0-9_]+)\b")
_RUNTIME_LITERAL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b((?:E|W)_[A-Z0-9_]+)\b")
_TYPED_LITERAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b((?:E_[A-Z0-9_]+|FACTORY_[A-Z0-9_]+))\b"
)
_RUNTIME_SCOPE_REQUIRED_PREFIXES: Final[tuple[str, ...]] = (
    "src/rfmna/parser/",
    "src/rfmna/assembler/",
    "src/rfmna/solver/",
    "src/rfmna/sweep_engine/",
    "src/rfmna/rf_metrics/",
    "src/rfmna/cli/",
    "src/rfmna/viz_io/",
)
_TYPED_SCOPE_REQUIRED_PREFIXES: Final[tuple[str, ...]] = (
    "src/rfmna/parser/",
    "src/rfmna/assembler/",
    "src/rfmna/solver/",
    "src/rfmna/sweep_engine/",
    "src/rfmna/viz_io/",
    "src/rfmna/elements/",
)
_REQUIRED_TRACK_B_FAMILY_MODES: Final[dict[str, str]] = {
    "E_PARSE_*": "typed_error_only",
    "E_ASSEMBLER_*": "typed_error_only",
    "E_INDEX_*": "typed_error_only",
    "E_MANIFEST_*": "typed_error_only",
    "E_SOLVER_CONFIG_*": "typed_error_only",
}
_THRESHOLDS_REL_PATH: Final[str] = "docs/spec/thresholds_v4_0_0.yaml"
_TYPED_ERROR_MATRIX_REL_PATH: Final[str] = "docs/dev/typed_error_mapping_matrix.yaml"

type MappingMode = Literal["typed_error_only", "diagnostic_equivalent_required"]


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticInventory:
    schema_version: int
    runtime_emission_paths: tuple[str, ...]
    non_diagnostic_scoped_paths: tuple[str, ...]
    runtime_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypedErrorMappingFamily:
    family: str
    mapping_mode: MappingMode
    source_paths: tuple[str, ...]
    mapped_runtime_diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypedErrorMappingMatrix:
    schema_version: int
    families: tuple[TypedErrorMappingFamily, ...]


@dataclass(frozen=True, slots=True)
class TypedErrorRegistryEntry:
    code: str
    family: str
    source_paths: tuple[str, ...]
    diagnostic_equivalent_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypedErrorRegistry:
    schema_version: int
    entries: tuple[TypedErrorRegistryEntry, ...]


def load_runtime_diagnostic_inventory(path: Path) -> RuntimeDiagnosticInventory:
    payload = _load_structured_dict(path)
    _require_keys(
        payload,
        required_keys=(
            "schema_version",
            "runtime_emission_paths",
            "non_diagnostic_scoped_paths",
            "runtime_codes",
        ),
        field_name=path.as_posix(),
    )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("runtime diagnostic inventory schema_version must be 1")
    runtime_emission_paths = _sorted_unique_strings(
        payload.get("runtime_emission_paths"),
        field_name="runtime_emission_paths",
        allow_empty=False,
    )
    non_diagnostic_scoped_paths = _sorted_unique_strings(
        payload.get("non_diagnostic_scoped_paths"),
        field_name="non_diagnostic_scoped_paths",
        allow_empty=False,
    )
    runtime_codes = _sorted_unique_strings(
        payload.get("runtime_codes"),
        field_name="runtime_codes",
        allow_empty=False,
    )
    return RuntimeDiagnosticInventory(
        schema_version=1,
        runtime_emission_paths=runtime_emission_paths,
        non_diagnostic_scoped_paths=non_diagnostic_scoped_paths,
        runtime_codes=runtime_codes,
    )


def load_typed_error_mapping_matrix(path: Path) -> TypedErrorMappingMatrix:
    payload = _load_structured_dict(path)
    _require_keys(
        payload,
        required_keys=("schema_version", "families"),
        field_name=path.as_posix(),
    )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("typed error mapping matrix schema_version must be 1")
    families_raw = payload.get("families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("typed error mapping matrix families must be a non-empty list")

    families: list[TypedErrorMappingFamily] = []
    seen_families: set[str] = set()
    for index, family_entry in enumerate(families_raw):
        if not isinstance(family_entry, dict):
            raise ValueError(f"typed error mapping matrix families[{index}] must be a mapping")
        _require_keys(
            family_entry,
            required_keys=(
                "family",
                "mapping_mode",
                "source_paths",
                "mapped_runtime_diagnostic_codes",
            ),
            field_name=f"typed error mapping matrix families[{index}]",
        )
        family = family_entry.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(
                f"typed error mapping matrix families[{index}].family must be non-empty"
            )
        _family_prefix(family)
        if family in seen_families:
            raise ValueError(f"typed error mapping matrix contains duplicate family: {family}")
        seen_families.add(family)

        mapping_mode = family_entry.get("mapping_mode")
        if mapping_mode not in {"typed_error_only", "diagnostic_equivalent_required"}:
            raise ValueError(
                f"typed error mapping matrix families[{index}].mapping_mode must be "
                "typed_error_only or diagnostic_equivalent_required"
            )
        source_paths = _sorted_unique_strings(
            family_entry.get("source_paths"),
            field_name=f"typed error mapping matrix families[{index}].source_paths",
            allow_empty=False,
        )
        mapped_runtime_codes = _sorted_unique_strings(
            family_entry.get("mapped_runtime_diagnostic_codes"),
            field_name=f"typed error mapping matrix families[{index}].mapped_runtime_diagnostic_codes",
            allow_empty=True,
        )
        families.append(
            TypedErrorMappingFamily(
                family=family,
                mapping_mode=mapping_mode,
                source_paths=source_paths,
                mapped_runtime_diagnostic_codes=mapped_runtime_codes,
            )
        )

    if tuple(sorted(seen_families)) != tuple(row.family for row in families):
        raise ValueError("typed error mapping matrix families must be sorted by family")

    return TypedErrorMappingMatrix(schema_version=1, families=tuple(families))


def load_typed_error_registry(path: Path) -> TypedErrorRegistry:
    payload = _load_structured_dict(path)
    _require_keys(
        payload,
        required_keys=("schema_version", "entries"),
        field_name=path.as_posix(),
    )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("typed error code registry schema_version must be 1")

    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError("typed error code registry entries must be a non-empty list")

    entries: list[TypedErrorRegistryEntry] = []
    seen_codes: set[str] = set()
    for index, entry in enumerate(entries_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"typed error code registry entries[{index}] must be a mapping")
        _require_keys(
            entry,
            required_keys=("code", "family", "source_paths", "diagnostic_equivalent_codes"),
            field_name=f"typed error code registry entries[{index}]",
        )
        code = entry.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError(f"typed error code registry entries[{index}].code must be non-empty")
        if not _TYPED_CODE_PATTERN.fullmatch(code):
            raise ValueError(
                f"typed error code registry entries[{index}].code has invalid format: {code}"
            )
        if code in seen_codes:
            raise ValueError(f"typed error code registry contains duplicate code: {code}")
        seen_codes.add(code)

        family = entry.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"typed error code registry entries[{index}].family must be non-empty")
        _family_prefix(family)
        source_paths = _sorted_unique_strings(
            entry.get("source_paths"),
            field_name=f"typed error code registry entries[{index}].source_paths",
            allow_empty=False,
        )
        diagnostic_equivalent_codes = _sorted_unique_strings(
            entry.get("diagnostic_equivalent_codes"),
            field_name=f"typed error code registry entries[{index}].diagnostic_equivalent_codes",
            allow_empty=True,
        )
        entries.append(
            TypedErrorRegistryEntry(
                code=code,
                family=family,
                source_paths=source_paths,
                diagnostic_equivalent_codes=diagnostic_equivalent_codes,
            )
        )

    if tuple(sorted(seen_codes)) != tuple(entry.code for entry in entries):
        raise ValueError("typed error code registry entries must be sorted by code")

    return TypedErrorRegistry(schema_version=1, entries=tuple(entries))


def derive_runtime_emitted_codes(
    *,
    repo_root: Path,
    runtime_emission_paths: tuple[str, ...],
    typed_error_only_families: tuple[str, ...],
) -> tuple[str, ...]:
    codes: set[str] = set()
    for rel_path in runtime_emission_paths:
        text = _read_repo_file(repo_root=repo_root, rel_path=rel_path)
        if rel_path == _THRESHOLDS_REL_PATH:
            codes.update(_extract_solver_warning_codes_from_thresholds(text))
            continue
        for code in _extract_code_literals(
            text=text,
            pattern=_RUNTIME_LITERAL_PATTERN,
            source_path=rel_path,
        ):
            if _code_matches_any_family(code=code, families=typed_error_only_families):
                continue
            if "/solver/" in rel_path and not code.startswith(("E_NUM_", "W_NUM_")):
                continue
            codes.add(code)
    return tuple(sorted(codes))


def validate_runtime_diagnostic_inventory(
    *,
    repo_root: Path,
    inventory: RuntimeDiagnosticInventory,
    catalog_codes: set[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    scoped_paths = tuple(
        sorted(set((*inventory.runtime_emission_paths, *inventory.non_diagnostic_scoped_paths)))
    )
    for prefix in _RUNTIME_SCOPE_REQUIRED_PREFIXES:
        if not any(path.startswith(prefix) for path in scoped_paths):
            errors.append(f"runtime inventory scope does not cover module prefix: {prefix}")

    runtime_emission_paths = set(inventory.runtime_emission_paths)
    non_diagnostic_scoped_paths = set(inventory.non_diagnostic_scoped_paths)
    if _THRESHOLDS_REL_PATH not in runtime_emission_paths:
        errors.append(
            f"runtime diagnostic inventory missing required runtime source path: {_THRESHOLDS_REL_PATH}"
        )
    typed_error_only_families, matrix_load_errors = _load_typed_error_only_families(
        repo_root=repo_root
    )
    errors.extend(matrix_load_errors)

    declared_scope_paths = set(inventory.runtime_emission_paths) | set(
        inventory.non_diagnostic_scoped_paths
    )
    scope_paths_with_codes = set(
        _discover_scope_paths_with_runtime_code_literals(repo_root=repo_root)
    )
    missing_scope_paths = sorted(scope_paths_with_codes - declared_scope_paths)
    if missing_scope_paths:
        errors.append(
            "runtime inventory scope missing code-bearing path(s): "
            + ", ".join(missing_scope_paths)
        )

    (
        discovered_runtime_paths,
        discovered_runtime_codes,
        runtime_discovery_errors,
    ) = _discover_runtime_emission_candidates(
        repo_root=repo_root,
        typed_error_only_families=typed_error_only_families,
    )
    errors.extend(runtime_discovery_errors)
    declared_runtime_paths = runtime_emission_paths - {_THRESHOLDS_REL_PATH}
    missing_runtime_paths = sorted(set(discovered_runtime_paths) - declared_runtime_paths)
    if missing_runtime_paths:
        errors.append(
            "runtime diagnostic inventory missing runtime emission path(s): "
            + ", ".join(missing_runtime_paths)
        )
    misclassified_runtime_paths = sorted(
        set(discovered_runtime_paths) & non_diagnostic_scoped_paths
    )
    if misclassified_runtime_paths:
        errors.append(
            "runtime diagnostic inventory misclassifies runtime emission path(s) as non_diagnostic_scoped_paths: "
            + ", ".join(misclassified_runtime_paths)
        )
    extra_runtime_paths = sorted(declared_runtime_paths - set(discovered_runtime_paths))
    if extra_runtime_paths:
        errors.append(
            "runtime diagnostic inventory declares non-runtime emission path(s): "
            + ", ".join(extra_runtime_paths)
        )
    typed_only_runtime_emissions = _discover_typed_only_codes_on_runtime_paths(
        repo_root=repo_root,
        runtime_emission_paths=inventory.runtime_emission_paths,
        typed_error_only_families=typed_error_only_families,
    )
    if typed_only_runtime_emissions:
        errors.append(
            "runtime diagnostic inventory found typed_error_only code emissions on runtime paths "
            "(explicit promotion required): " + ", ".join(typed_only_runtime_emissions)
        )

    derived_codes = derive_runtime_emitted_codes(
        repo_root=repo_root,
        runtime_emission_paths=inventory.runtime_emission_paths,
        typed_error_only_families=typed_error_only_families,
    )
    derived_set = set(derived_codes)
    declared_set = set(inventory.runtime_codes)
    missing = sorted(set(discovered_runtime_codes) - declared_set)
    if missing:
        errors.append("runtime diagnostic inventory missing emitted codes: " + ", ".join(missing))
    extras = sorted(declared_set - derived_set)
    if extras:
        errors.append(
            "runtime diagnostic inventory declares non-emitted codes: " + ", ".join(extras)
        )

    uncataloged = sorted(declared_set - catalog_codes)
    if uncataloged:
        errors.append(
            "runtime diagnostic inventory includes uncataloged codes: " + ", ".join(uncataloged)
        )
    uncataloged_emitted = sorted(set(discovered_runtime_codes) - catalog_codes)
    if uncataloged_emitted:
        errors.append(
            "runtime diagnostic inventory discovered uncataloged emitted runtime codes: "
            + ", ".join(uncataloged_emitted)
        )
    return tuple(errors)


def validate_typed_error_registry(
    *,
    repo_root: Path,
    registry: TypedErrorRegistry,
    matrix: TypedErrorMappingMatrix,
    catalog_codes: set[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    matrix_by_family = {row.family: row for row in matrix.families}
    errors.extend(_validate_required_track_b_families(matrix_by_family))

    (
        discovered_repo_codes_by_family,
        discovered_repo_paths_by_family,
        uncovered_typed_codes,
    ) = _discover_repo_typed_codes_by_family(
        repo_root=repo_root,
        matrix=matrix,
        catalog_codes=catalog_codes,
    )
    if uncovered_typed_codes:
        errors.append(
            "typed error discovery found uncategorized non-diagnostic code(s): "
            + ", ".join(uncovered_typed_codes)
        )
    registry_by_family: dict[str, set[str]] = {}
    registry_source_paths_by_family: dict[str, set[str]] = {}
    for entry in registry.entries:
        registry_by_family.setdefault(entry.family, set()).add(entry.code)
        registry_source_paths_by_family.setdefault(entry.family, set()).update(entry.source_paths)
        errors.extend(
            _validate_typed_registry_entry(
                entry=entry,
                matrix_by_family=matrix_by_family,
                catalog_codes=catalog_codes,
            )
        )

    for row in matrix.families:
        discovered = set(discovered_repo_codes_by_family.get(row.family, ()))
        declared = registry_by_family.get(row.family, set())
        missing = sorted(discovered - declared)
        if missing:
            errors.append(
                f"typed error registry missing discovered code(s) for family {row.family}: "
                + ", ".join(missing)
            )
        extras = sorted(declared - discovered)
        if extras:
            errors.append(
                f"typed error registry declares non-discovered code(s) for family {row.family}: "
                + ", ".join(extras)
            )
        discovered_paths = set(discovered_repo_paths_by_family.get(row.family, ()))
        missing_discovered_paths = sorted(discovered_paths - set(row.source_paths))
        if missing_discovered_paths:
            errors.append(
                f"typed error mapping matrix missing discovered source path(s) for family {row.family}: "
                + ", ".join(missing_discovered_paths)
            )
        matrix_source_paths = set(row.source_paths)
        registry_source_paths = registry_source_paths_by_family.get(row.family, set())
        missing_matrix_source_paths = sorted(registry_source_paths - matrix_source_paths)
        if missing_matrix_source_paths:
            errors.append(
                f"typed error mapping matrix missing source path(s) for family {row.family}: "
                + ", ".join(missing_matrix_source_paths)
            )
        errors.extend(_validate_matrix_family_alignment(row=row, catalog_codes=catalog_codes))
    return tuple(errors)


def _discover_runtime_emission_candidates(
    *,
    repo_root: Path,
    typed_error_only_families: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    discovered_paths: set[str] = set()
    discovered_codes: set[str] = set()
    errors: list[str] = []

    for prefix in _RUNTIME_SCOPE_REQUIRED_PREFIXES:
        directory = repo_root / prefix
        if not directory.exists():
            continue
        for source_file in sorted(directory.rglob("*.py")):
            rel_path = source_file.relative_to(repo_root).as_posix()
            text = source_file.read_text(encoding="utf-8")
            try:
                literals = _extract_code_literals(
                    text=text,
                    pattern=_RUNTIME_LITERAL_PATTERN,
                    source_path=rel_path,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            runtime_like_codes = {
                code
                for code in literals
                if not _code_matches_any_family(code=code, families=typed_error_only_families)
            }
            if runtime_like_codes:
                discovered_paths.add(rel_path)
                discovered_codes.update(runtime_like_codes)

    try:
        thresholds_text = _read_repo_file(repo_root=repo_root, rel_path=_THRESHOLDS_REL_PATH)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        try:
            discovered_codes.update(_extract_solver_warning_codes_from_thresholds(thresholds_text))
        except ValueError as exc:
            errors.append(str(exc))
    return (
        tuple(sorted(discovered_paths)),
        tuple(sorted(discovered_codes)),
        tuple(errors),
    )


def _discover_scope_paths_with_runtime_code_literals(*, repo_root: Path) -> tuple[str, ...]:
    discovered: list[str] = []
    for prefix in _RUNTIME_SCOPE_REQUIRED_PREFIXES:
        directory = repo_root / prefix
        if not directory.exists():
            continue
        for source_file in sorted(directory.rglob("*.py")):
            rel_path = source_file.relative_to(repo_root).as_posix()
            text = source_file.read_text(encoding="utf-8")
            try:
                literals = _extract_code_literals(
                    text=text,
                    pattern=_RUNTIME_LITERAL_PATTERN,
                    source_path=rel_path,
                )
            except ValueError:
                continue
            if literals:
                discovered.append(rel_path)
    return tuple(discovered)


def _discover_typed_only_codes_on_runtime_paths(
    *,
    repo_root: Path,
    runtime_emission_paths: tuple[str, ...],
    typed_error_only_families: tuple[str, ...],
) -> tuple[str, ...]:
    violations: set[str] = set()
    for rel_path in runtime_emission_paths:
        if rel_path == _THRESHOLDS_REL_PATH:
            continue
        text = _read_repo_file(repo_root=repo_root, rel_path=rel_path)
        for code in _extract_code_keyword_literals(text=text, source_path=rel_path):
            if _code_matches_any_family(code=code, families=typed_error_only_families):
                violations.add(f"{code}@{rel_path}")
    return tuple(sorted(violations))


def _discover_repo_typed_codes_by_family(
    *,
    repo_root: Path,
    matrix: TypedErrorMappingMatrix,
    catalog_codes: set[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], tuple[str, ...]]:
    codes_by_family: dict[str, set[str]] = {row.family: set() for row in matrix.families}
    paths_by_family: dict[str, set[str]] = {row.family: set() for row in matrix.families}
    uncovered: set[str] = set()

    for prefix in _TYPED_SCOPE_REQUIRED_PREFIXES:
        directory = repo_root / prefix
        if not directory.exists():
            continue
        for source_file in sorted(directory.rglob("*.py")):
            rel_path = source_file.relative_to(repo_root).as_posix()
            text = source_file.read_text(encoding="utf-8")
            literals = _extract_code_literals(
                text=text,
                pattern=_TYPED_LITERAL_PATTERN,
                source_path=rel_path,
            )
            for code in literals:
                matched_family = False
                for row in matrix.families:
                    if _code_matches_family(code, row.family):
                        matched_family = True
                        codes_by_family[row.family].add(code)
                        paths_by_family[row.family].add(rel_path)
                if not matched_family and code not in catalog_codes:
                    uncovered.add(f"{code}@{rel_path}")

    codes_tuple = {family: tuple(sorted(codes)) for family, codes in codes_by_family.items()}
    paths_tuple = {family: tuple(sorted(paths)) for family, paths in paths_by_family.items()}
    return (codes_tuple, paths_tuple, tuple(sorted(uncovered)))


def _validate_required_track_b_families(
    matrix_by_family: dict[str, TypedErrorMappingFamily],
) -> tuple[str, ...]:
    errors: list[str] = []
    for family, required_mode in _REQUIRED_TRACK_B_FAMILY_MODES.items():
        row = matrix_by_family.get(family)
        if row is None:
            errors.append(f"typed error mapping matrix missing required family: {family}")
            continue
        if row.mapping_mode != required_mode:
            errors.append(
                "typed error mapping matrix family has invalid mapping_mode: "
                f"{family} expected={required_mode} actual={row.mapping_mode}"
            )
    return tuple(errors)


def _validate_typed_registry_entry(
    *,
    entry: TypedErrorRegistryEntry,
    matrix_by_family: dict[str, TypedErrorMappingFamily],
    catalog_codes: set[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    row = matrix_by_family.get(entry.family)
    if row is None:
        errors.append(f"typed error registry entry references unknown family: {entry.family}")
        return tuple(errors)
    if not _code_matches_family(entry.code, entry.family):
        errors.append(
            f"typed error registry entry code does not match family: {entry.code} !~ {entry.family}"
        )
    if row.mapping_mode == "typed_error_only":
        if entry.diagnostic_equivalent_codes:
            errors.append(
                f"typed_error_only family entry must not map diagnostic equivalents: {entry.code}"
            )
        return tuple(errors)

    if not entry.diagnostic_equivalent_codes:
        errors.append(
            f"diagnostic_equivalent_required family entry missing diagnostic mapping: {entry.code}"
        )
    allowed_diagnostic_codes = set(row.mapped_runtime_diagnostic_codes)
    invalid_mapped = sorted(
        code for code in entry.diagnostic_equivalent_codes if code not in allowed_diagnostic_codes
    )
    if invalid_mapped:
        errors.append(
            "typed error entry diagnostic mapping not allowed by matrix family "
            f"{entry.family}: {entry.code} -> {', '.join(invalid_mapped)}"
        )
    uncataloged_mapped = sorted(
        code for code in entry.diagnostic_equivalent_codes if code not in catalog_codes
    )
    if uncataloged_mapped:
        errors.append(
            f"typed error entry maps to uncataloged diagnostic code(s): {entry.code} -> "
            + ", ".join(uncataloged_mapped)
        )
    return tuple(errors)


def _validate_matrix_family_alignment(
    *,
    row: TypedErrorMappingFamily,
    catalog_codes: set[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    if row.mapping_mode == "diagnostic_equivalent_required":
        if not row.mapped_runtime_diagnostic_codes:
            errors.append(
                "diagnostic_equivalent_required family must define mapped_runtime_diagnostic_codes: "
                f"{row.family}"
            )
        uncataloged_row_codes = sorted(
            code for code in row.mapped_runtime_diagnostic_codes if code not in catalog_codes
        )
        if uncataloged_row_codes:
            errors.append(
                f"typed error matrix family maps uncataloged diagnostic codes: {row.family} -> "
                + ", ".join(uncataloged_row_codes)
            )
        return tuple(errors)
    if row.mapped_runtime_diagnostic_codes:
        errors.append(
            f"typed_error_only family must not define mapped_runtime_diagnostic_codes: {row.family}"
        )
    return tuple(errors)


def _load_structured_dict(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact is missing: {path.as_posix()}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML payload: {path.as_posix()}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a mapping: {path.as_posix()}")
    return payload


def _read_repo_file(*, repo_root: Path, rel_path: str) -> str:
    if Path(rel_path).is_absolute():
        raise ValueError(f"artifact path must be repo-relative: {rel_path}")
    target = (repo_root / rel_path).resolve(strict=False)
    root = repo_root.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"artifact path must stay within repository: {rel_path}")
    if not target.exists() or not target.is_file():
        raise ValueError(f"artifact path does not exist: {rel_path}")
    return target.read_text(encoding="utf-8")


def _extract_solver_warning_codes_from_thresholds(payload_text: str) -> tuple[str, ...]:
    try:
        payload = yaml.safe_load(payload_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML payload: {_THRESHOLDS_REL_PATH}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} must decode to mapping")
    numeric_contract = payload.get("numeric_contract")
    if not isinstance(numeric_contract, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} missing numeric_contract mapping")
    condition_indicator = numeric_contract.get("condition_indicator")
    if not isinstance(condition_indicator, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} missing condition_indicator mapping")
    estimator = condition_indicator.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} missing estimator mapping")
    unavailable_policy = estimator.get("unavailable_policy")
    if not isinstance(unavailable_policy, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} missing unavailable_policy mapping")
    bands = condition_indicator.get("bands")
    if not isinstance(bands, dict):
        raise ValueError(f"{_THRESHOLDS_REL_PATH} missing bands mapping")

    codes: list[str] = []
    for field_name, mapping in (
        (
            "numeric_contract.condition_indicator.estimator.unavailable_policy.warning_code",
            unavailable_policy,
        ),
        ("numeric_contract.condition_indicator.bands.warning_code", bands),
    ):
        value = mapping.get("warning_code")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{_THRESHOLDS_REL_PATH} missing valid {field_name}")
        if not value.startswith("W_NUM_"):
            raise ValueError(f"{_THRESHOLDS_REL_PATH} {field_name} must start with W_NUM_: {value}")
        codes.append(value)
    return tuple(sorted(set(codes)))


def _require_keys(
    mapping: dict[str, object],
    *,
    required_keys: tuple[str, ...],
    field_name: str,
) -> None:
    missing = sorted(set(required_keys) - set(mapping))
    if missing:
        raise ValueError(f"{field_name} missing required keys: {', '.join(missing)}")


def _sorted_unique_strings(
    value: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        strings.append(item)
    canonical = tuple(sorted(set(strings)))
    if tuple(strings) != canonical:
        raise ValueError(f"{field_name} entries must be sorted ascending and unique")
    if not allow_empty and not canonical:
        raise ValueError(f"{field_name} must be non-empty")
    return canonical


def _family_prefix(family: str) -> str:
    if not family.endswith("*"):
        raise ValueError(f"family must end with '*': {family}")
    if len(family) <= 1:
        raise ValueError(f"family must be non-empty before '*': {family}")
    return family[:-1]


def _code_matches_family(code: str, family: str) -> bool:
    return code.startswith(_family_prefix(family))


def _extract_code_literals(
    *,
    text: str,
    pattern: re.Pattern[str],
    source_path: str,
) -> tuple[str, ...]:
    ignored_docstring_nodes: set[ast.Constant] = set()
    try:
        tree = ast.parse(text, filename=source_path)
    except SyntaxError as exc:
        raise ValueError(
            "failed to parse python source for diagnostics taxonomy discovery: "
            f"{source_path}:{exc.lineno}:{exc.offset}: {exc.msg}"
        ) from exc

    def mark_docstring(body: list[ast.stmt]) -> None:
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ignored_docstring_nodes.add(first.value)

    mark_docstring(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark_docstring(node.body)

    codes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node in ignored_docstring_nodes:
            continue
        for match in pattern.finditer(node.value):
            codes.add(match.group(1))
    return tuple(sorted(codes))


def _extract_code_keyword_literals(*, text: str, source_path: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(text, filename=source_path)
    except SyntaxError as exc:
        raise ValueError(
            "failed to parse python source for diagnostics taxonomy discovery: "
            f"{source_path}:{exc.lineno}:{exc.offset}: {exc.msg}"
        ) from exc

    codes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "code":
                continue
            if not isinstance(keyword.value, ast.Constant) or not isinstance(
                keyword.value.value, str
            ):
                continue
            for match in _RUNTIME_LITERAL_PATTERN.finditer(keyword.value.value):
                codes.add(match.group(1))
    return tuple(sorted(codes))


def _load_typed_error_only_families(*, repo_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    matrix_path = repo_root / _TYPED_ERROR_MATRIX_REL_PATH
    try:
        matrix = load_typed_error_mapping_matrix(matrix_path)
    except ValueError as exc:
        return (
            (),
            (
                "runtime diagnostic inventory could not load Track B mapping matrix for typed-only "
                f"family filtering: {exc}",
            ),
        )
    families = tuple(
        sorted(row.family for row in matrix.families if row.mapping_mode == "typed_error_only")
    )
    return (families, ())


def _code_matches_any_family(*, code: str, families: tuple[str, ...]) -> bool:
    return any(_code_matches_family(code, family) for family in families)

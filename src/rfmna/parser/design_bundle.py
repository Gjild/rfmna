from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Final, Literal, Never, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from rfmna.diagnostics import (
    DiagnosticEvent,
    Severity,
    SolverStage,
    build_diagnostic_event,
    sort_diagnostics,
)
from rfmna.diagnostics.catalog import CANONICAL_DIAGNOSTIC_CATALOG
from rfmna.ir import CanonicalIR, IRAuxUnknown, IRElement, IRNode, IRPort
from rfmna.ir.models import canonicalize_element_kind
from rfmna.parser._loader_exclusions_runtime import (
    LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
    load_loader_temp_exclusions_payload_text,
    load_packaged_loader_temp_exclusions_payload_text,
    repo_loader_temp_exclusions_artifact_path,
)
from rfmna.parser.errors import ParseError
from rfmna.parser.expressions import evaluate_expression
from rfmna.parser.params import ResolvedParameters, resolve_parameters
from rfmna.parser.preflight import IdealVSource, PortDecl, PreflightInput
from rfmna.parser.units import parse_frequency_unit
from rfmna.sweep_engine import frequency_grid

DESIGN_BUNDLE_SCHEMA_ID: Final[str] = "docs/spec/schemas/design_bundle_v1.json"
DESIGN_BUNDLE_SCHEMA_VERSION: Final[int] = 1
_DESIGN_LOADER_CONTEXT: Final[str] = "cli.design_loader"
_DEFAULT_RF_Z0_OHM: Final[float] = 50.0
_TWO_NODE_COUNT: Final[int] = 2
_FOUR_NODE_COUNT: Final[int] = 4
_MIN_SCHEMA_PARAM_RULE_BRANCHES: Final[int] = 2
_LOADER_TEMP_EXCLUSIONS_SCHEMA_VERSION: Final[int] = 1
_INTERIM_DEFERRED_STATUS: Final[str] = "interim_deferred"
_EXCLUDED_CAPABILITY_DIAGNOSTIC_CODE: Final[str] = "E_CLI_DESIGN_EXCLUDED_CAPABILITY"
_EXCLUSION_POLICY_INVALID_CODE: Final[str] = "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID"
_PARAMETER_SWEEP_EXCLUSION_CAPABILITY_ID: Final[str] = "parameter_sweep_support"
_DESIGN_BUNDLE_SCHEMA_RESOURCE_PATH: Final[str] = "rfmna/parser/resources/design_bundle_v1.json"

BundleSweepMode = Literal["linear", "log"]
LoaderDiagnosticCommand = Literal["check", "run"]
_AUX_REQUIRED_KINDS: Final[frozenset[str]] = frozenset(("L", "V", "VCVS"))
_Y_BLOCK_NORMALIZED_KINDS: Final[frozenset[str]] = frozenset(
    (
        "Y",
        "YBLOCK",
        "Y_BLOCK",
        "Y1",
        "Y1P",
        "Y_1P",
        "ONE_PORT_Y",
        "ONE_PORT_Y_BLOCK",
        "Y2",
        "Y2P",
        "Y_2P",
        "TWO_PORT_Y",
        "TWO_PORT_Y_BLOCK",
    )
)
_Z_BLOCK_NORMALIZED_KINDS: Final[frozenset[str]] = frozenset(
    (
        "Z",
        "ZBLOCK",
        "Z_BLOCK",
        "Z1",
        "Z1P",
        "Z_1P",
        "ONE_PORT_Z",
        "ONE_PORT_Z_BLOCK",
        "Z2",
        "Z2P",
        "Z_2P",
        "TWO_PORT_Z",
        "TWO_PORT_Z_BLOCK",
    )
)
_FD_NORMALIZED_KINDS: Final[frozenset[str]] = frozenset(
    (
        "FD",
        "FD_LINEAR",
        "FREQ_DEP",
        "FREQUENCY_DEPENDENT",
        "FREQUENCY_DEPENDENT_LINEAR",
        "COMPACT_LINEAR_FD",
    )
)


@dataclass(frozen=True, slots=True)
class BundleElementDecl:
    element_id: str
    kind: str
    nodes: tuple[str, ...]
    params: Mapping[str, float | str]


@dataclass(frozen=True, slots=True)
class BundlePortDecl:
    port_id: str
    p_plus: str
    p_minus: str
    z0_ohm: float | str


@dataclass(frozen=True, slots=True)
class BundleFrequencyValue:
    value: float | str
    unit: str


@dataclass(frozen=True, slots=True)
class BundleFrequencySweep:
    mode: BundleSweepMode
    start: BundleFrequencyValue
    stop: BundleFrequencyValue
    points: int


@dataclass(frozen=True, slots=True)
class BundleParameterSweep:
    parameter: str
    values: tuple[float | str, ...]


@dataclass(frozen=True, slots=True)
class BundleDocument:
    source_path: Path
    reference_node: str
    declared_nodes: tuple[str, ...]
    parameters: Mapping[str, float | str]
    elements: tuple[BundleElementDecl, ...]
    ports: tuple[BundlePortDecl, ...]
    frequency_sweep: BundleFrequencySweep
    parameter_sweeps: tuple[BundleParameterSweep, ...]


@dataclass(frozen=True, slots=True)
class DesignBundleSchemaContract:
    root_allowed_keys: tuple[str, ...]
    root_required_keys: tuple[str, ...]
    design_allowed_keys: tuple[str, ...]
    design_required_keys: tuple[str, ...]
    analysis_allowed_keys: tuple[str, ...]
    analysis_required_keys: tuple[str, ...]
    analysis_type: str
    accepted_element_kind_tokens: tuple[str, ...]
    supported_kind_node_counts: Mapping[str, int]
    supported_kind_required_params: Mapping[str, tuple[str, ...]]
    frequency_sweep_modes: tuple[str, ...]
    frequency_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedDesignBundle:
    source_path: Path
    preflight_input: PreflightInput
    ir: CanonicalIR
    rf_ports: tuple[IRPort, ...]
    rf_z0_ohm: float | tuple[float, ...]
    frequencies_hz: NDArray[np.float64]
    resolved_parameters: ResolvedParameters
    manifest_input_payload: Mapping[str, object]
    manifest_resolved_params_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LoaderDiagContext:
    element_id: str | None = None
    port_id: str | None = None


@dataclass(frozen=True, slots=True)
class GovernedLoaderExclusion:
    capability_id: str
    label: str
    status: str
    check_diagnostic_code: str
    run_diagnostic_code: str
    witness_capability_id: str


class DesignBundleLoadError(ValueError):
    def __init__(self, diagnostics: Sequence[DiagnosticEvent]) -> None:
        ordered = tuple(sort_diagnostics(diagnostics))
        if not ordered:
            raise ValueError("DesignBundleLoadError requires at least one diagnostic")
        super().__init__(ordered[0].message)
        self.diagnostics = ordered


@dataclass(frozen=True, slots=True)
class _DuplicateMappingKeyError(ValueError):
    key: str
    line: int | None = None
    column: int | None = None


def load_design_bundle_document(design: str | Path) -> ParsedDesignBundle:
    source_path = Path(design)
    payload = _read_payload(source_path)
    document = _parse_bundle_document(payload=payload, source_path=source_path)
    exclusion_diagnostics = _exclusion_diagnostics(document)
    if exclusion_diagnostics:
        raise DesignBundleLoadError(exclusion_diagnostics)
    return _build_parsed_bundle(document, source_payload=payload)


def adapt_loader_diagnostics_for_command(
    diagnostics: Sequence[DiagnosticEvent],
    *,
    command: LoaderDiagnosticCommand,
) -> tuple[DiagnosticEvent, ...]:
    if command == "check":
        return tuple(sort_diagnostics(diagnostics))

    try:
        governed_exclusions = {
            exclusion.capability_id: exclusion for exclusion in _load_governed_loader_exclusions()
        }
    except DesignBundleLoadError:
        return tuple(sort_diagnostics(diagnostics))
    adapted = tuple(
        _adapt_loader_diagnostic_for_command(
            diagnostic,
            command=command,
            governed_exclusions=governed_exclusions,
        )
        for diagnostic in diagnostics
    )
    return tuple(sort_diagnostics(adapted))


def _read_payload(source_path: Path) -> dict[str, object]:
    try:
        payload_text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_READ_FAILED",
                    message="design bundle read failed",
                    suggested_action="provide a readable JSON design bundle path",
                    witness={
                        "design": source_path.as_posix(),
                        "errno": exc.errno,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
        ) from exc

    try:
        payload = _parse_design_bundle_json(payload_text)
    except _DuplicateMappingKeyError as exc:
        witness: dict[str, object] = {
            "design": source_path.as_posix(),
            "key": exc.key,
        }
        if exc.line is not None:
            witness["line"] = exc.line
        if exc.column is not None:
            witness["column"] = exc.column
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_PARSE_FAILED",
                    message="design bundle JSON contains duplicate object keys",
                    suggested_action="remove duplicate JSON object keys and retry",
                    witness=witness,
                ),
            )
        ) from exc
    except json.JSONDecodeError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_PARSE_FAILED",
                    message="design bundle JSON parse failed",
                    suggested_action="fix JSON syntax and retry",
                    witness={
                        "column": exc.colno,
                        "design": source_path.as_posix(),
                        "line": exc.lineno,
                        "position": exc.pos,
                    },
                ),
            )
        ) from exc

    if not isinstance(payload, dict):
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_SCHEMA_INVALID",
                    message="design bundle root must be a JSON object",
                    suggested_action="wrap the design bundle payload in a top-level JSON object",
                    witness={"design": source_path.as_posix(), "json_type": type(payload).__name__},
                ),
            )
        )
    return payload


def _parse_design_bundle_json(payload_text: str) -> object:
    def _reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key, value in pairs:
            if key in mapping:
                raise _DuplicateMappingKeyError(key=str(key))
            mapping[key] = value
        return mapping

    return json.loads(payload_text, object_pairs_hook=_reject_duplicate_keys)


def _repo_design_bundle_schema_artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / DESIGN_BUNDLE_SCHEMA_ID


def _source_tree_design_bundle_schema_resource_path() -> Path:
    return Path(__file__).with_name("resources") / "design_bundle_v1.json"


def _load_design_bundle_schema_payload_text() -> str:
    repo_artifact_path = _repo_design_bundle_schema_artifact_path()
    if repo_artifact_path.exists():
        return repo_artifact_path.read_text(encoding="utf-8")

    resource_path = _source_tree_design_bundle_schema_resource_path()
    if resource_path.exists():
        return resource_path.read_text(encoding="utf-8")

    raise FileNotFoundError(_DESIGN_BUNDLE_SCHEMA_RESOURCE_PATH)


@lru_cache(maxsize=1)
def _load_design_bundle_schema_contract() -> DesignBundleSchemaContract:
    try:
        payload = json.loads(_load_design_bundle_schema_payload_text())
        if not isinstance(payload, dict):
            raise ValueError("design bundle schema root must be an object")
        defs = _schema_mapping(payload.get("$defs"), path="$.$defs")
        design_def = _schema_mapping(defs.get("design"), path="$.$defs.design")
        analysis_def = _schema_mapping(defs.get("analysis"), path="$.$defs.analysis")
        frequency_sweep_def = _schema_mapping(
            defs.get("frequency_sweep"),
            path="$.$defs.frequency_sweep",
        )
        frequency_value_def = _schema_mapping(
            defs.get("frequency_value"),
            path="$.$defs.frequency_value",
        )
        accepted_kind_tokens = _schema_string_sequence(
            _schema_mapping(defs.get("accepted_element_kind_token"), path="$.$defs.accepted_element_kind_token").get("enum"),
            path="$.$defs.accepted_element_kind_token.enum",
        )
        supported_kind_node_counts, supported_kind_required_params = _supported_kind_contract_from_schema(
            defs=defs,
        )
        return DesignBundleSchemaContract(
            root_allowed_keys=_schema_properties(payload, path="$.properties"),
            root_required_keys=_schema_string_sequence(payload.get("required"), path="$.required"),
            design_allowed_keys=_schema_properties(design_def, path="$.$defs.design.properties"),
            design_required_keys=_schema_string_sequence(
                design_def.get("required"),
                path="$.$defs.design.required",
            ),
            analysis_allowed_keys=_schema_properties(analysis_def, path="$.$defs.analysis.properties"),
            analysis_required_keys=_schema_string_sequence(
                analysis_def.get("required"),
                path="$.$defs.analysis.required",
            ),
            analysis_type=_schema_const_string(
                _schema_mapping(analysis_def.get("properties"), path="$.$defs.analysis.properties")
                .get("type"),
                path="$.$defs.analysis.properties.type",
            ),
            accepted_element_kind_tokens=accepted_kind_tokens,
            supported_kind_node_counts=supported_kind_node_counts,
            supported_kind_required_params=supported_kind_required_params,
            frequency_sweep_modes=_schema_enum_strings(
                _schema_mapping(
                    _schema_mapping(frequency_sweep_def.get("properties"), path="$.$defs.frequency_sweep.properties").get("mode"),
                    path="$.$defs.frequency_sweep.properties.mode",
                ),
                path="$.$defs.frequency_sweep.properties.mode",
            ),
            frequency_units=_schema_enum_strings(
                _schema_mapping(
                    _schema_mapping(frequency_value_def.get("properties"), path="$.$defs.frequency_value.properties").get("unit"),
                    path="$.$defs.frequency_value.properties.unit",
                ),
                path="$.$defs.frequency_value.properties.unit",
            ),
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_SCHEMA_INVALID",
                    message="design bundle schema contract artifact is invalid or unavailable",
                    suggested_action=(
                        "restore docs/spec/schemas/design_bundle_v1.json or the packaged runtime copy"
                    ),
                    witness={"schema": DESIGN_BUNDLE_SCHEMA_ID},
                ),
            )
        ) from exc


def _schema_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    key_set = set(value)
    if not all(isinstance(key, str) and key for key in key_set):
        raise ValueError(f"{path} keys must be non-empty strings")
    return cast(Mapping[str, object], value)


def _schema_string_sequence(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty string list")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{path} entries must be non-empty strings")
        items.append(entry)
    return tuple(items)


def _schema_properties(schema_mapping: Mapping[str, object], *, path: str) -> tuple[str, ...]:
    properties = _schema_mapping(schema_mapping.get("properties"), path=path)
    return tuple(sorted(properties))


def _schema_const_string(value: object, *, path: str) -> str:
    mapping = _schema_mapping(value, path=path)
    const = mapping.get("const")
    if not isinstance(const, str) or not const:
        raise ValueError(f"{path}.const must be a non-empty string")
    return const


def _schema_enum_strings(value: Mapping[str, object], *, path: str) -> tuple[str, ...]:
    return _schema_string_sequence(value.get("enum"), path=f"{path}.enum")


def _supported_kind_contract_from_schema(
    *,
    defs: Mapping[str, object],
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    element_def = _schema_mapping(defs.get("element"), path="$.$defs.element")
    all_of = element_def.get("allOf")
    if not isinstance(all_of, list) or not all_of:
        raise ValueError("$.$defs.element.allOf must be a non-empty list")

    node_counts: dict[str, int] = {}
    required_params: dict[str, tuple[str, ...]] = {}
    for index, entry in enumerate(all_of):
        rule = _schema_mapping(entry, path=f"$.$defs.element.allOf[{index}]")
        kind_tokens = _kind_tokens_from_schema_rule(rule=rule, index=index)
        node_count = _node_count_from_schema_rule(rule=rule, index=index)
        params = _required_params_from_schema_rule(rule=rule, index=index)
        for token in kind_tokens:
            canonical_kind = canonicalize_element_kind(token)
            if canonical_kind is None:
                raise ValueError(f"unsupported kind token in schema rule: {token}")
            existing_node_count = node_counts.get(canonical_kind)
            if existing_node_count is not None and existing_node_count != node_count:
                raise ValueError(f"inconsistent node count in schema for {canonical_kind}")
            existing_params = required_params.get(canonical_kind)
            if existing_params is not None and existing_params != params:
                raise ValueError(f"inconsistent required params in schema for {canonical_kind}")
            node_counts[canonical_kind] = node_count
            required_params[canonical_kind] = params
    return node_counts, required_params


def _kind_tokens_from_schema_rule(*, rule: Mapping[str, object], index: int) -> tuple[str, ...]:
    if_branch = _schema_mapping(rule.get("if"), path=f"$.$defs.element.allOf[{index}].if")
    properties = _schema_mapping(
        if_branch.get("properties"),
        path=f"$.$defs.element.allOf[{index}].if.properties",
    )
    kind_selector = _schema_mapping(
        properties.get("kind"),
        path=f"$.$defs.element.allOf[{index}].if.properties.kind",
    )
    const = kind_selector.get("const")
    if isinstance(const, str) and const:
        return (const,)
    return _schema_string_sequence(
        kind_selector.get("enum"),
        path=f"$.$defs.element.allOf[{index}].if.properties.kind.enum",
    )


def _node_count_from_schema_rule(*, rule: Mapping[str, object], index: int) -> int:
    then_branch = _schema_mapping(rule.get("then"), path=f"$.$defs.element.allOf[{index}].then")
    properties = _schema_mapping(
        then_branch.get("properties"),
        path=f"$.$defs.element.allOf[{index}].then.properties",
    )
    node_schema = _schema_mapping(
        properties.get("nodes"),
        path=f"$.$defs.element.allOf[{index}].then.properties.nodes",
    )
    node_ref = node_schema.get("$ref")
    if node_ref == "#/$defs/two_node_string_array":
        return _TWO_NODE_COUNT
    if node_ref == "#/$defs/four_node_string_array":
        return _FOUR_NODE_COUNT
    raise ValueError(f"unsupported node ref in schema rule: {node_ref}")


def _required_params_from_schema_rule(*, rule: Mapping[str, object], index: int) -> tuple[str, ...]:
    then_branch = _schema_mapping(rule.get("then"), path=f"$.$defs.element.allOf[{index}].then")
    properties = _schema_mapping(
        then_branch.get("properties"),
        path=f"$.$defs.element.allOf[{index}].then.properties",
    )
    params_schema = _schema_mapping(
        properties.get("params"),
        path=f"$.$defs.element.allOf[{index}].then.properties.params",
    )
    params_all_of = params_schema.get("allOf")
    if not isinstance(params_all_of, list) or len(params_all_of) < _MIN_SCHEMA_PARAM_RULE_BRANCHES:
        raise ValueError(f"$.$defs.element.allOf[{index}].then.properties.params.allOf is invalid")
    required_branch = _schema_mapping(
        params_all_of[1],
        path=f"$.$defs.element.allOf[{index}].then.properties.params.allOf[1]",
    )
    return _schema_string_sequence(
        required_branch.get("required"),
        path=f"$.$defs.element.allOf[{index}].then.properties.params.allOf[1].required",
    )


def _parse_bundle_document(*, payload: dict[str, object], source_path: Path) -> BundleDocument:
    schema_contract = _load_design_bundle_schema_contract()
    _validate_allowed_keys(
        payload,
        path="$",
        allowed_keys=schema_contract.root_allowed_keys,
        required_keys=schema_contract.root_required_keys,
    )
    schema_id = _require_nonempty_string(payload["schema"], path="$.schema")
    schema_version = _require_int(payload["schema_version"], path="$.schema_version", min_value=1)
    if (
        schema_id != DESIGN_BUNDLE_SCHEMA_ID
        or schema_version != DESIGN_BUNDLE_SCHEMA_VERSION
    ):
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_SCHEMA_UNSUPPORTED",
                    message="unsupported design bundle schema selection",
                    suggested_action=(
                        "set schema to docs/spec/schemas/design_bundle_v1.json and schema_version to 1"
                    ),
                    witness={
                        "active_schema": DESIGN_BUNDLE_SCHEMA_ID,
                        "active_schema_version": DESIGN_BUNDLE_SCHEMA_VERSION,
                        "schema": schema_id,
                        "schema_version": schema_version,
                    },
                ),
            )
        )

    design_raw = _require_mapping(payload["design"], path="$.design")
    analysis_raw = _require_mapping(payload["analysis"], path="$.analysis")
    _validate_allowed_keys(
        design_raw,
        path="$.design",
        allowed_keys=schema_contract.design_allowed_keys,
        required_keys=schema_contract.design_required_keys,
    )
    _validate_allowed_keys(
        analysis_raw,
        path="$.analysis",
        allowed_keys=schema_contract.analysis_allowed_keys,
        required_keys=schema_contract.analysis_required_keys,
    )
    _require_literal_string(
        analysis_raw.get("type"),
        path="$.analysis.type",
        allowed=(schema_contract.analysis_type,),
    )
    return BundleDocument(
        source_path=source_path,
        reference_node=_require_nonempty_string(
            design_raw.get("reference_node"), path="$.design.reference_node"
        ),
        declared_nodes=_parse_string_list(design_raw.get("nodes", []), path="$.design.nodes"),
        parameters=_parse_scalar_mapping(
            design_raw.get("parameters", {}), path="$.design.parameters"
        ),
        elements=_parse_elements(design_raw.get("elements"), path="$.design.elements"),
        ports=_parse_ports(design_raw.get("ports", []), path="$.design.ports"),
        frequency_sweep=_parse_frequency_sweep(
            analysis_raw.get("frequency_sweep"), path="$.analysis.frequency_sweep"
        ),
        parameter_sweeps=_parse_parameter_sweeps(
            analysis_raw.get("parameter_sweeps", []), path="$.analysis.parameter_sweeps"
        ),
    )


def _build_parsed_bundle(
    document: BundleDocument,
    *,
    source_payload: Mapping[str, object],
) -> ParsedDesignBundle:
    resolved_parameters = _resolve_parameters(document.parameters)
    resolved_parameter_map = resolved_parameters.as_dict()
    frequency_values = _build_frequency_values(
        sweep=document.frequency_sweep,
        resolved_parameters=resolved_parameter_map,
    )
    try:
        element_params = tuple(
            _build_ir_element(element=element, resolved_parameters=resolved_parameter_map)
            for element in document.elements
        )
        rf_ports = tuple(
            _build_rf_port(port=port, resolved_parameters=resolved_parameter_map)
            for port in document.ports
        )
        preflight_input = _build_preflight_input(
            document=document,
            resolved_parameters=resolved_parameter_map,
        )
    except DesignBundleLoadError:
        raise
    except ValueError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design bundle model validation failed: {exc}",
                    suggested_action="fix design element and port values, then retry",
                    witness={"design": document.source_path.as_posix(), "stage": "model_build"},
                ),
            )
        ) from exc
    compile_nodes = _compile_node_ids(document=document)
    aux_unknowns = _allocated_aux_unknowns(element_params)
    try:
        ir = CanonicalIR(
            nodes=tuple(
                IRNode(node_id=node_id, is_reference=(node_id == document.reference_node))
                for node_id in compile_nodes
            ),
            aux_unknowns=aux_unknowns,
            elements=element_params,
            ports=rf_ports,
            resolved_params=resolved_parameters.items,
        )
    except ValueError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design bundle model validation failed: {exc}",
                    suggested_action="fix duplicate identifiers and required model fields, then retry",
                    witness={
                        "design": document.source_path.as_posix(),
                        "stage": "canonical_ir",
                    },
                ),
            )
        ) from exc

    return ParsedDesignBundle(
        source_path=document.source_path,
        preflight_input=preflight_input,
        ir=ir,
        rf_ports=ir.ports,
        rf_z0_ohm=_canonical_rf_z0(ir.ports),
        frequencies_hz=frequency_values,
        resolved_parameters=resolved_parameters,
        manifest_input_payload=_manifest_input_payload(
            document=document,
            source_payload=source_payload,
            frequencies_hz=frequency_values,
        ),
        manifest_resolved_params_payload=_manifest_resolved_params_payload(resolved_parameters),
    )


def _resolve_parameters(parameters: Mapping[str, float | str]) -> ResolvedParameters:
    try:
        return resolve_parameters(parameters)
    except ParseError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design parameter resolution failed: {exc.detail.message}",
                    suggested_action="fix parameter expressions and retry",
                    witness={
                        "input_text": exc.detail.input_text,
                        "source_code": exc.detail.code,
                        "witness": list(exc.detail.witness or ()),
                    },
                ),
            )
        ) from exc


def _build_frequency_values(
    *,
    sweep: BundleFrequencySweep,
    resolved_parameters: Mapping[str, float],
) -> NDArray[np.float64]:
    start_hz = _evaluate_frequency_value(
        value=sweep.start,
        resolved_parameters=resolved_parameters,
        path="analysis.frequency_sweep.start",
    )
    stop_hz = _evaluate_frequency_value(
        value=sweep.stop,
        resolved_parameters=resolved_parameters,
        path="analysis.frequency_sweep.stop",
    )
    try:
        return frequency_grid(sweep.mode, start_hz, stop_hz, sweep.points)
    except ParseError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design frequency sweep is invalid: {exc.detail.message}",
                    suggested_action="fix analysis.frequency_sweep bounds or point count and retry",
                    witness={
                        "input_text": exc.detail.input_text,
                        "source_code": exc.detail.code,
                    },
                ),
            )
        ) from exc


def _build_ir_element(
    *,
    element: BundleElementDecl,
    resolved_parameters: Mapping[str, float],
) -> IRElement:
    evaluated_params = tuple(
        (
            param_name,
            _evaluate_scalar_token(
                param_value,
                resolved_parameters=resolved_parameters,
                path=f"design.elements[{element.element_id}].params.{param_name}",
            ),
        )
        for param_name, param_value in sorted(element.params.items())
    )
    return IRElement(
        element_id=element.element_id,
        element_type=element.kind,
        nodes=element.nodes,
        params=evaluated_params,
    )


def _build_rf_port(
    *,
    port: BundlePortDecl,
    resolved_parameters: Mapping[str, float],
) -> IRPort:
    try:
        z0_ohm = _evaluate_scalar_token(
            port.z0_ohm,
            resolved_parameters=resolved_parameters,
            path=f"design.ports[{port.port_id}].z0_ohm",
        )
        return IRPort(
            port_id=port.port_id,
            p_plus=port.p_plus,
            p_minus=port.p_minus,
            z0_ohm=z0_ohm,
        )
    except ValueError as exc:
        message = str(exc)
        known_code = message.split(":", maxsplit=1)[0]
        if known_code in {"E_MODEL_PORT_Z0_COMPLEX", "E_MODEL_PORT_Z0_NONPOSITIVE"}:
            raise DesignBundleLoadError(
                (
                    build_diagnostic_event(
                        code=known_code,
                        message=message.split(":", maxsplit=1)[1].strip(),
                        port_id=port.port_id,
                        witness={
                            "port_id": port.port_id,
                            "path": f"design.ports[{port.port_id}].z0_ohm",
                        },
                    ),
                )
            ) from exc
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design port validation failed: {exc}",
                    suggested_action="fix RF port declarations and retry",
                    witness={"path": f"design.ports[{port.port_id}]"},
                    context=LoaderDiagContext(port_id=port.port_id),
                ),
            )
        ) from exc


def _build_preflight_input(
    *,
    document: BundleDocument,
    resolved_parameters: Mapping[str, float],
) -> PreflightInput:
    explicit_nodes = tuple(document.declared_nodes)
    inferred_nodes = _inferred_node_ids(document)
    missing_nodes = tuple(node_id for node_id in sorted(inferred_nodes) if node_id not in explicit_nodes)
    nodes = explicit_nodes + missing_nodes
    ports = tuple(
        PortDecl(port_id=port.port_id, p_plus=port.p_plus, p_minus=port.p_minus)
        for port in document.ports
    )
    voltage_sources = tuple(
        IdealVSource(
            source_id=element.element_id,
            p_plus=element.nodes[0],
            p_minus=element.nodes[1],
            voltage_v=_evaluate_scalar_token(
                element.params["voltage_v"],
                resolved_parameters=resolved_parameters,
                path=f"design.elements[{element.element_id}].params.voltage_v",
            ),
        )
        for element in document.elements
        if canonicalize_element_kind(element.kind) == "V" and "voltage_v" in element.params
    )
    return PreflightInput(
        nodes=nodes,
        reference_node=document.reference_node,
        ports=ports,
        voltage_sources=voltage_sources,
        hard_constraints=(),
        edges_for_connectivity=_connectivity_edges(document),
    )


def _compile_node_ids(document: BundleDocument) -> tuple[str, ...]:
    nodes = set(_inferred_node_ids(document))
    nodes.add(document.reference_node)
    return tuple(sorted(nodes))


def _inferred_node_ids(document: BundleDocument) -> set[str]:
    nodes = set(document.declared_nodes)
    nodes.add(document.reference_node)
    for element in document.elements:
        nodes.update(element.nodes)
    for port in document.ports:
        nodes.add(port.p_plus)
        nodes.add(port.p_minus)
    return nodes


def _allocated_aux_unknowns(elements: Sequence[IRElement]) -> tuple[IRAuxUnknown, ...]:
    aux_unknowns: list[IRAuxUnknown] = []
    for element in elements:
        canonical_kind = canonicalize_element_kind(element.element_type)
        if canonical_kind in _AUX_REQUIRED_KINDS:
            aux_unknowns.append(
                IRAuxUnknown(aux_id=f"{element.element_id}:i", kind="branch_current", owner_element_id=element.element_id)
            )
    return tuple(aux_unknowns)


def _connectivity_edges(document: BundleDocument) -> tuple[tuple[str, str], ...]:
    edges: set[tuple[str, str]] = set()
    for element in document.elements:
        edges.update(_edges_for_element(element))
    return tuple(sorted(edges))


def _edges_for_element(element: BundleElementDecl) -> tuple[tuple[str, str], ...]:
    nodes = element.nodes
    if len(nodes) == _TWO_NODE_COUNT:
        left, right = nodes
        if left == right:
            return ()
        return ((_ordered_edge(left, right)),)
    if len(nodes) == _FOUR_NODE_COUNT:
        left_edge = () if nodes[0] == nodes[1] else (_ordered_edge(nodes[0], nodes[1]),)
        right_edge = () if nodes[2] == nodes[3] else (_ordered_edge(nodes[2], nodes[3]),)
        return left_edge + right_edge
    return ()


def _ordered_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _canonical_rf_z0(ports: Sequence[IRPort]) -> float | tuple[float, ...]:
    ordered_ports = tuple(sorted(ports, key=lambda port: port.port_id))
    if not ordered_ports:
        return _DEFAULT_RF_Z0_OHM
    z0_values = tuple(port.z0_ohm for port in ordered_ports)
    first = z0_values[0]
    if all(value == first for value in z0_values[1:]):
        return float(first)
    return z0_values


def _manifest_input_payload(
    *,
    document: BundleDocument,
    source_payload: Mapping[str, object],
    frequencies_hz: NDArray[np.float64],
) -> Mapping[str, object]:
    return {
        "analysis": _load_design_bundle_schema_contract().analysis_type,
        "design_bundle": dict(source_payload),
        "frequencies_hz": [float(value) for value in frequencies_hz.tolist()],
    }


def _manifest_resolved_params_payload(
    resolved_parameters: ResolvedParameters,
) -> Mapping[str, object]:
    return {
        name: float(value)
        for name, value in resolved_parameters.items
    }


def _exclusion_diagnostics(document: BundleDocument) -> tuple[DiagnosticEvent, ...]:
    schema_contract = _load_design_bundle_schema_contract()
    supported_kinds = tuple(sorted(schema_contract.supported_kind_node_counts))
    governed_exclusions = {
        exclusion.capability_id: exclusion for exclusion in _load_governed_loader_exclusions()
    }
    diagnostics: list[DiagnosticEvent] = []
    if document.parameter_sweeps:
        for parameter_sweep in document.parameter_sweeps:
            exclusion = governed_exclusions.get("parameter_sweep_support")
            if exclusion is None:
                diagnostics.append(
                    _loader_diag(
                        code="E_CLI_DESIGN_VALUE_INVALID",
                        message=(
                            "design bundle parameter sweeps are not implemented by the active loader runtime"
                        ),
                        suggested_action=(
                            "remove analysis.parameter_sweeps or implement parameter-sweep support before closure"
                        ),
                        witness={
                            "capability_id": "parameter_sweep_support",
                            "parameter": parameter_sweep.parameter,
                            "path": "analysis.parameter_sweeps",
                            "policy_state": "not_deferred",
                        },
                    )
                )
                continue
            diagnostics.append(
                _governed_exclusion_diagnostic(
                    exclusion=exclusion,
                    message="design bundle parameter sweeps are temporarily excluded in P3-01 interim scope",
                    suggested_action="remove analysis.parameter_sweeps or wait for Phase 3 closure support",
                    witness_extra={
                        "parameter": parameter_sweep.parameter,
                        "path": "analysis.parameter_sweeps",
                    },
                )
            )

    for element in document.elements:
        normalized_kind = _normalize_kind_token(element.kind)
        capability_id = _excluded_capability_id(normalized_kind)
        if capability_id is None:
            canonical_kind = canonicalize_element_kind(element.kind)
            if canonical_kind is None:
                diagnostics.append(
                    build_diagnostic_event(
                        code="E_IR_KIND_UNKNOWN",
                        message=f"unsupported element kind '{element.kind}' for element '{element.element_id}'",
                        element_id=element.element_id,
                        witness={
                            "element_id": element.element_id,
                            "normalized_candidate": normalized_kind,
                            "raw_kind": element.kind,
                            "supported_kinds": supported_kinds,
                        },
                    )
                )
            continue
        exclusion = governed_exclusions.get(capability_id)
        if exclusion is None:
            diagnostics.append(
                build_diagnostic_event(
                    code="E_IR_KIND_UNKNOWN",
                    message=f"unsupported element kind '{element.kind}' for element '{element.element_id}'",
                    element_id=element.element_id,
                    witness={
                        "deferred_capability_id": capability_id,
                        "element_id": element.element_id,
                        "normalized_candidate": normalized_kind,
                        "policy_state": "not_deferred",
                        "raw_kind": element.kind,
                        "supported_kinds": supported_kinds,
                    },
                )
            )
            continue
        diagnostics.append(
            _governed_exclusion_diagnostic(
                exclusion=exclusion,
                message=f"design element '{element.element_id}' uses temporarily excluded capability '{capability_id}'",
                suggested_action="remove the excluded element kind or wait for Phase 3 closure support",
                witness_extra={
                    "element_id": element.element_id,
                    "kind": element.kind,
                    "normalized_kind": normalized_kind,
                },
                context=LoaderDiagContext(element_id=element.element_id),
            )
        )
    return tuple(sort_diagnostics(diagnostics))


def _excluded_capability_id(normalized_kind: str) -> str | None:
    if normalized_kind in _FD_NORMALIZED_KINDS:
        return "frequency_dependent_compact_linear_forms"
    if normalized_kind in _Y_BLOCK_NORMALIZED_KINDS:
        return "y_block_elements"
    if normalized_kind in _Z_BLOCK_NORMALIZED_KINDS:
        return "z_block_elements"
    return None


@lru_cache(maxsize=1)
def _load_governed_loader_exclusions() -> tuple[GovernedLoaderExclusion, ...]:
    payload = _read_governed_loader_exclusions_payload()
    exclusions = _parse_governed_loader_exclusions_payload(payload)
    _validate_governed_loader_exclusions(exclusions)
    return exclusions


def _read_governed_loader_exclusions_payload() -> dict[str, object]:
    repo_artifact_path = repo_loader_temp_exclusions_artifact_path()
    try:
        payload = _parse_governed_loader_exclusions_json(load_loader_temp_exclusions_payload_text())
    except FileNotFoundError as exc:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact is unavailable",
                    witness={"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE},
                ),
            )
        ) from exc
    _validate_missing_governed_exclusions_artifact_policy(repo_artifact_path=repo_artifact_path)
    return payload


def _parse_governed_loader_exclusions_json(payload_text: str) -> dict[str, object]:
    try:
        payload = yaml.load(payload_text, Loader=_LoaderTempExclusionsSafeLoader)
    except _DuplicateMappingKeyError as exc:
        witness: dict[str, object] = {
            "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
            "key": exc.key,
        }
        if exc.line is not None:
            witness["line"] = exc.line
        if exc.column is not None:
            witness["column"] = exc.column
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact contains duplicate mapping keys",
                    witness=witness,
                ),
            )
        ) from exc
    except yaml.YAMLError as exc:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact is not valid YAML",
                    witness={"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE},
                ),
            )
        ) from exc

    if not isinstance(payload, dict):
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="packaged loader temporary exclusions payload must be a mapping",
                    witness={"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE},
                ),
            )
        )
    return payload


class _LoaderTempExclusionsSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


def _construct_unique_yaml_mapping(
    loader: _LoaderTempExclusionsSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateMappingKeyError(
                key=str(key),
                line=key_node.start_mark.line + 1,
                column=key_node.start_mark.column + 1,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_LoaderTempExclusionsSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _parse_governed_loader_exclusions_payload(
    payload: Mapping[str, object],
) -> tuple[GovernedLoaderExclusion, ...]:
    _validate_governed_exclusion_allowed_keys(
        payload,
        path="$",
        allowed_keys=("schema_version", "exclusions", "notes"),
        required_keys=("schema_version", "exclusions"),
    )
    schema_version = payload.get("schema_version")
    if schema_version != _LOADER_TEMP_EXCLUSIONS_SCHEMA_VERSION:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact has an unsupported schema_version",
                    witness={
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                        "schema_version": schema_version,
                    },
                ),
            )
        )

    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact notes must be a string when present",
                    witness={
                        "notes_type": type(notes).__name__,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )

    exclusions_raw = payload.get("exclusions")
    if not isinstance(exclusions_raw, list):
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact must define exclusions as a list",
                    witness={"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE},
                ),
            )
        )

    exclusions: list[GovernedLoaderExclusion] = []
    seen_ids: set[str] = set()
    for index, raw_exclusion in enumerate(exclusions_raw):
        exclusions.append(
            _parse_governed_loader_exclusion(
                raw_exclusion=raw_exclusion,
                index=index,
                seen_ids=seen_ids,
            )
        )
    return tuple(exclusions)


def _parse_governed_loader_exclusion(
    *,
    raw_exclusion: object,
    index: int,
    seen_ids: set[str],
) -> GovernedLoaderExclusion:
    if not isinstance(raw_exclusion, dict):
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions entries must be mappings",
                    witness={
                        "index": index,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    _validate_governed_exclusion_allowed_keys(
        raw_exclusion,
        path=f"$.exclusions[{index}]",
        allowed_keys=(
            "capability_id",
            "label",
            "status",
            "check_diagnostic_code",
            "run_diagnostic_code",
            "witness_capability_id",
        ),
        required_keys=(
            "capability_id",
            "label",
            "status",
            "check_diagnostic_code",
            "run_diagnostic_code",
            "witness_capability_id",
        ),
    )
    capability_id = _require_governed_exclusion_string(
        raw_exclusion.get("capability_id"),
        field_name="capability_id",
        index=index,
    )
    if capability_id in seen_ids:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact contains duplicate capability_id values",
                    witness={
                        "capability_id": capability_id,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    seen_ids.add(capability_id)
    return GovernedLoaderExclusion(
        capability_id=capability_id,
        label=_require_governed_exclusion_string(
            raw_exclusion.get("label"),
            field_name="label",
            index=index,
        ),
        status=_require_governed_exclusion_string(
            raw_exclusion.get("status"),
            field_name="status",
            index=index,
        ),
        check_diagnostic_code=_require_governed_exclusion_string(
            raw_exclusion.get("check_diagnostic_code"),
            field_name="check_diagnostic_code",
            index=index,
        ),
        run_diagnostic_code=_require_governed_exclusion_string(
            raw_exclusion.get("run_diagnostic_code"),
            field_name="run_diagnostic_code",
            index=index,
        ),
        witness_capability_id=_require_governed_exclusion_string(
            raw_exclusion.get("witness_capability_id"),
            field_name="witness_capability_id",
            index=index,
        ),
    )


def _validate_governed_loader_exclusions(
    exclusions: Sequence[GovernedLoaderExclusion],
) -> None:
    sorted_capability_ids = tuple(sorted(exclusion.capability_id for exclusion in exclusions))
    if tuple(exclusion.capability_id for exclusion in exclusions) != sorted_capability_ids:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact capability_id values must be sorted ascending",
                    witness={"source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE},
                ),
            )
        )

    for exclusion in exclusions:
        _validate_governed_loader_exclusion(exclusion)
    _validate_governed_loader_exclusion_completeness(exclusions)


def _validate_governed_loader_exclusion_completeness(
    exclusions: Sequence[GovernedLoaderExclusion],
) -> None:
    expected_exclusions = _expected_interim_governed_exclusions()
    if tuple(exclusions) == expected_exclusions:
        return
    actual_capability_ids = tuple(exclusion.capability_id for exclusion in exclusions)
    expected_capability_ids = tuple(exclusion.capability_id for exclusion in expected_exclusions)
    missing_capability_ids = tuple(
        capability_id for capability_id in expected_capability_ids if capability_id not in actual_capability_ids
    )
    unexpected_capability_ids = tuple(
        capability_id for capability_id in actual_capability_ids if capability_id not in expected_capability_ids
    )
    expected_by_capability = {
        exclusion.capability_id: exclusion for exclusion in expected_exclusions
    }
    actual_by_capability = {exclusion.capability_id: exclusion for exclusion in exclusions}
    drifted_capability_ids = tuple(
        capability_id
        for capability_id in expected_capability_ids
        if capability_id in actual_by_capability
        and actual_by_capability[capability_id] != expected_by_capability[capability_id]
    )
    mismatched_fields_by_capability = {
        capability_id: _mismatched_exclusion_fields(
            actual=actual_by_capability[capability_id],
            expected=expected_by_capability[capability_id],
        )
        for capability_id in drifted_capability_ids
    }
    raise DesignBundleLoadError(
        (
            _exclusion_policy_diagnostic(
                message=(
                    "loader temporary exclusions artifact must match the packaged interim exclusion policy"
                ),
                witness={
                    "actual_capability_ids": list(actual_capability_ids),
                    "drifted_capability_ids": list(drifted_capability_ids),
                    "expected_capability_ids": list(expected_capability_ids),
                    "missing_capability_ids": list(missing_capability_ids),
                    "mismatched_fields_by_capability": mismatched_fields_by_capability,
                    "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    "unexpected_capability_ids": list(unexpected_capability_ids),
                },
            ),
        )
    )


def _expected_interim_governed_exclusions() -> tuple[GovernedLoaderExclusion, ...]:
    try:
        payload = _parse_governed_loader_exclusions_json(
            load_packaged_loader_temp_exclusions_payload_text()
        )
    except (DesignBundleLoadError, FileNotFoundError):
        return ()
    exclusions = _parse_governed_loader_exclusions_payload(payload)
    for exclusion in exclusions:
        _validate_governed_loader_exclusion(exclusion)
    return exclusions


def _active_interim_exclusion_capability_ids() -> tuple[str, ...]:
    return tuple(
        exclusion.capability_id for exclusion in _expected_interim_governed_exclusions()
    )


def _mismatched_exclusion_fields(
    *,
    actual: GovernedLoaderExclusion,
    expected: GovernedLoaderExclusion,
) -> list[str]:
    mismatches: list[str] = []
    for field_name in (
        "label",
        "status",
        "check_diagnostic_code",
        "run_diagnostic_code",
        "witness_capability_id",
    ):
        if getattr(actual, field_name) != getattr(expected, field_name):
            mismatches.append(field_name)
    return mismatches


def _validate_missing_governed_exclusions_artifact_policy(*, repo_artifact_path: Path) -> None:
    if repo_artifact_path.is_file() or not repo_artifact_path.parent.is_dir():
        return
    active_capability_ids = _active_interim_exclusion_capability_ids()
    if not active_capability_ids:
        return
    raise DesignBundleLoadError(
        (
            _exclusion_policy_diagnostic(
                message="loader temporary exclusions artifact is missing while interim exclusions remain active",
                witness={
                    "expected_capability_ids": list(active_capability_ids),
                    "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                },
            ),
        )
    )


def _validate_governed_loader_exclusion(exclusion: GovernedLoaderExclusion) -> None:
    if exclusion.status != _INTERIM_DEFERRED_STATUS:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact entries must use status 'interim_deferred'",
                    witness={
                        "capability_id": exclusion.capability_id,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                        "status": exclusion.status,
                    },
                ),
            )
        )
    _validate_governed_loader_exclusion_diagnostic_code(
        exclusion=exclusion,
        field_name="check_diagnostic_code",
        code=exclusion.check_diagnostic_code,
    )
    _validate_governed_loader_exclusion_diagnostic_code(
        exclusion=exclusion,
        field_name="run_diagnostic_code",
        code=exclusion.run_diagnostic_code,
    )
    if exclusion.witness_capability_id != exclusion.capability_id:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact witness_capability_id must match capability_id",
                    witness={
                        "capability_id": exclusion.capability_id,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                        "witness_capability_id": exclusion.witness_capability_id,
                    },
                ),
            )
        )


def _require_governed_exclusion_string(value: object, *, field_name: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact fields must be non-empty strings",
                    witness={
                        "field_name": field_name,
                        "index": index,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    return value.strip()


def _validate_governed_exclusion_allowed_keys(
    mapping: Mapping[str, object],
    *,
    path: str,
    allowed_keys: Sequence[str],
    required_keys: Sequence[str],
) -> None:
    allowed = set(allowed_keys)
    missing = tuple(key for key in required_keys if key not in mapping)
    if missing:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message=f"loader temporary exclusions artifact is missing required keys at {path}",
                    witness={
                        "missing_keys": list(missing),
                        "path": path,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    extras = tuple(sorted(set(mapping) - allowed))
    if extras:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message=f"loader temporary exclusions artifact contains unsupported keys at {path}",
                    witness={
                        "path": path,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                        "unsupported_keys": list(extras),
                    },
                ),
            )
        )


def _governed_exclusion_diagnostic(
    *,
    exclusion: GovernedLoaderExclusion,
    message: str,
    suggested_action: str,
    witness_extra: Mapping[str, object],
    context: LoaderDiagContext | None = None,
) -> DiagnosticEvent:
    return _loader_diag(
        code=_governed_exclusion_diagnostic_code(exclusion, command="check"),
        message=message,
        suggested_action=suggested_action,
        witness={
            "capability_id": exclusion.witness_capability_id,
            **dict(witness_extra),
        },
        context=context,
    )


def _exclusion_policy_diagnostic(*, message: str, witness: Mapping[str, object]) -> DiagnosticEvent:
    return _loader_diag(
        code=_EXCLUSION_POLICY_INVALID_CODE,
        message=message,
        suggested_action=(
            "fix docs/dev/p3_loader_temporary_exclusions.yaml to match the active interim loader exclusions"
        ),
        witness=witness,
    )


def _validate_governed_loader_exclusion_diagnostic_code(
    *,
    exclusion: GovernedLoaderExclusion,
    field_name: str,
    code: str,
) -> None:
    catalog_entry = CANONICAL_DIAGNOSTIC_CATALOG.get(code)
    if catalog_entry is None:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact diagnostic codes must exist in the canonical catalog",
                    witness={
                        "capability_id": exclusion.capability_id,
                        "code": code,
                        "field_name": field_name,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    if code != _EXCLUDED_CAPABILITY_DIAGNOSTIC_CODE:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message=(
                        "loader temporary exclusions artifact diagnostic codes must use the exclusion-specific runtime code"
                    ),
                    witness={
                        "capability_id": exclusion.capability_id,
                        "code": code,
                        "expected_code": _EXCLUDED_CAPABILITY_DIAGNOSTIC_CODE,
                        "field_name": field_name,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )
    if catalog_entry.severity is not Severity.ERROR or catalog_entry.solver_stage is not SolverStage.PARSE:
        raise DesignBundleLoadError(
            (
                _exclusion_policy_diagnostic(
                    message="loader temporary exclusions artifact diagnostic codes must be parse-stage error diagnostics",
                    witness={
                        "capability_id": exclusion.capability_id,
                        "code": code,
                        "field_name": field_name,
                        "severity": catalog_entry.severity,
                        "solver_stage": catalog_entry.solver_stage,
                        "source": LOADER_TEMP_EXCLUSIONS_ARTIFACT_SOURCE,
                    },
                ),
            )
        )


def _governed_exclusion_diagnostic_code(
    exclusion: GovernedLoaderExclusion,
    *,
    command: LoaderDiagnosticCommand,
) -> str:
    if command == "run":
        return exclusion.run_diagnostic_code
    return exclusion.check_diagnostic_code


def _adapt_loader_diagnostic_for_command(
    diagnostic: DiagnosticEvent,
    *,
    command: LoaderDiagnosticCommand,
    governed_exclusions: Mapping[str, GovernedLoaderExclusion],
) -> DiagnosticEvent:
    witness = diagnostic.witness
    if not isinstance(witness, Mapping):
        return diagnostic
    capability_id = witness.get("capability_id")
    if not isinstance(capability_id, str):
        return diagnostic
    exclusion = governed_exclusions.get(capability_id)
    if exclusion is None:
        return diagnostic
    if diagnostic.code != exclusion.check_diagnostic_code:
        return diagnostic

    return build_diagnostic_event(
        code=_governed_exclusion_diagnostic_code(exclusion, command=command),
        severity=diagnostic.severity,
        message=diagnostic.message,
        suggested_action=diagnostic.suggested_action,
        solver_stage=diagnostic.solver_stage,
        element_id=diagnostic.element_id,
        node_id=(None if diagnostic.node_context is None else diagnostic.node_context.node_id),
        port_id=(None if diagnostic.port_context is None else diagnostic.port_context.port_id),
        frequency_hz=diagnostic.frequency_hz,
        frequency_index=diagnostic.frequency_index,
        sweep_index=diagnostic.sweep_index,
        witness=diagnostic.witness,
    )


def _normalize_kind_token(raw_kind: str) -> str:
    return raw_kind.strip().upper().replace("-", "_").replace(" ", "_")


def _evaluate_frequency_value(
    *,
    value: BundleFrequencyValue,
    resolved_parameters: Mapping[str, float],
    path: str,
) -> float:
    scalar = _evaluate_scalar_token(value.value, resolved_parameters=resolved_parameters, path=path)
    try:
        unit_scale = parse_frequency_unit(value.unit)
    except ParseError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design frequency unit is invalid: {exc.detail.message}",
                    suggested_action="choose frequency units from Hz, kHz, MHz, GHz",
                    witness={"input_text": exc.detail.input_text, "path": path},
                ),
            )
        ) from exc
    frequency_hz = scalar * unit_scale
    if not isfinite(frequency_hz):
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message="design frequency value must be finite after unit scaling",
                    suggested_action="fix the frequency expression or unit and retry",
                    witness={"path": path, "unit": value.unit},
                ),
            )
        )
    return float(frequency_hz)


def _evaluate_scalar_token(
    token: float | str,
    *,
    resolved_parameters: Mapping[str, float],
    path: str,
) -> float:
    if isinstance(token, float):
        if not isfinite(token):
            raise DesignBundleLoadError(
                (
                    _loader_diag(
                        code="E_CLI_DESIGN_VALUE_INVALID",
                        message="design numeric values must be finite",
                        suggested_action="replace non-finite numeric values with finite numbers",
                        witness={"path": path},
                    ),
                )
            )
        return token
    try:
        return float(evaluate_expression(token, resolved_parameters))
    except ParseError as exc:
        raise DesignBundleLoadError(
            (
                _loader_diag(
                    code="E_CLI_DESIGN_VALUE_INVALID",
                    message=f"design expression evaluation failed: {exc.detail.message}",
                    suggested_action="fix the expression syntax or referenced parameters and retry",
                    witness={
                        "input_text": exc.detail.input_text,
                        "path": path,
                        "source_code": exc.detail.code,
                        "witness": list(exc.detail.witness or ()),
                    },
                ),
            )
        ) from exc


def _parse_elements(value: object, *, path: str) -> tuple[BundleElementDecl, ...]:
    schema_contract = _load_design_bundle_schema_contract()
    raw_elements = _require_list(value, path=path)
    elements: list[BundleElementDecl] = []
    for index, raw_element in enumerate(raw_elements):
        element_path = f"{path}[{index}]"
        element_mapping = _require_mapping(raw_element, path=element_path)
        _validate_allowed_keys(
            element_mapping,
            path=element_path,
            allowed_keys=("id", "kind", "nodes", "params"),
            required_keys=("id", "kind", "nodes", "params"),
        )
        kind = _require_literal_string(
            element_mapping["kind"],
            path=f"{element_path}.kind",
            allowed=schema_contract.accepted_element_kind_tokens,
        )
        nodes = _parse_string_list(element_mapping["nodes"], path=f"{element_path}.nodes")
        params = _parse_scalar_mapping(element_mapping["params"], path=f"{element_path}.params")
        _validate_supported_element_schema_shape(
            kind=kind,
            nodes=nodes,
            params=params,
            element_path=element_path,
        )
        elements.append(
            BundleElementDecl(
                element_id=_require_nonempty_string(element_mapping["id"], path=f"{element_path}.id"),
                kind=kind,
                nodes=nodes,
                params=params,
            )
        )
    return tuple(elements)


def _validate_supported_element_schema_shape(
    *,
    kind: str,
    nodes: tuple[str, ...],
    params: Mapping[str, float | str],
    element_path: str,
) -> None:
    schema_contract = _load_design_bundle_schema_contract()
    canonical_kind = canonicalize_element_kind(kind)
    if canonical_kind is None or canonical_kind not in schema_contract.supported_kind_node_counts:
        return

    expected_node_count = schema_contract.supported_kind_node_counts[canonical_kind]
    if len(nodes) != expected_node_count:
        _schema_error(
            f"expected exactly {expected_node_count} node ids for kind {kind}",
            path=f"{element_path}.nodes",
        )

    missing_params = tuple(
        param_name
        for param_name in schema_contract.supported_kind_required_params[canonical_kind]
        if param_name not in params
    )
    if missing_params:
        _schema_error(
            f"missing required keys: {', '.join(missing_params)}",
            path=f"{element_path}.params",
        )


def _parse_ports(value: object, *, path: str) -> tuple[BundlePortDecl, ...]:
    raw_ports = _require_list(value, path=path)
    ports: list[BundlePortDecl] = []
    for index, raw_port in enumerate(raw_ports):
        port_path = f"{path}[{index}]"
        port_mapping = _require_mapping(raw_port, path=port_path)
        _validate_allowed_keys(
            port_mapping,
            path=port_path,
            allowed_keys=("id", "p_plus", "p_minus", "z0_ohm"),
            required_keys=("id", "p_plus", "p_minus"),
        )
        ports.append(
            BundlePortDecl(
                port_id=_require_nonempty_string(port_mapping["id"], path=f"{port_path}.id"),
                p_plus=_require_nonempty_string(port_mapping["p_plus"], path=f"{port_path}.p_plus"),
                p_minus=_require_nonempty_string(port_mapping["p_minus"], path=f"{port_path}.p_minus"),
                z0_ohm=_parse_scalar_token(port_mapping.get("z0_ohm", _DEFAULT_RF_Z0_OHM), path=f"{port_path}.z0_ohm"),
            )
        )
    return tuple(ports)


def _parse_frequency_sweep(value: object, *, path: str) -> BundleFrequencySweep:
    schema_contract = _load_design_bundle_schema_contract()
    raw = _require_mapping(value, path=path)
    _validate_allowed_keys(
        raw,
        path=path,
        allowed_keys=("mode", "start", "stop", "points"),
        required_keys=("mode", "start", "stop", "points"),
    )
    mode = cast(
        BundleSweepMode,
        _require_literal_string(
            raw["mode"],
            path=f"{path}.mode",
            allowed=schema_contract.frequency_sweep_modes,
        ),
    )
    return BundleFrequencySweep(
        mode=mode,
        start=_parse_frequency_value(raw["start"], path=f"{path}.start"),
        stop=_parse_frequency_value(raw["stop"], path=f"{path}.stop"),
        points=_require_int(raw["points"], path=f"{path}.points", min_value=1),
    )


def _parse_frequency_value(value: object, *, path: str) -> BundleFrequencyValue:
    schema_contract = _load_design_bundle_schema_contract()
    raw = _require_mapping(value, path=path)
    _validate_allowed_keys(
        raw,
        path=path,
        allowed_keys=("value", "unit"),
        required_keys=("value", "unit"),
    )
    return BundleFrequencyValue(
        value=_parse_scalar_token(raw["value"], path=f"{path}.value"),
        unit=_require_literal_string(raw["unit"], path=f"{path}.unit", allowed=schema_contract.frequency_units),
    )


def _parse_parameter_sweeps(value: object, *, path: str) -> tuple[BundleParameterSweep, ...]:
    raw_sweeps = _require_list(value, path=path)
    sweeps: list[BundleParameterSweep] = []
    for index, raw_sweep in enumerate(raw_sweeps):
        sweep_path = f"{path}[{index}]"
        sweep_mapping = _require_mapping(raw_sweep, path=sweep_path)
        _validate_allowed_keys(
            sweep_mapping,
            path=sweep_path,
            allowed_keys=("parameter", "values"),
            required_keys=("parameter", "values"),
        )
        sweeps.append(
            BundleParameterSweep(
                parameter=_require_nonempty_string(
                    sweep_mapping["parameter"], path=f"{sweep_path}.parameter"
                ),
                values=tuple(
                    _parse_scalar_token(raw_value, path=f"{sweep_path}.values[{value_index}]")
                    for value_index, raw_value in enumerate(
                        _require_list(sweep_mapping["values"], path=f"{sweep_path}.values")
                    )
                ),
            )
        )
    return tuple(sweeps)


def _parse_scalar_mapping(value: object, *, path: str) -> Mapping[str, float | str]:
    raw_mapping = _require_mapping(value, path=path)
    parsed: dict[str, float | str] = {}
    for key in sorted(raw_mapping):
        if not key.strip():
            _schema_error("mapping keys must be non-empty strings", path=path)
        parsed[key] = _parse_scalar_token(raw_mapping[key], path=f"{path}.{key}")
    return parsed


def _parse_string_list(value: object, *, path: str) -> tuple[str, ...]:
    raw_items = _require_list(value, path=path)
    return tuple(
        _require_nonempty_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(raw_items)
    )


def _parse_scalar_token(value: object, *, path: str) -> float | str:
    if isinstance(value, bool):
        _schema_error("boolean values are not allowed", path=path)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            _schema_error("numeric values must be finite", path=path)
        return numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            _schema_error("string values must be non-empty", path=path)
        return text
    _schema_error("expected a finite number or non-empty expression string", path=path)


def _validate_allowed_keys(
    mapping: Mapping[str, object],
    *,
    path: str,
    allowed_keys: Sequence[str],
    required_keys: Sequence[str],
) -> None:
    allowed = set(allowed_keys)
    missing = tuple(key for key in required_keys if key not in mapping)
    if missing:
        _schema_error(f"missing required keys: {', '.join(missing)}", path=path)
    extras = tuple(sorted(set(mapping) - allowed))
    if extras:
        _schema_error(f"unsupported keys: {', '.join(extras)}", path=path)


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _schema_error("expected a JSON object", path=path)
    mapping = cast(dict[object, object], value)
    for key in mapping:
        if not isinstance(key, str) or not key:
            _schema_error("object keys must be non-empty strings", path=path)
    return cast(dict[str, object], mapping)


def _require_list(value: object, *, path: str) -> list[object]:
    if not isinstance(value, list):
        _schema_error("expected a JSON array", path=path)
    return cast(list[object], value)


def _require_nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        _schema_error("expected a non-empty string", path=path)
    text = value.strip()
    if not text:
        _schema_error("expected a non-empty string", path=path)
    return text


def _require_literal_string(value: object, *, path: str, allowed: Sequence[str]) -> str:
    text = _require_nonempty_string(value, path=path)
    if text not in allowed:
        _schema_error(f"expected one of: {', '.join(allowed)}", path=path)
    return text


def _require_int(value: object, *, path: str, min_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _schema_error("expected an integer", path=path)
    if value < min_value:
        _schema_error(f"expected an integer >= {min_value}", path=path)
    return value


def _schema_error(message: str, *, path: str) -> Never:
    raise DesignBundleLoadError(
        (
            _loader_diag(
                code="E_CLI_DESIGN_SCHEMA_INVALID",
                message=f"design bundle schema validation failed at {path}: {message}",
                suggested_action="fix the design bundle shape to match design_bundle_v1.json",
                witness={"path": path},
            ),
        )
    )


def _loader_diag(
    *,
    code: str,
    message: str,
    suggested_action: str,
    witness: Mapping[str, object],
    context: LoaderDiagContext | None = None,
) -> DiagnosticEvent:
    resolved_context = context or LoaderDiagContext(element_id=_DESIGN_LOADER_CONTEXT)
    return build_diagnostic_event(
        code=code,
        message=message,
        suggested_action=suggested_action,
        solver_stage=SolverStage.PARSE,
        element_id=resolved_context.element_id,
        port_id=resolved_context.port_id,
        witness=witness,
    )

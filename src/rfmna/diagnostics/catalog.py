from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .models import Severity, SolverStage


@dataclass(frozen=True, slots=True)
class DiagnosticCatalogEntry:
    code: str
    severity: Severity
    solver_stage: SolverStage
    suggested_action: str

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("diagnostic catalog code must be non-empty")
        if not self.suggested_action:
            raise ValueError(
                f"diagnostic catalog entry '{self.code}' suggested_action must be non-empty"
            )


def _entry(
    code: str,
    severity: Severity,
    solver_stage: SolverStage,
    suggested_action: str,
) -> DiagnosticCatalogEntry:
    return DiagnosticCatalogEntry(
        code=code,
        severity=severity,
        solver_stage=solver_stage,
        suggested_action=suggested_action,
    )


def _build_catalog(
    entries: tuple[DiagnosticCatalogEntry, ...],
) -> Mapping[str, DiagnosticCatalogEntry]:
    catalog: dict[str, DiagnosticCatalogEntry] = {}
    for entry in entries:
        if entry.code in catalog:
            raise ValueError(f"duplicate diagnostic catalog code: {entry.code}")
        catalog[entry.code] = entry
    return MappingProxyType(catalog)


_CATALOG_ENTRIES: tuple[DiagnosticCatalogEntry, ...] = (
    _entry(
        "E_MODEL_R_NONPOSITIVE",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set resistance_ohm to a finite value > 0",
    ),
    _entry(
        "E_MODEL_C_NEGATIVE",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set capacitance_f to a finite value >= 0",
    ),
    _entry(
        "E_MODEL_G_NEGATIVE",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set conductance_s to a finite value >= 0",
    ),
    _entry(
        "E_MODEL_L_NONPOSITIVE",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set inductance_h to a finite value > 0",
    ),
    _entry(
        "E_MODEL_ISRC_INVALID",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set current_a to a finite value",
    ),
    _entry(
        "E_MODEL_VSRC_INVALID",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set voltage_v to a finite value",
    ),
    _entry(
        "E_MODEL_VCCS_INVALID",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set transconductance_s to a finite value",
    ),
    _entry(
        "E_MODEL_VCVS_INVALID",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set gain_mu to a finite value",
    ),
    _entry(
        "E_IR_KIND_UNKNOWN",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "use one of: R,C,G,L,I,V,VCCS,VCVS",
    ),
    _entry(
        "E_MODEL_PORT_Z0_COMPLEX",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set port z0_ohm to a finite real value > 0",
    ),
    _entry(
        "E_MODEL_PORT_Z0_NONPOSITIVE",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "set port z0_ohm to a finite real value > 0",
    ),
    _entry(
        "E_TOPO_REFERENCE_INVALID",
        Severity.ERROR,
        SolverStage.PREFLIGHT,
        "declare exactly one valid reference node and deduplicate node declarations",
    ),
    _entry(
        "E_TOPO_FLOATING_NODE",
        Severity.ERROR,
        SolverStage.PREFLIGHT,
        "connect floating nodes/components to the reference-reachable graph",
    ),
    _entry(
        "E_TOPO_PORT_INVALID",
        Severity.ERROR,
        SolverStage.PREFLIGHT,
        "ensure each port has unique id/orientation and distinct declared nodes",
    ),
    _entry(
        "E_TOPO_VSRC_LOOP_INCONSISTENT",
        Severity.ERROR,
        SolverStage.PREFLIGHT,
        "adjust ideal voltage-source values or topology to remove contradictory loops",
    ),
    _entry(
        "E_TOPO_HARD_CONSTRAINT_CONFLICT",
        Severity.ERROR,
        SolverStage.PREFLIGHT,
        "remove or reconcile contradictory hard constraints",
    ),
    _entry(
        "E_TOPO_RF_BOUNDARY_INCONSISTENT",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "resolve conflicting/unknown RF boundary declarations before solve",
    ),
    _entry(
        "E_NUM_RF_BOUNDARY_SINGULAR",
        Severity.ERROR,
        SolverStage.ASSEMBLE,
        "remove redundant RF voltage boundary constraints",
    ),
    _entry(
        "E_NUM_SOLVE_FAILED",
        Severity.ERROR,
        SolverStage.SOLVE,
        "inspect topology, conditioning, and solver trace for the affected point",
    ),
    _entry(
        "E_NUM_SINGULAR_MATRIX",
        Severity.ERROR,
        SolverStage.SOLVE,
        "remove singular constraints or improve conditioning before solve",
    ),
    _entry(
        "E_NUM_ZBLOCK_SINGULAR",
        Severity.ERROR,
        SolverStage.POSTPROCESS,
        "fix singular Z block or use direct extraction path",
    ),
    _entry(
        "E_NUM_ZBLOCK_ILL_CONDITIONED",
        Severity.ERROR,
        SolverStage.POSTPROCESS,
        "improve conditioning or use direct extraction path",
    ),
    _entry(
        "E_NUM_S_CONVERSION_SINGULAR",
        Severity.ERROR,
        SolverStage.POSTPROCESS,
        "adjust Z0 or improve Y/Z conditioning before S conversion",
    ),
    _entry(
        "E_NUM_IMPEDANCE_UNDEFINED",
        Severity.ERROR,
        SolverStage.POSTPROCESS,
        "inspect boundary constraints and conditioning for selected impedance ports",
    ),
    _entry(
        "W_NUM_COND_UNAVAILABLE",
        Severity.WARNING,
        SolverStage.SOLVE,
        "review condition-estimator support and backend metadata for this point",
    ),
    _entry(
        "W_NUM_ILL_CONDITIONED",
        Severity.WARNING,
        SolverStage.SOLVE,
        "inspect conditioning and retry controls for this point",
    ),
    _entry(
        "W_RF_RECIPROCITY",
        Severity.WARNING,
        SolverStage.POSTPROCESS,
        "inspect port orientation and reciprocal-assumption applicability",
    ),
    _entry(
        "W_RF_PASSIVITY",
        Severity.WARNING,
        SolverStage.POSTPROCESS,
        "inspect model passivity assumptions and reference impedance choices",
    ),
    _entry(
        "E_CLI_RF_METRIC_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "choose --rf from: y,z,s,zin,zout",
    ),
    _entry(
        "E_CLI_RF_OPTIONS_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "provide compatible --rf options and deterministic RF port mapping",
    ),
    _entry(
        "E_CLI_CHECK_LOADER_FAILED",
        Severity.ERROR,
        SolverStage.PARSE,
        "integrate a deterministic design loader and retry check",
    ),
    _entry(
        "E_CLI_CHECK_INTERNAL",
        Severity.ERROR,
        SolverStage.PARSE,
        "inspect check-command integration and retry",
    ),
    _entry(
        "E_CLI_DESIGN_READ_FAILED",
        Severity.ERROR,
        SolverStage.PARSE,
        "provide a readable JSON design bundle path",
    ),
    _entry(
        "E_CLI_DESIGN_PARSE_FAILED",
        Severity.ERROR,
        SolverStage.PARSE,
        "fix JSON syntax and retry",
    ),
    _entry(
        "E_CLI_DESIGN_SCHEMA_UNSUPPORTED",
        Severity.ERROR,
        SolverStage.PARSE,
        "set schema to docs/spec/schemas/design_bundle_v1.json and schema_version to 1",
    ),
    _entry(
        "E_CLI_DESIGN_SCHEMA_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "fix the design bundle shape to match design_bundle_v1.json",
    ),
    _entry(
        "E_CLI_DESIGN_VALUE_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "fix the design bundle model fields and retry",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_DEFINITION",
        Severity.ERROR,
        SolverStage.PARSE,
        "rename the duplicate hierarchy definitions so each normalized id is unique",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_DEFINITION_CONFLICT",
        Severity.ERROR,
        SolverStage.PARSE,
        "rename the conflicting macro/subcircuit definitions so each normalized id resolves unambiguously",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_INSTANCE_ID",
        Severity.ERROR,
        SolverStage.PARSE,
        "rename the duplicate hierarchy instance ids so each scope-local id is unique",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_DUPLICATE_LOCAL_ELEMENT_ID",
        Severity.ERROR,
        SolverStage.PARSE,
        "rename the duplicate subcircuit-local element ids so each scope-local id is unique",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_DECLARATION_ILLEGAL",
        Severity.ERROR,
        SolverStage.PARSE,
        "rename or disambiguate illegal hierarchy declaration names so each normalized declaration name is unique within its declared scope",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_INSTANCE_ARITY_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "match the instance nodes list to the referenced macro node_formals or subcircuit ports arity",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_UNDEFINED",
        Severity.ERROR,
        SolverStage.PARSE,
        "declare the referenced hierarchy definition or fix the instance target id/type",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_REFERENCE_TYPE_MISMATCH",
        Severity.ERROR,
        SolverStage.PARSE,
        "align the instance_type with the referenced definition kind or retarget the instance",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_RECURSION_ILLEGAL",
        Severity.ERROR,
        SolverStage.PARSE,
        "remove the recursive subcircuit reference cycle before elaboration",
    ),
    _entry(
        "E_CLI_DESIGN_HIERARCHY_UNSUPPORTED",
        Severity.ERROR,
        SolverStage.PARSE,
        "remove design.instances for now or wait for P3-03 hierarchy elaboration support",
    ),
    _entry(
        "E_CLI_DESIGN_EXCLUSION_POLICY_INVALID",
        Severity.ERROR,
        SolverStage.PARSE,
        "fix docs/dev/p3_loader_temporary_exclusions.yaml to match the active interim loader exclusions",
    ),
    _entry(
        "E_CLI_DESIGN_EXCLUDED_CAPABILITY",
        Severity.ERROR,
        SolverStage.PARSE,
        "remove the excluded capability usage or wait for Phase 3 closure support",
    ),
)


CANONICAL_DIAGNOSTIC_CATALOG: Mapping[str, DiagnosticCatalogEntry] = _build_catalog(
    _CATALOG_ENTRIES
)

REQUIRED_CATALOG_FIELDS: tuple[str, ...] = ("code", "severity", "solver_stage", "suggested_action")

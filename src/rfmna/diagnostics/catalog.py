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
) -> tuple[str, DiagnosticCatalogEntry]:
    return (
        code,
        DiagnosticCatalogEntry(
            code=code,
            severity=severity,
            solver_stage=solver_stage,
            suggested_action=suggested_action,
        ),
    )


CANONICAL_DIAGNOSTIC_CATALOG: Mapping[str, DiagnosticCatalogEntry] = MappingProxyType(
    dict(
        (
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
        )
    )
)

REQUIRED_CATALOG_FIELDS: tuple[str, ...] = ("code", "severity", "solver_stage", "suggested_action")

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from rfmna.assembler import AssemblerError, build_unknown_indexing, compile_pattern, fill_numeric
from rfmna.diagnostics import (
    DiagnosticEvent,
    SolverStage,
    adapt_validation_issue,
    build_diagnostic_event,
    sort_diagnostics,
)
from rfmna.elements import StampContext
from rfmna.elements.factory import FactoryConstructionError, build_stamps_from_canonical_ir
from rfmna.elements.factory_models import FactoryModelNormalizationError
from rfmna.parser.design_bundle import (
    DesignBundleLoadError,
    ParsedDesignBundle,
    load_design_bundle_document,
)
from rfmna.parser.preflight import PreflightInput
from rfmna.rf_metrics import PortBoundary
from rfmna.sweep_engine import AssemblePointFn, SweepLayout

_LOADER_CONTEXT = "cli.design_loader"


@dataclass(frozen=True, slots=True)
class CliDesignBundle:
    preflight_input: PreflightInput
    frequencies_hz: Sequence[float] | NDArray[np.float64]
    sweep_layout: SweepLayout
    assemble_point: AssemblePointFn
    rf_ports: tuple[PortBoundary, ...] = ()
    rf_z0_ohm: float | tuple[float, ...] = 50.0
    manifest_input_payload: dict[str, object] = field(default_factory=dict)
    manifest_resolved_params_payload: dict[str, object] = field(default_factory=dict)


def load_design_bundle(design: str) -> CliDesignBundle:
    parsed = load_design_bundle_document(design)
    return _compile_cli_bundle(parsed)


def _compile_cli_bundle(parsed: ParsedDesignBundle) -> CliDesignBundle:
    try:
        indexing = build_unknown_indexing(
            node_ids=tuple(node.node_id for node in parsed.ir.nodes),
            reference_node=_reference_node(parsed),
            aux_ids=tuple(aux.aux_id for aux in parsed.ir.aux_unknowns),
        )
        stamps = build_stamps_from_canonical_ir(parsed.ir, indexing=indexing)
    except FactoryModelNormalizationError as exc:
        raise DesignBundleLoadError(
            (
                _model_invalid_diagnostic(
                    message=exc.detail.message,
                    element_id=exc.detail.element_id,
                    witness=dict(exc.detail.witness),
                ),
            )
        ) from exc
    except ValueError as exc:
        raise DesignBundleLoadError(
            (
                _model_invalid_diagnostic(
                    message=str(exc),
                    witness={"stage": "unknown_indexing"},
                ),
            )
        ) from exc
    except FactoryConstructionError as exc:
        raise DesignBundleLoadError(
            (
                _model_invalid_diagnostic(
                    message=exc.detail.message,
                    element_id=exc.detail.element_id,
                    witness=dict(exc.detail.witness),
                ),
            )
        ) from exc

    validation_ctx = StampContext(
        omega_rad_s=0.0,
        resolved_params=parsed.resolved_parameters.as_dict(),
        frequency_index=0,
    )
    validation_diagnostics = [
        adapt_validation_issue(issue, element_id=stamp.element_id)
        for stamp in stamps
        for issue in stamp.validate(validation_ctx)
    ]
    if validation_diagnostics:
        raise DesignBundleLoadError(tuple(sort_diagnostics(validation_diagnostics)))

    try:
        pattern = compile_pattern(indexing.total_unknowns, stamps, validation_ctx)
    except AssemblerError as exc:
        raise DesignBundleLoadError(
            (
                _model_invalid_diagnostic(
                    message=exc.message,
                    witness={"assembler_code": exc.code, "stage": "compile_pattern"},
                ),
            )
        ) from exc

    rf_ports = tuple(
        PortBoundary(
            port_id=port.port_id,
            p_plus_index=indexing.node_index(port.p_plus),
            p_minus_index=indexing.node_index(port.p_minus),
        )
        for port in parsed.rf_ports
    )

    def assemble_point(point_index: int, frequency_hz: float) -> tuple[csc_matrix, NDArray[np.complex128]]:
        ctx = StampContext(
            omega_rad_s=2.0 * np.pi * float(frequency_hz),
            resolved_params=parsed.resolved_parameters.as_dict(),
            frequency_index=point_index,
        )
        filled = fill_numeric(pattern, stamps, ctx)
        return (
            filled.A.tocsc(),
            np.asarray(filled.b, dtype=np.complex128),
        )

    return CliDesignBundle(
        preflight_input=parsed.preflight_input,
        frequencies_hz=np.asarray(parsed.frequencies_hz, dtype=np.float64),
        sweep_layout=SweepLayout(n_nodes=indexing.n_nodes, n_aux=indexing.n_aux),
        assemble_point=assemble_point,
        rf_ports=rf_ports,
        rf_z0_ohm=parsed.rf_z0_ohm,
        manifest_input_payload=dict(parsed.manifest_input_payload),
        manifest_resolved_params_payload=dict(parsed.manifest_resolved_params_payload),
    )


def _reference_node(parsed: ParsedDesignBundle) -> str:
    for node in parsed.ir.nodes:
        if node.is_reference:
            return node.node_id
    raise DesignBundleLoadError(
        (
            _model_invalid_diagnostic(
                message="design bundle canonical model is missing a reference node",
                witness={"stage": "reference_node_resolution"},
            ),
        )
    )


def _model_invalid_diagnostic(
    *,
    message: str,
    witness: dict[str, object],
    element_id: str | None = None,
) -> DiagnosticEvent:
    return build_diagnostic_event(
        code="E_CLI_DESIGN_VALUE_INVALID",
        message=f"design bundle model validation failed: {message}",
        suggested_action="fix the design bundle model fields and retry",
        solver_stage=SolverStage.PARSE,
        element_id=_LOADER_CONTEXT if element_id is None else element_id,
        witness=witness,
    )

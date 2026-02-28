from __future__ import annotations

import pytest

import rfmna.parser as parser_api
import rfmna.solver as solver_api
import rfmna.viz_io as viz_io_api

pytestmark = pytest.mark.unit


def test_parser_public_api_surface_is_explicit_and_stable() -> None:
    assert parser_api.__all__ == [
        "HardConstraint",
        "IdealVSource",
        "ParseError",
        "ParseErrorCode",
        "ParseErrorDetail",
        "PortDecl",
        "PreflightInput",
        "ResolvedParameters",
        "evaluate_expression",
        "extract_dependencies",
        "parse_frequency_unit",
        "parse_scalar_number",
        "preflight_check",
        "resolve_parameters",
    ]
    assert not hasattr(parser_api, "FREQUENCY_UNIT_SCALE")
    assert not hasattr(parser_api, "VSRC_LOOP_RESIDUAL_ABS_TOL")


def test_solver_public_api_surface_is_explicit_and_stable() -> None:
    assert solver_api.__all__ == [
        "AttemptTraceRecord",
        "BackendMetadata",
        "BackendNotes",
        "BackendSolveOptions",
        "BackendSolveResult",
        "DEFAULT_THRESHOLDS_PATH",
        "DEGRADED_MAX",
        "EPSILON",
        "FallbackRunConfig",
        "PASS_MAX",
        "ResidualMetrics",
        "SolveWarning",
        "SciPySparseBackend",
        "SolverConfigError",
        "SolveResult",
        "SolveStatus",
        "SolverBackend",
        "classify_status",
        "compute_residual_metrics",
        "load_solver_threshold_config",
        "solve_linear_system",
    ]
    assert not hasattr(solver_api, "ConditionEstimator")
    assert not hasattr(solver_api, "run_fallback_ladder")
    assert not hasattr(solver_api, "apply_gmin_shunt")


def test_viz_io_public_api_surface_is_explicit_and_stable() -> None:
    assert viz_io_api.__all__ == [
        "ManifestError",
        "RunArtifactWithManifest",
        "RunManifest",
        "attach_manifest_to_run_payload",
        "build_manifest",
        "stable_manifest_hash",
        "stable_projection",
        "to_canonical_dict",
        "to_canonical_json",
    ]
    assert not hasattr(viz_io_api, "CORE_DEPENDENCIES")
    assert not hasattr(viz_io_api, "THREAD_ENV_KEYS")
    assert not hasattr(viz_io_api, "VOLATILE_STABLE_EXCLUDE_KEYS")

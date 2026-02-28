from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from rfmna.solver import (
    DEFAULT_THRESHOLDS_PATH,
    BackendMetadata,
    BackendNotes,
    BackendSolveOptions,
    BackendSolveResult,
    FallbackRunConfig,
    SolverBackend,
    SolverConfigError,
    load_solver_threshold_config,
    solve_linear_system,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _PlannedResponse:
    success: bool


@dataclass(frozen=True, slots=True)
class _BackendCall:
    matrix_dense: NDArray[np.complex128]
    vector: NDArray[np.complex128]
    options: BackendSolveOptions


class _PlannedBackend(SolverBackend):
    def __init__(self, responses: tuple[_PlannedResponse, ...]) -> None:
        self._responses = responses
        self.calls: list[_BackendCall] = []

    def solve(
        self,
        A: object,
        b: NDArray[np.complex128],
        *,
        options: BackendSolveOptions | None = None,
    ) -> BackendSolveResult:
        idx = len(self.calls)
        response = (
            self._responses[idx] if idx < len(self._responses) else _PlannedResponse(success=False)
        )
        selected_options = options if options is not None else BackendSolveOptions()
        sparse_matrix = csr_matrix(A)
        self.calls.append(
            _BackendCall(
                matrix_dense=np.asarray(sparse_matrix.toarray(), dtype=np.complex128),
                vector=np.asarray(b, dtype=np.complex128).copy(),
                options=selected_options,
            )
        )

        metadata = BackendMetadata(
            backend_id="planned",
            backend_name="planned",
            python_version=None,
            scipy_version=None,
            permutation_row=None,
            permutation_col=None,
            pivot_order=None,
            success=response.success,
            failure_category=None if response.success else "numeric",
            failure_code=None if response.success else "E_NUM_SOLVE_FAILED",
            failure_reason=None if response.success else "planned_failure",
            failure_message=None if response.success else "planned failure",
            notes=BackendNotes(
                input_format="csr",
                normalized_format="csr",
                n_unknowns=int(b.shape[0]),
                nnz=int(sparse_matrix.nnz),
                pivot_profile=selected_options.pivot_profile,
                scaling_enabled=selected_options.scaling_enabled,
            ),
        )
        if response.success:
            return BackendSolveResult(
                x=np.asarray(b, dtype=np.complex128).copy(), metadata=metadata
            )
        return BackendSolveResult(x=None, metadata=metadata)


@dataclass(frozen=True, slots=True)
class _ConstantEstimator:
    value: float | None

    def estimate(self, A: object) -> float | None:
        del A
        return self.value


def _identity_system(size: int) -> tuple[csr_matrix, NDArray[np.complex128]]:
    matrix = csr_matrix(np.eye(size, dtype=np.complex128))
    vector = np.asarray([complex(i + 1, 0.0) for i in range(size)], dtype=np.complex128)
    return matrix, vector


def _expected_stage_sequence(gmin_count: int) -> list[str]:
    return ["baseline", "alt_pivot", "scaling"] + (["gmin"] * gmin_count) + ["final_fail"]


def _write_thresholds_with_order(tmp_path: Path, order_tokens: list[str]) -> Path:
    payload = yaml.safe_load(Path(DEFAULT_THRESHOLDS_PATH).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["solver_defaults"]["fallback_ladder"]["order"] = order_tokens
    output_path = tmp_path / "thresholds_order.yaml"
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _write_thresholds_with_fail_min_exclusive(tmp_path: Path, fail_min_exclusive: float) -> Path:
    payload = yaml.safe_load(Path(DEFAULT_THRESHOLDS_PATH).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["numeric_contract"]["residual"]["status_bands"]["fail_min_exclusive"] = (
        fail_min_exclusive
    )
    output_path = tmp_path / "thresholds_fail_min.yaml"
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def test_exact_ladder_order_and_no_hidden_retries() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(2)
    run_attempts = 3 + len(defaults.gmin_values)
    backend = _PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts)))

    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.5),
    )

    assert [row.stage for row in result.attempt_trace] == _expected_stage_sequence(
        len(defaults.gmin_values)
    )
    assert [row.attempt_index for row in result.attempt_trace] == list(
        range(len(result.attempt_trace))
    )
    assert all(row.stage_state == "run" for row in result.attempt_trace[:-1])
    assert result.attempt_trace[-1].stage == "final_fail"
    assert result.attempt_trace[-1].stage_state == "run"
    assert len(backend.calls) == run_attempts
    assert result.status == "fail"


@pytest.mark.parametrize(
    ("success_call_index", "expected_stage"),
    [
        (0, "baseline"),
        (1, "alt_pivot"),
        (2, "scaling"),
        (3, "gmin"),
    ],
)
def test_short_circuit_stops_on_first_success(success_call_index: int, expected_stage: str) -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(2)
    run_attempts = 3 + len(defaults.gmin_values)
    responses = tuple(
        _PlannedResponse(success=(idx == success_call_index)) for idx in range(run_attempts)
    )
    backend = _PlannedBackend(responses)

    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.5),
    )

    assert result.status == "pass"
    assert len(backend.calls) == success_call_index + 1
    success_rows = [row for row in result.attempt_trace if row.success]
    assert len(success_rows) == 1
    assert success_rows[0].stage == expected_stage
    for row in result.attempt_trace[success_call_index + 1 :]:
        assert row.stage_state == "skipped"
    assert result.attempt_trace[-1].stage == "final_fail"
    assert result.attempt_trace[-1].stage_state == "skipped"


def test_full_failure_has_final_fail_marker() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)
    run_attempts = 3 + len(defaults.gmin_values)
    backend = _PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts)))

    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.4),
    )

    assert result.status == "fail"
    assert result.attempt_trace[-1].stage == "final_fail"
    assert result.attempt_trace[-1].stage_state == "run"


def test_gmin_semantics_order_node_only_shunt_and_no_carryover() -> None:
    defaults = load_solver_threshold_config()
    base_dense = np.asarray(
        [
            [10.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 20.0 + 0.0j, 2.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.0j, 30.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    A = csr_matrix(base_dense)
    b = np.asarray([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128)
    run_attempts = 3 + len(defaults.gmin_values)
    backend = _PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts)))

    _ = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=2,
        condition_estimator=_ConstantEstimator(0.3),
    )

    gmin_calls = backend.calls[3:]
    assert len(gmin_calls) == len(defaults.gmin_values)
    for idx, gmin in enumerate(defaults.gmin_values):
        expected = base_dense.copy()
        expected[0, 0] += gmin
        expected[1, 1] += gmin
        assert np.allclose(gmin_calls[idx].matrix_dense, expected)
        assert gmin_calls[idx].matrix_dense[2, 2] == pytest.approx(30.0 + 0.0j)


def test_per_point_reset_restarts_ladder_and_gmin_sequence() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(2)
    run_attempts = 3 + len(defaults.gmin_values)

    left = solve_linear_system(
        A,
        b,
        backend=_PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts))),
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.2),
    )
    right = solve_linear_system(
        A,
        b,
        backend=_PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts))),
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.2),
    )

    left_gmin = [row.gmin_value for row in left.attempt_trace if row.stage == "gmin"]
    right_gmin = [row.gmin_value for row in right.attempt_trace if row.stage == "gmin"]
    assert left.attempt_trace[0].attempt_index == 0
    assert right.attempt_trace[0].attempt_index == 0
    assert left_gmin == list(defaults.gmin_values)
    assert right_gmin == list(defaults.gmin_values)


def test_condition_indicator_available_and_unavailable_contract() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)
    backend = _PlannedBackend((_PlannedResponse(True),))

    available = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.125),
    )
    unavailable = solve_linear_system(
        A,
        b,
        backend=_PlannedBackend((_PlannedResponse(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(None),
    )

    assert available.cond_ind == pytest.approx(0.125)
    assert available.warnings == ()
    assert math.isnan(unavailable.cond_ind)
    assert len(unavailable.warnings) == 1
    assert unavailable.warnings[0].code == defaults.conditioning.unavailable_warning_code


def test_condition_indicator_warn_band_emits_warning_without_forcing_fail() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)
    warn_value = (defaults.conditioning.warn_max + defaults.conditioning.fail_max) * 0.5
    assert warn_value > defaults.conditioning.fail_max
    assert warn_value <= defaults.conditioning.warn_max

    result = solve_linear_system(
        A,
        b,
        backend=_PlannedBackend((_PlannedResponse(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(warn_value),
    )

    assert result.status == "pass"
    assert result.failure_code is None
    assert len(result.warnings) == 1
    assert result.warnings[0].code == defaults.conditioning.ill_conditioned_warning_code


def test_condition_indicator_fail_band_forces_numeric_failure_classification() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)

    result = solve_linear_system(
        A,
        b,
        backend=_PlannedBackend((_PlannedResponse(True),)),
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(defaults.conditioning.fail_max),
    )

    assert result.status == "fail"
    assert result.failure_code == "E_NUM_SOLVE_FAILED"
    assert result.failure_reason == "condition_indicator_below_fail_threshold"
    assert isinstance(result.failure_message, str)
    assert "<= fail_max" in result.failure_message
    assert result.warnings == ()


def test_artifact_sourcing_and_config_errors(tmp_path: Path) -> None:
    defaults = load_solver_threshold_config()
    assert defaults.residual.epsilon == pytest.approx(1.0e-30)
    assert defaults.residual.pass_max == pytest.approx(1.0e-9)
    assert defaults.residual.degraded_max == pytest.approx(1.0e-6)
    assert defaults.residual.fail_min_exclusive == pytest.approx(1.0e-6)
    assert defaults.gmin_values[0] == pytest.approx(0.0)
    assert defaults.fallback_order == ("baseline", "alt_pivot", "scaling", "gmin", "final_fail")

    with pytest.raises(SolverConfigError) as missing:
        _ = load_solver_threshold_config(tmp_path / "missing_thresholds.yaml")
    assert missing.value.code == "E_SOLVER_CONFIG_READ_FAILED"

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("not: [valid", encoding="utf-8")
    with pytest.raises(SolverConfigError) as invalid:
        _ = load_solver_threshold_config(invalid_path)
    assert invalid.value.code == "E_SOLVER_CONFIG_PARSE_FAILED"

    incomplete_path = tmp_path / "incomplete.yaml"
    incomplete_path.write_text("version: 4.0.0\n", encoding="utf-8")
    with pytest.raises(SolverConfigError) as incomplete:
        _ = load_solver_threshold_config(incomplete_path)
    assert incomplete.value.code == "E_SOLVER_CONFIG_INVALID"


def test_invalid_fail_min_exclusive_in_artifact_raises_config_error(tmp_path: Path) -> None:
    invalid_path = _write_thresholds_with_fail_min_exclusive(tmp_path, fail_min_exclusive=1.0e-5)
    with pytest.raises(SolverConfigError) as exc_info:
        _ = load_solver_threshold_config(invalid_path)
    assert exc_info.value.code == "E_SOLVER_CONFIG_INVALID"


def test_fallback_order_is_consumed_from_artifact_and_drives_attempt_sequence(
    tmp_path: Path,
) -> None:
    A, b = _identity_system(1)
    reordered_path = _write_thresholds_with_order(
        tmp_path,
        [
            "baseline",
            "scaling_enabled_retry",
            "alt_permutation_or_pivot",
            "gmin_ladder_retry",
            "final_fail_bundle",
        ],
    )
    thresholds = load_solver_threshold_config(reordered_path)
    assert thresholds.fallback_order == ("baseline", "scaling", "alt_pivot", "gmin", "final_fail")

    run_attempts = 3 + len(thresholds.gmin_values)
    backend = _PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts)))
    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.5),
        thresholds=thresholds,
    )

    expected = (
        ["baseline", "scaling", "alt_pivot"]
        + (["gmin"] * len(thresholds.gmin_values))
        + ["final_fail"]
    )
    assert [row.stage for row in result.attempt_trace] == expected


def test_invalid_fallback_order_in_artifact_raises_config_error(tmp_path: Path) -> None:
    invalid_path = _write_thresholds_with_order(
        tmp_path,
        [
            "baseline",
            "alt_permutation_or_pivot",
            "scaling_enabled_retry",
            "gmin_ladder_retry",
            "baseline",
        ],
    )
    with pytest.raises(SolverConfigError) as exc_info:
        _ = load_solver_threshold_config(invalid_path)
    assert exc_info.value.code == "E_SOLVER_CONFIG_INVALID"


def test_policy_a_continuity_on_unrecovered_numeric_failure() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)
    run_attempts = 3 + len(defaults.gmin_values)
    backend = _PlannedBackend(tuple(_PlannedResponse(False) for _ in range(run_attempts)))

    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.1),
    )

    assert result.status == "fail"
    assert result.x is None
    assert math.isnan(result.residual.res_rel)
    assert result.failure_code == "E_NUM_SOLVE_FAILED"


def test_orchestration_purity_inputs_unchanged() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(2)
    A_before = A.copy()
    b_before = b.copy()
    run_config = FallbackRunConfig(
        enable_alt_pivot=True,
        enable_scaling=True,
        enable_gmin=True,
    )
    backend = _PlannedBackend(
        tuple(_PlannedResponse(False) for _ in range(3 + len(defaults.gmin_values)))
    )

    _ = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        run_config=run_config,
        condition_estimator=_ConstantEstimator(0.2),
    )

    assert (A_before != A).nnz == 0
    np.testing.assert_array_equal(b_before, b)
    assert run_config == FallbackRunConfig(
        enable_alt_pivot=True, enable_scaling=True, enable_gmin=True
    )


def test_attempt_trace_schema_fields_and_types() -> None:
    defaults = load_solver_threshold_config()
    A, b = _identity_system(1)
    backend = _PlannedBackend(
        tuple(_PlannedResponse(False) for _ in range(3 + len(defaults.gmin_values)))
    )
    result = solve_linear_system(
        A,
        b,
        backend=backend,
        node_voltage_count=1,
        condition_estimator=_ConstantEstimator(0.25),
    )

    row = result.attempt_trace[0]
    row_payload = asdict(row)
    assert tuple(row_payload) == (
        "attempt_index",
        "stage",
        "stage_state",
        "scaling_enabled",
        "pivot_profile",
        "gmin_value",
        "success",
        "backend_failure_category",
        "backend_failure_reason",
        "backend_failure_code",
        "backend_failure_message",
        "res_l2",
        "res_linf",
        "res_rel",
        "status",
        "skip_reason",
    )
    assert isinstance(row.attempt_index, int)
    assert isinstance(row.stage, str)
    assert isinstance(row.stage_state, str)
    assert isinstance(row.scaling_enabled, bool)

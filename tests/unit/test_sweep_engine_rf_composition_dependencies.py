from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

import rfmna.sweep_engine.run as sweep_run
from rfmna.diagnostics import DiagnosticEvent, Severity, SolverStage
from rfmna.rf_metrics import PortBoundary, SParameterResult, YParameterResult, ZParameterResult
from rfmna.rf_metrics.impedance import ZinZoutResult
from rfmna.sweep_engine import SweepRFRequest

pytestmark = pytest.mark.unit

_FREQUENCIES = np.asarray((1.0, 2.0), dtype=np.float64)
_PORTS = (
    PortBoundary(port_id="P2", p_plus_index=1, p_minus_index=None),
    PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),
)


def _assemble_point(_: int, __: float) -> tuple[csc_matrix, np.ndarray]:
    matrix = csc_matrix(np.eye(2, dtype=np.complex128))
    rhs = np.zeros(2, dtype=np.complex128)
    return matrix, rhs


def _y_result() -> YParameterResult:
    return YParameterResult(
        frequencies_hz=_FREQUENCIES,
        port_ids=("P1", "P2"),
        y=np.asarray(
            [
                [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
                [[5.0 + 0.0j, 6.0 + 0.0j], [7.0 + 0.0j, 8.0 + 0.0j]],
            ],
            dtype=np.complex128,
        ),
        status=np.asarray(("pass", "pass"), dtype=np.dtype("<U8")),
        diagnostics_by_point=((), ()),
    )


def _z_result(*, statuses: Sequence[str] = ("pass", "pass")) -> ZParameterResult:
    return ZParameterResult(
        frequencies_hz=_FREQUENCIES,
        port_ids=("P1", "P2"),
        z=np.asarray(
            [
                [[11.0 + 0.0j, 12.0 + 0.0j], [13.0 + 0.0j, 14.0 + 0.0j]],
                [[15.0 + 0.0j, 16.0 + 0.0j], [17.0 + 0.0j, 18.0 + 0.0j]],
            ],
            dtype=np.complex128,
        ),
        status=np.asarray(tuple(statuses), dtype=np.dtype("<U8")),
        diagnostics_by_point=((), ()),
        extraction_mode="direct",
    )


def _s_result() -> SParameterResult:
    return SParameterResult(
        frequencies_hz=_FREQUENCIES,
        port_ids=("P1", "P2"),
        s=np.asarray(
            [
                [[0.1 + 0.01j, 0.2 + 0.02j], [0.3 + 0.03j, 0.4 + 0.04j]],
                [[0.5 + 0.05j, 0.6 + 0.06j], [0.7 + 0.07j, 0.8 + 0.08j]],
            ],
            dtype=np.complex128,
        ),
        status=np.asarray(("pass", "pass"), dtype=np.dtype("<U8")),
        diagnostics_by_point=((), ()),
        conversion_source="from_z",
    )


def _impedance_result() -> ZinZoutResult:
    return ZinZoutResult(
        frequencies_hz=_FREQUENCIES,
        port_ids=("P1", "P2"),
        input_port_id="P1",
        output_port_id="P2",
        zin=np.asarray((51.0 + 1.0j, 52.0 + 2.0j), dtype=np.complex128),
        zout=np.asarray((61.0 + 1.0j, 62.0 + 2.0j), dtype=np.complex128),
        status=np.asarray(("pass", "pass"), dtype=np.dtype("<U8")),
        diagnostics_by_point=((), ()),
    )


def _request(*, metrics: Sequence[str]) -> SweepRFRequest:
    return SweepRFRequest(ports=_PORTS, metrics=tuple(metrics))


def _unexpected(name: str) -> None:
    pytest.fail(f"unexpected RF dependency path: {name}")


@pytest.mark.parametrize(
    ("requested_metrics", "expected_metric_names", "expected_calls"),
    [
        (("y",), ("y",), ("extract_y",)),
        (("z",), ("z",), ("extract_z:direct",)),
        (("s",), ("s",), ("extract_z:direct", "convert_z_to_s")),
        (("zin",), ("zin",), ("extract_zin_zout",)),
        (("zout",), ("zout",), ("extract_zin_zout",)),
    ],
)
def test_rf_composition_matrix_rows_use_explicit_dependency_paths(
    monkeypatch: pytest.MonkeyPatch,
    requested_metrics: tuple[str, ...],
    expected_metric_names: tuple[str, ...],
    expected_calls: tuple[str, ...],
) -> None:
    calls: list[str] = []

    def fake_extract_y(*args: object, **kwargs: object) -> YParameterResult:
        del args, kwargs
        calls.append("extract_y")
        return _y_result()

    def fake_extract_z(*args: object, **kwargs: object) -> ZParameterResult:
        del args
        calls.append(f"extract_z:{kwargs.get('extraction_mode')}")
        return _z_result()

    def fake_convert_z_to_s(*args: object, **kwargs: object) -> SParameterResult:
        del args, kwargs
        calls.append("convert_z_to_s")
        return _s_result()

    def fake_extract_zin_zout(*args: object, **kwargs: object) -> ZinZoutResult:
        del args, kwargs
        calls.append("extract_zin_zout")
        return _impedance_result()

    monkeypatch.setattr(sweep_run, "extract_y_parameters", fake_extract_y)
    monkeypatch.setattr(sweep_run, "extract_z_parameters", fake_extract_z)
    monkeypatch.setattr(sweep_run, "convert_z_to_s", fake_convert_z_to_s)
    monkeypatch.setattr(sweep_run, "extract_zin_zout", fake_extract_zin_zout)
    monkeypatch.setattr(
        sweep_run, "convert_y_to_s", lambda *args, **kwargs: _unexpected("convert_y_to_s")
    )

    payloads = sweep_run._compute_rf_payloads(
        frequencies=_FREQUENCIES,
        rf_request=_request(metrics=requested_metrics),
        assemble_point=_assemble_point,
        solve_point=None,
    )

    assert payloads.metric_names == expected_metric_names
    assert tuple(calls) == expected_calls


def test_rf_composition_canonical_execution_order_is_request_order_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_for_request(metrics: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        calls: list[str] = []

        def fake_extract_y(*args: object, **kwargs: object) -> YParameterResult:
            del args, kwargs
            calls.append("extract_y")
            return _y_result()

        def fake_extract_z(*args: object, **kwargs: object) -> ZParameterResult:
            del args
            calls.append(f"extract_z:{kwargs.get('extraction_mode')}")
            return _z_result()

        def fake_convert_z_to_s(*args: object, **kwargs: object) -> SParameterResult:
            del args, kwargs
            calls.append("convert_z_to_s")
            return _s_result()

        def fake_extract_zin_zout(*args: object, **kwargs: object) -> ZinZoutResult:
            del args, kwargs
            calls.append("extract_zin_zout")
            return _impedance_result()

        monkeypatch.setattr(sweep_run, "extract_y_parameters", fake_extract_y)
        monkeypatch.setattr(sweep_run, "extract_z_parameters", fake_extract_z)
        monkeypatch.setattr(sweep_run, "convert_z_to_s", fake_convert_z_to_s)
        monkeypatch.setattr(sweep_run, "extract_zin_zout", fake_extract_zin_zout)
        monkeypatch.setattr(
            sweep_run, "convert_y_to_s", lambda *args, **kwargs: _unexpected("convert_y_to_s")
        )

        payloads = sweep_run._compute_rf_payloads(
            frequencies=_FREQUENCIES,
            rf_request=_request(metrics=metrics),
            assemble_point=_assemble_point,
            solve_point=None,
        )
        return payloads.metric_names, tuple(calls)

    first_metrics, first_calls = run_for_request(("zout", "s", "y", "zin", "z"))
    second_metrics, second_calls = run_for_request(("y", "zin", "zout", "s", "z"))

    expected_metrics = ("y", "z", "s", "zin", "zout")
    expected_calls = ("extract_y", "extract_z:direct", "convert_z_to_s", "extract_zin_zout")
    assert first_metrics == expected_metrics
    assert second_metrics == expected_metrics
    assert first_calls == expected_calls
    assert second_calls == expected_calls


def test_zin_and_zout_share_single_impedance_extraction_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_extract_zin_zout(*args: object, **kwargs: object) -> ZinZoutResult:
        del args, kwargs
        calls.append("extract_zin_zout")
        return _impedance_result()

    monkeypatch.setattr(sweep_run, "extract_zin_zout", fake_extract_zin_zout)
    monkeypatch.setattr(
        sweep_run, "extract_y_parameters", lambda *args, **kwargs: _unexpected("extract_y")
    )
    monkeypatch.setattr(
        sweep_run, "extract_z_parameters", lambda *args, **kwargs: _unexpected("extract_z")
    )
    monkeypatch.setattr(
        sweep_run, "convert_z_to_s", lambda *args, **kwargs: _unexpected("convert_z_to_s")
    )
    monkeypatch.setattr(
        sweep_run, "convert_y_to_s", lambda *args, **kwargs: _unexpected("convert_y_to_s")
    )

    payloads = sweep_run._compute_rf_payloads(
        frequencies=_FREQUENCIES,
        rf_request=_request(metrics=("zout", "zin")),
        assemble_point=_assemble_point,
        solve_point=None,
    )

    assert payloads.metric_names == ("zin", "zout")
    assert calls == ["extract_zin_zout"]


def test_s_from_z_dependency_failure_propagates_to_sentinel_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail_diagnostic = DiagnosticEvent(
        code="E_NUM_SOLVE_FAILED",
        severity=Severity.ERROR,
        message="upstream z extraction failure",
        suggested_action="inspect z extraction diagnostics",
        solver_stage=SolverStage.SOLVE,
        element_id="rf_z_params",
        frequency_hz=2.0,
        frequency_index=1,
        witness={"source": "z_dependency"},
    )
    fail_z = ZParameterResult(
        frequencies_hz=np.asarray((1.0, 2.0), dtype=np.float64),
        port_ids=("P1",),
        z=np.asarray(
            [
                [[50.0 + 0.0j]],
                [[np.complex128(complex(float("nan"), float("nan")))]],
            ],
            dtype=np.complex128,
        ),
        status=np.asarray(("pass", "fail"), dtype=np.dtype("<U8")),
        diagnostics_by_point=((), (fail_diagnostic,)),
        extraction_mode="direct",
    )
    calls: list[str] = []

    def fake_extract_z(*args: object, **kwargs: object) -> ZParameterResult:
        del args
        calls.append(f"extract_z:{kwargs.get('extraction_mode')}")
        return fail_z

    monkeypatch.setattr(sweep_run, "extract_z_parameters", fake_extract_z)
    monkeypatch.setattr(
        sweep_run, "extract_y_parameters", lambda *args, **kwargs: _unexpected("extract_y")
    )

    payloads = sweep_run._compute_rf_payloads(
        frequencies=np.asarray((1.0, 2.0), dtype=np.float64),
        rf_request=SweepRFRequest(
            ports=(PortBoundary(port_id="P1", p_plus_index=0, p_minus_index=None),),
            metrics=("s",),
        ),
        assemble_point=_assemble_point,
        solve_point=None,
    )

    s_payload = payloads.get("s")
    assert s_payload is not None
    assert isinstance(s_payload, SParameterResult)
    assert calls == ["extract_z:direct"]
    assert s_payload.conversion_source == "from_z"
    assert list(s_payload.status.astype(str)) == ["pass", "fail"]
    assert np.isnan(s_payload.s[1].real).all()
    assert np.isnan(s_payload.s[1].imag).all()
    assert [diag.code for diag in s_payload.diagnostics_by_point[1]] == ["E_NUM_SOLVE_FAILED"]

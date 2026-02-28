from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rfmna.ir import CanonicalIR, IRElement, IRNode, canonical_ir_json, hash_canonical_ir
from rfmna.rf_metrics import convert_y_to_s
from rfmna.rf_metrics.y_params import YParameterResult

pytestmark = pytest.mark.property

_POSITIVE_CONDUCTANCE = st.floats(
    min_value=1e-3, max_value=0.2, allow_nan=False, allow_infinity=False
)
_SUSCEPTANCE = st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)
_MUTUAL_SUSCEPTANCE = st.floats(
    min_value=-0.02,
    max_value=0.02,
    allow_nan=False,
    allow_infinity=False,
)
_PORT_Z0 = st.floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False)

_NODE_IDS = ("0", "n1", "n2", "n3")
_ELEMENT_DEFS: tuple[tuple[str, str, tuple[str, ...], tuple[tuple[str, float], ...]], ...] = (
    ("R1", "resistor", ("n1", "0"), (("resistance_ohm", 100.0),)),
    ("C1", "cap", ("n1", "n2"), (("capacitance_f", 1e-12),)),
    ("G1", "conductance", ("n2", "0"), (("conductance_s", 0.02),)),
    ("E1", "vcvs", ("n3", "0", "n1", "0"), (("offset_v", 0.0), ("gain_mu", 2.0))),
)
_RESOLVED_PARAMS = (("beta", 2.0), ("alpha", 1.0), ("gamma", 3.0))


@dataclass(frozen=True, slots=True)
class _ReciprocalYCase:
    g1: float
    g2: float
    g12: float
    b1: float
    b2: float
    b12: float
    z0: float


@dataclass(frozen=True, slots=True)
class _IRPermutationCase:
    left_node_order: tuple[int, ...]
    right_node_order: tuple[int, ...]
    left_element_order: tuple[int, ...]
    right_element_order: tuple[int, ...]
    left_param_order: tuple[int, ...]
    right_param_order: tuple[int, ...]
    left_reverse_e1: bool
    right_reverse_e1: bool


@st.composite
def _reciprocal_y_cases(draw: st.DrawFn) -> _ReciprocalYCase:
    return _ReciprocalYCase(
        g1=draw(_POSITIVE_CONDUCTANCE),
        g2=draw(_POSITIVE_CONDUCTANCE),
        g12=draw(st.floats(min_value=1e-5, max_value=0.05, allow_nan=False, allow_infinity=False)),
        b1=draw(_SUSCEPTANCE),
        b2=draw(_SUSCEPTANCE),
        b12=draw(_MUTUAL_SUSCEPTANCE),
        z0=draw(_PORT_Z0),
    )


@st.composite
def _ir_permutation_cases(draw: st.DrawFn) -> _IRPermutationCase:
    return _IRPermutationCase(
        left_node_order=draw(st.permutations(tuple(range(len(_NODE_IDS))))),
        right_node_order=draw(st.permutations(tuple(range(len(_NODE_IDS))))),
        left_element_order=draw(st.permutations(tuple(range(len(_ELEMENT_DEFS))))),
        right_element_order=draw(st.permutations(tuple(range(len(_ELEMENT_DEFS))))),
        left_param_order=draw(st.permutations(tuple(range(len(_RESOLVED_PARAMS))))),
        right_param_order=draw(st.permutations(tuple(range(len(_RESOLVED_PARAMS))))),
        left_reverse_e1=draw(st.booleans()),
        right_reverse_e1=draw(st.booleans()),
    )


def _build_ir(
    *,
    node_order: tuple[int, ...],
    element_order: tuple[int, ...],
    resolved_param_order: tuple[int, ...],
    reverse_e1_params: bool,
) -> CanonicalIR:
    nodes = tuple(
        IRNode(node_id=_NODE_IDS[index], is_reference=_NODE_IDS[index] == "0")
        for index in node_order
    )

    elements: list[IRElement] = []
    for index in element_order:
        element_id, element_type, element_nodes, params = _ELEMENT_DEFS[index]
        normalized_params = (
            tuple(reversed(params)) if (reverse_e1_params and element_id == "E1") else params
        )
        elements.append(IRElement(element_id, element_type, element_nodes, normalized_params))

    resolved_params = tuple(_RESOLVED_PARAMS[index] for index in resolved_param_order)
    return CanonicalIR(
        nodes=nodes,
        aux_unknowns=(),
        elements=tuple(elements),
        ports=(),
        resolved_params=resolved_params,
    )


@given(case=_reciprocal_y_cases())
def test_passive_reciprocal_y_to_s_remains_reciprocal(case: _ReciprocalYCase) -> None:
    frequencies = np.asarray([1.0], dtype=np.float64)
    y_matrix = np.asarray(
        [
            [
                [case.g1 + case.g12 + (1j * case.b1), -case.g12 + (1j * case.b12)],
                [-case.g12 + (1j * case.b12), case.g2 + case.g12 + (1j * case.b2)],
            ]
        ],
        dtype=np.complex128,
    )
    y_result = YParameterResult(
        frequencies_hz=frequencies,
        port_ids=("P1", "P2"),
        y=y_matrix,
        status=np.asarray(["pass"], dtype=np.dtype("<U8")),
        diagnostics_by_point=((),),
    )

    s_result = convert_y_to_s(y_result, z0_ohm=case.z0)

    assert str(s_result.status[0]) == "pass"
    assert s_result.diagnostics_by_point == ((),)
    assert np.isfinite(s_result.s[0].real).all()
    assert np.isfinite(s_result.s[0].imag).all()
    np.testing.assert_allclose(s_result.s[0, 0, 1], s_result.s[0, 1, 0], rtol=1e-10, atol=1e-10)


@given(case=_ir_permutation_cases())
def test_canonical_ir_hash_is_permutation_invariant(case: _IRPermutationCase) -> None:
    left = _build_ir(
        node_order=case.left_node_order,
        element_order=case.left_element_order,
        resolved_param_order=case.left_param_order,
        reverse_e1_params=case.left_reverse_e1,
    )
    right = _build_ir(
        node_order=case.right_node_order,
        element_order=case.right_element_order,
        resolved_param_order=case.right_param_order,
        reverse_e1_params=case.right_reverse_e1,
    )

    assert canonical_ir_json(left) == canonical_ir_json(right)
    assert hash_canonical_ir(left) == hash_canonical_ir(right)

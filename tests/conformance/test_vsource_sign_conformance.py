from __future__ import annotations

import json
from random import Random

import pytest

from rfmna.parser import IdealVSource, PreflightInput, preflight_check

pytestmark = pytest.mark.conformance


def _diag_json(diags: tuple[object, ...]) -> str:
    return json.dumps(
        [diag.model_dump(mode="json") for diag in diags],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_independent_vsource_orientation_sign_conventions() -> None:
    consistent = preflight_check(
        PreflightInput(
            nodes=("0", "n1"),
            reference_node="0",
            voltage_sources=(
                IdealVSource("V1", "0", "n1", 1.0),
                IdealVSource("V2", "n1", "0", -1.0),
            ),
        )
    )
    assert "E_TOPO_VSRC_LOOP_INCONSISTENT" not in {diag.code for diag in consistent}

    inconsistent = preflight_check(
        PreflightInput(
            nodes=("0", "n1"),
            reference_node="0",
            voltage_sources=(
                IdealVSource("V1", "0", "n1", 1.0),
                IdealVSource("V2", "n1", "0", 1.0),
            ),
        )
    )
    assert "E_TOPO_VSRC_LOOP_INCONSISTENT" in {diag.code for diag in inconsistent}


def test_vsource_loop_diagnostics_are_permutation_stable() -> None:
    base = PreflightInput(
        nodes=("0", "n1", "n2"),
        reference_node="0",
        voltage_sources=(
            IdealVSource("V1", "0", "n1", 1.0),
            IdealVSource("V2", "n1", "n2", 1.0),
            IdealVSource("V3", "n2", "0", -1.5),
        ),
    )
    baseline = _diag_json(preflight_check(base))
    rng = Random(0)

    for _ in range(25):
        nodes = list(base.nodes)
        sources = list(base.voltage_sources)
        rng.shuffle(nodes)
        rng.shuffle(sources)
        permuted = PreflightInput(
            nodes=tuple(nodes),
            reference_node=base.reference_node,
            voltage_sources=tuple(sources),
        )
        assert _diag_json(preflight_check(permuted)) == baseline

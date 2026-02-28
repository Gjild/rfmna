from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from rfmna.assembler import UnknownAuxIdError, build_unknown_indexing
from rfmna.elements import MatrixEntry, StampContext, VCCSStamp, VCVSStamp

pytestmark = pytest.mark.conformance

CONTROLLED_SOURCE_CONFORMANCE_MAP: Mapping[str, Sequence[str]] = {
    "STAMP_APPENDIX_5_VCCS_KCL": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_vccs_kcl_equations_match_stamp_appendix_section_5",
    ),
    "STAMP_APPENDIX_5_VCCS_INVALID_PARAM": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_invalid_controlled_source_parameters_emit_required_codes",
    ),
    "STAMP_APPENDIX_6_VCVS_AUX_REQUIRED": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_requires_allocated_aux_unknown",
    ),
    "STAMP_APPENDIX_6_VCVS_STAMP": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_vcvs_stamp_matches_appendix_section_6_with_aux_row",
    ),
    "STAMP_APPENDIX_6_VCVS_INVALID_PARAM": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_invalid_controlled_source_parameters_emit_required_codes",
    ),
    "STAMP_APPENDIX_5_6_ORIENTATION_MATRIX": (
        "tests/conformance/test_controlled_stamps_conformance.py::test_full_polarity_orientation_matrix_matches_sign_conventions",
    ),
}

EXPECTED_CONTROLLED_SOURCE_CONFORMANCE_IDS = {
    "STAMP_APPENDIX_5_VCCS_KCL",
    "STAMP_APPENDIX_5_VCCS_INVALID_PARAM",
    "STAMP_APPENDIX_6_VCVS_AUX_REQUIRED",
    "STAMP_APPENDIX_6_VCVS_STAMP",
    "STAMP_APPENDIX_6_VCVS_INVALID_PARAM",
    "STAMP_APPENDIX_5_6_ORIENTATION_MATRIX",
}

CONTROLLED_SOURCE_DIAGNOSTIC_CATALOG: Mapping[str, Mapping[str, str]] = {
    "E_MODEL_VCCS_INVALID": {
        "severity": "error",
        "solver_stage": "assemble",
        "suggested_action": "check VCCS transconductance value",
    },
    "E_MODEL_VCVS_INVALID": {
        "severity": "error",
        "solver_stage": "assemble",
        "suggested_action": "check VCVS gain value",
    },
}

_REQUIRED_CATALOG_FIELDS = ("severity", "solver_stage", "suggested_action")


def _ctx() -> StampContext:
    return StampContext(omega_rad_s=1.0, resolved_params={})


def test_controlled_conformance_mapping_declares_required_ids() -> None:
    assert set(CONTROLLED_SOURCE_CONFORMANCE_MAP) == EXPECTED_CONTROLLED_SOURCE_CONFORMANCE_IDS
    for conformance_id, entries in CONTROLLED_SOURCE_CONFORMANCE_MAP.items():
        assert entries, f"conformance id '{conformance_id}' must map to at least one test"


def test_controlled_conformance_mapping_entries_resolve(request: pytest.FixtureRequest) -> None:
    collected_nodeids = {item.nodeid.split("[", maxsplit=1)[0] for item in request.session.items}
    missing: list[str] = []
    for conformance_id, entries in CONTROLLED_SOURCE_CONFORMANCE_MAP.items():
        for nodeid in entries:
            if nodeid not in collected_nodeids:
                missing.append(f"{conformance_id}:{nodeid}")
    assert missing == []


def test_controlled_source_diagnostic_catalog_is_unique_and_schema_complete() -> None:
    assert set(CONTROLLED_SOURCE_DIAGNOSTIC_CATALOG) == {
        "E_MODEL_VCCS_INVALID",
        "E_MODEL_VCVS_INVALID",
    }
    assert len(CONTROLLED_SOURCE_DIAGNOSTIC_CATALOG) == len(
        set(CONTROLLED_SOURCE_DIAGNOSTIC_CATALOG)
    )
    for code, metadata in CONTROLLED_SOURCE_DIAGNOSTIC_CATALOG.items():
        assert code.startswith("E_MODEL_")
        assert set(metadata) == set(_REQUIRED_CATALOG_FIELDS)
        for field in _REQUIRED_CATALOG_FIELDS:
            assert metadata[field]


def test_vccs_kcl_equations_match_stamp_appendix_section_5() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0")
    stamp = VCCSStamp("G1", "n1", "n2", "nc", "nd", 2.0, indexing)

    assert stamp.stamp_A(_ctx()) == (
        MatrixEntry(0, 2, 2.0 + 0.0j),
        MatrixEntry(0, 3, -2.0 + 0.0j),
        MatrixEntry(1, 2, -2.0 + 0.0j),
        MatrixEntry(1, 3, 2.0 + 0.0j),
    )


def test_vcvs_requires_allocated_aux_unknown() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))
    stamp = VCVSStamp("E2", "n1", "n2", "nc", "nd", "missing_aux", 1.0, indexing)

    with pytest.raises(UnknownAuxIdError) as exc_info:
        stamp.touched_indices(_ctx())
    assert exc_info.value.code == "E_INDEX_AUX_UNKNOWN"


def test_vcvs_stamp_matches_appendix_section_6_with_aux_row() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))
    stamp = VCVSStamp("E1", "n1", "n2", "nc", "nd", "E1:i", 3.0, indexing)

    assert stamp.stamp_A(_ctx()) == (
        MatrixEntry(0, 4, 1.0 + 0.0j),
        MatrixEntry(1, 4, -1.0 + 0.0j),
        MatrixEntry(4, 0, 1.0 + 0.0j),
        MatrixEntry(4, 1, -1.0 + 0.0j),
        MatrixEntry(4, 2, -3.0 + 0.0j),
        MatrixEntry(4, 3, 3.0 + 0.0j),
    )


def test_invalid_controlled_source_parameters_emit_required_codes() -> None:
    indexing = build_unknown_indexing(("0", "n1", "n2", "nc", "nd"), "0", ("E1:i",))

    assert VCCSStamp("Gbad", "n1", "n2", "nc", "nd", float("nan"), indexing).validate(_ctx())[
        0
    ].code == ("E_MODEL_VCCS_INVALID")
    assert VCVSStamp("Ebad", "n1", "n2", "nc", "nd", "E1:i", float("inf"), indexing).validate(
        _ctx()
    )[0].code == ("E_MODEL_VCVS_INVALID")


def test_full_polarity_orientation_matrix_matches_sign_conventions() -> None:
    indexing = build_unknown_indexing(
        ("0", "n1", "n2", "nc", "nd"), "0", ("E1:i", "E2:i", "E3:i", "E4:i")
    )
    gm = 2.0
    mu = 3.0
    orientations = (
        ("n1", "n2", "nc", "nd", 1.0, 1.0),
        ("n2", "n1", "nc", "nd", -1.0, 1.0),
        ("n1", "n2", "nd", "nc", 1.0, -1.0),
        ("n2", "n1", "nd", "nc", -1.0, -1.0),
    )

    for ordinal, (a, b, c, d, output_sign, control_sign) in enumerate(orientations, start=1):
        vccs = VCCSStamp(f"G{ordinal}", a, b, c, d, gm, indexing)
        assert vccs.stamp_A(_ctx()) == (
            MatrixEntry(0, 2, (output_sign * control_sign * gm) + 0.0j),
            MatrixEntry(0, 3, (-output_sign * control_sign * gm) + 0.0j),
            MatrixEntry(1, 2, (-output_sign * control_sign * gm) + 0.0j),
            MatrixEntry(1, 3, (output_sign * control_sign * gm) + 0.0j),
        )

        vcvs = VCVSStamp(f"E{ordinal}", a, b, c, d, f"E{ordinal}:i", mu, indexing)
        assert vcvs.stamp_A(_ctx()) == (
            MatrixEntry(0, 4 + (ordinal - 1), output_sign + 0.0j),
            MatrixEntry(1, 4 + (ordinal - 1), -output_sign + 0.0j),
            MatrixEntry(4 + (ordinal - 1), 0, output_sign + 0.0j),
            MatrixEntry(4 + (ordinal - 1), 1, -output_sign + 0.0j),
            MatrixEntry(4 + (ordinal - 1), 2, (-control_sign * mu) + 0.0j),
            MatrixEntry(4 + (ordinal - 1), 3, (control_sign * mu) + 0.0j),
        )

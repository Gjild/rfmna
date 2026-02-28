from __future__ import annotations

import hashlib
import json

from .models import CanonicalIR


def _canonical_ir_payload(ir: CanonicalIR) -> dict[str, object]:
    return {
        "aux_unknowns": [
            {
                "aux_id": aux.aux_id,
                "kind": aux.kind,
                "owner_element_id": aux.owner_element_id,
            }
            for aux in ir.aux_unknowns
        ],
        "elements": [
            {
                "element_id": element.element_id,
                "element_type": element.element_type,
                "nodes": list(element.nodes),
                "params": [[key, value] for key, value in element.params],
            }
            for element in ir.elements
        ],
        "nodes": [
            {
                "is_reference": node.is_reference,
                "node_id": node.node_id,
            }
            for node in ir.nodes
        ],
        "ports": [
            {
                "p_minus": port.p_minus,
                "p_plus": port.p_plus,
                "port_id": port.port_id,
                "z0_ohm": port.z0_ohm,
            }
            for port in ir.ports
        ],
        "resolved_params": [[key, value] for key, value in ir.resolved_params],
    }


def canonical_ir_json(ir: CanonicalIR) -> str:
    return json.dumps(
        _canonical_ir_payload(ir),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def serialize_canonical_ir(ir: CanonicalIR) -> bytes:
    return canonical_ir_json(ir).encode("utf-8")


def hash_canonical_ir(ir: CanonicalIR, algo: str = "sha256") -> str:
    hasher = hashlib.new(algo)
    hasher.update(serialize_canonical_ir(ir))
    return hasher.hexdigest()

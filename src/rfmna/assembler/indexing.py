from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


def _validate_id(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _duplicate_id(values: Sequence[str]) -> str | None:
    previous: str | None = None
    for value in sorted(values):
        if value == previous:
            return value
        previous = value
    return None


class UnknownAuxIdError(KeyError):
    code = "E_INDEX_AUX_UNKNOWN"

    def __init__(self, aux_id: str) -> None:
        super().__init__(f"unknown auxiliary id: {aux_id}")
        self.aux_id = aux_id


@dataclass(frozen=True, slots=True)
class UnknownIndexing:
    n_nodes: int
    n_aux: int
    total_unknowns: int
    ordered_node_ids: tuple[str, ...]
    ordered_aux_ids: tuple[str, ...]
    reference_node: str
    node_voltage_slice: slice
    aux_slice: slice
    _node_to_index: Mapping[str, int]
    _aux_to_index: Mapping[str, int]

    def node_index(self, node_id: str) -> int | None:
        if node_id == self.reference_node:
            return None
        return self._node_to_index.get(node_id)

    def aux_index(self, aux_id: str) -> int:
        index = self._aux_to_index.get(aux_id)
        if index is None:
            raise UnknownAuxIdError(aux_id)
        return index

    def node_indices(self, node_ids: Sequence[str]) -> tuple[int | None, ...]:
        return tuple(self.node_index(node_id) for node_id in node_ids)

    def aux_indices(self, aux_ids: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.aux_index(aux_id) for aux_id in aux_ids)

    def projection(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, int], ...]]:
        return (
            self.ordered_node_ids,
            self.ordered_aux_ids,
            tuple((aux_id, self.aux_index(aux_id)) for aux_id in self.ordered_aux_ids),
        )


def build_unknown_indexing(
    node_ids: Sequence[str],
    reference_node: str,
    aux_ids: Sequence[str] = (),
) -> UnknownIndexing:
    _validate_id("reference_node", reference_node)
    for node_id in node_ids:
        _validate_id("node_id", node_id)
    for aux_id in aux_ids:
        _validate_id("aux_id", aux_id)

    duplicate_node = _duplicate_id(node_ids)
    if duplicate_node is not None:
        raise ValueError(f"duplicate node_id: {duplicate_node}")
    duplicate_aux = _duplicate_id(aux_ids)
    if duplicate_aux is not None:
        raise ValueError(f"duplicate aux_id: {duplicate_aux}")
    if reference_node not in node_ids:
        raise ValueError(f"reference_node is not declared: {reference_node}")

    ordered_nodes = tuple(node_id for node_id in node_ids if node_id != reference_node)
    ordered_aux = tuple(aux_ids)
    node_to_index = {node_id: idx for idx, node_id in enumerate(ordered_nodes)}
    n_nodes = len(ordered_nodes)
    aux_to_index = {aux_id: idx for idx, aux_id in enumerate(ordered_aux, start=n_nodes)}
    n_aux = len(ordered_aux)
    total_unknowns = n_nodes + n_aux

    return UnknownIndexing(
        n_nodes=n_nodes,
        n_aux=n_aux,
        total_unknowns=total_unknowns,
        ordered_node_ids=ordered_nodes,
        ordered_aux_ids=ordered_aux,
        reference_node=reference_node,
        node_voltage_slice=slice(0, n_nodes),
        aux_slice=slice(n_nodes, total_unknowns),
        _node_to_index=MappingProxyType(node_to_index),
        _aux_to_index=MappingProxyType(aux_to_index),
    )

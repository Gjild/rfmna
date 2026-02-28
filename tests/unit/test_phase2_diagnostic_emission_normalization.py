from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Final

import pytest

pytestmark = pytest.mark.unit

_NORMALIZED_RUNTIME_PREFIXES: Final[tuple[str, ...]] = (
    "src/rfmna/cli/",
    "src/rfmna/parser/",
    "src/rfmna/rf_metrics/",
    "src/rfmna/sweep_engine/",
)
_SOLVER_RUNTIME_PREFIX = "src/rfmna/solver/"
_DIAGNOSTIC_EVENT_TARGETS: Final[tuple[str, ...]] = (
    "DiagnosticEvent",
    "rfmna.diagnostics.DiagnosticEvent",
    "rfmna.diagnostics.models.DiagnosticEvent",
)
_DIAGNOSTIC_BUILDER_TARGETS: Final[tuple[str, ...]] = (
    "build_diagnostic_event",
    "rfmna.diagnostics.build_diagnostic_event",
    "rfmna.diagnostics.adapters.build_diagnostic_event",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_emission_python_paths() -> tuple[str, ...]:
    payload = json.loads(
        (_repo_root() / "docs/dev/diagnostic_runtime_code_inventory.yaml").read_text(
            encoding="utf-8"
        )
    )
    runtime_paths = payload["runtime_emission_paths"]
    assert isinstance(runtime_paths, list)
    paths = tuple(
        sorted(path for path in runtime_paths if isinstance(path, str) and path.endswith(".py"))
    )
    assert paths
    return paths


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in tree.body:
        _update_binding_for_statement(node=node, bindings=bindings)
    return bindings


def _update_binding_for_statement(*, node: ast.stmt, bindings: dict[str, str]) -> None:
    if isinstance(node, ast.Import):
        _update_import_binding(node=node, bindings=bindings)
        return
    if isinstance(node, ast.ImportFrom):
        _update_importfrom_binding(node=node, bindings=bindings)
        return
    if isinstance(node, ast.Assign):
        _update_assign_binding(node=node, bindings=bindings)
        return
    if isinstance(node, ast.AnnAssign):
        _update_annassign_binding(node=node, bindings=bindings)


def _update_import_binding(*, node: ast.Import, bindings: dict[str, str]) -> None:
    for alias in node.names:
        if alias.asname is not None:
            bindings[alias.asname] = alias.name
            continue
        root = alias.name.split(".", maxsplit=1)[0]
        bindings[root] = root


def _update_importfrom_binding(*, node: ast.ImportFrom, bindings: dict[str, str]) -> None:
    module = node.module
    if module is None:
        return
    for alias in node.names:
        if alias.name == "*":
            continue
        local = alias.asname if alias.asname is not None else alias.name
        bindings[local] = f"{module}.{alias.name}"


def _update_assign_binding(*, node: ast.Assign, bindings: dict[str, str]) -> None:
    if len(node.targets) != 1:
        return
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return
    resolved = _resolve_expr(node.value, bindings)
    if resolved is not None:
        bindings[target.id] = resolved


def _update_annassign_binding(*, node: ast.AnnAssign, bindings: dict[str, str]) -> None:
    if not isinstance(node.target, ast.Name):
        return
    if node.value is None:
        return
    resolved = _resolve_expr(node.value, bindings)
    if resolved is not None:
        bindings[node.target.id] = resolved


def _resolve_expr(expr: ast.expr, bindings: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return bindings.get(expr.id, expr.id)
    if isinstance(expr, ast.Attribute):
        base = _resolve_expr(expr.value, bindings)
        if base is None:
            return None
        return f"{base}.{expr.attr}"
    return None


def _call_symbol_counts(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    bindings = _import_bindings(tree)
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        symbol = _resolve_expr(node.func, bindings)
        if symbol is None:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _count_any(counts: dict[str, int], targets: tuple[str, ...]) -> int:
    return sum(counts.get(target, 0) for target in targets)


def test_runtime_inventory_scope_includes_parser_and_solver_paths() -> None:
    runtime_paths = _runtime_emission_python_paths()
    assert any(path.startswith("src/rfmna/parser/") for path in runtime_paths)
    assert any(path.startswith(_SOLVER_RUNTIME_PREFIX) for path in runtime_paths)


def test_runtime_emission_paths_disallow_direct_diagnostic_event_constructor_even_via_alias() -> (
    None
):
    root = _repo_root()
    for rel_path in _runtime_emission_python_paths():
        counts = _call_symbol_counts(root / rel_path)
        assert _count_any(counts, _DIAGNOSTIC_EVENT_TARGETS) == 0, rel_path


def test_normalized_runtime_paths_use_canonical_builder() -> None:
    root = _repo_root()
    for rel_path in _runtime_emission_python_paths():
        if rel_path.startswith(_SOLVER_RUNTIME_PREFIX):
            continue
        if not rel_path.startswith(_NORMALIZED_RUNTIME_PREFIXES):
            continue
        counts = _call_symbol_counts(root / rel_path)
        assert _count_any(counts, _DIAGNOSTIC_BUILDER_TARGETS) > 0, rel_path


def test_solver_runtime_paths_are_explicitly_in_scope_and_not_direct_diagnostic_emitters() -> None:
    root = _repo_root()
    solver_paths = tuple(
        path for path in _runtime_emission_python_paths() if path.startswith(_SOLVER_RUNTIME_PREFIX)
    )
    assert solver_paths
    for rel_path in solver_paths:
        counts = _call_symbol_counts(root / rel_path)
        assert _count_any(counts, _DIAGNOSTIC_EVENT_TARGETS) == 0, rel_path


def test_sweep_engine_has_single_sweep_diagnostic_adapter_point() -> None:
    counts = _call_symbol_counts(_repo_root() / "src/rfmna/sweep_engine/run.py")
    assert counts.get("SweepDiagnostic", 0) == 1

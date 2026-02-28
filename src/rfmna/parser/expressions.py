from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping

from .errors import ParseErrorCode, build_parse_error
from .numbers import parse_scalar_number

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda lhs, rhs: lhs + rhs,
    ast.Sub: lambda lhs, rhs: lhs - rhs,
    ast.Mult: lambda lhs, rhs: lhs * rhs,
    ast.Div: lambda lhs, rhs: lhs / rhs,
    ast.Pow: lambda lhs, rhs: lhs**rhs,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda value: +value,
    ast.USub: lambda value: -value,
}


def _parse_expression_body(expr: str) -> ast.expr:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise build_parse_error(
            ParseErrorCode.E_PARSE_EXPR_INVALID,
            "invalid parameter expression",
            expr,
        ) from exc
    return tree.body


def _constant_to_number(node: ast.Constant, expr: str) -> float:
    if isinstance(node.value, bool) or not isinstance(node.value, int | float):
        raise build_parse_error(
            ParseErrorCode.E_PARSE_EXPR_INVALID,
            "invalid parameter expression",
            expr,
        )
    literal = ast.get_source_segment(expr, node)
    if literal is None:
        literal = str(node.value)
    return parse_scalar_number(literal)


def _evaluate_node(node: ast.expr, expr: str, symbols: Mapping[str, float]) -> float:
    if isinstance(node, ast.BinOp):
        operator_fn = _BIN_OPS.get(type(node.op))
        if operator_fn is None:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_EXPR_INVALID,
                "invalid parameter expression",
                expr,
            )
        lhs = _evaluate_node(node.left, expr, symbols)
        rhs = _evaluate_node(node.right, expr, symbols)
        try:
            value = operator_fn(lhs, rhs)
        except (OverflowError, ZeroDivisionError) as exc:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_NONFINITE,
                "parameter expression must evaluate to a finite value",
                expr,
            ) from exc
        if not math.isfinite(value):
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_NONFINITE,
                "parameter expression must evaluate to a finite value",
                expr,
            )
        return float(value)

    if isinstance(node, ast.UnaryOp):
        unary_operator_fn = _UNARY_OPS.get(type(node.op))
        if unary_operator_fn is None:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_EXPR_INVALID,
                "invalid parameter expression",
                expr,
            )
        value = _evaluate_node(node.operand, expr, symbols)
        try:
            result = unary_operator_fn(value)
        except OverflowError as exc:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_NONFINITE,
                "parameter expression must evaluate to a finite value",
                expr,
            ) from exc
        if not math.isfinite(result):
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_NONFINITE,
                "parameter expression must evaluate to a finite value",
                expr,
            )
        return float(result)

    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_PARAM_UNDEFINED,
                f"undefined parameter reference: {node.id}",
                expr,
                witness=(node.id,),
            )
        return symbols[node.id]

    if isinstance(node, ast.Constant):
        return _constant_to_number(node, expr)

    raise build_parse_error(
        ParseErrorCode.E_PARSE_EXPR_INVALID,
        "invalid parameter expression",
        expr,
    )


def _collect_dependencies(node: ast.expr, expr: str, dependencies: set[str]) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_EXPR_INVALID,
                "invalid parameter expression",
                expr,
            )
        _collect_dependencies(node.left, expr, dependencies)
        _collect_dependencies(node.right, expr, dependencies)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise build_parse_error(
                ParseErrorCode.E_PARSE_EXPR_INVALID,
                "invalid parameter expression",
                expr,
            )
        _collect_dependencies(node.operand, expr, dependencies)
        return

    if isinstance(node, ast.Name):
        dependencies.add(node.id)
        return

    if isinstance(node, ast.Constant):
        _constant_to_number(node, expr)
        return

    raise build_parse_error(
        ParseErrorCode.E_PARSE_EXPR_INVALID,
        "invalid parameter expression",
        expr,
    )


def extract_dependencies(expr: str) -> tuple[str, ...]:
    body = _parse_expression_body(expr)
    dependencies: set[str] = set()
    _collect_dependencies(body, expr, dependencies)
    return tuple(sorted(dependencies))


def evaluate_expression(expr: str, symbols: Mapping[str, float]) -> float:
    body = _parse_expression_body(expr)
    value = _evaluate_node(body, expr, symbols)
    if not math.isfinite(value):
        raise build_parse_error(
            ParseErrorCode.E_PARSE_PARAM_NONFINITE,
            "parameter expression must evaluate to a finite value",
            expr,
        )
    return float(value)

"""Compile boolean expressions and tiny straight-line programs into networks.

>>> from pnn.expr import compile_expression
>>> net = compile_expression("(a & b) | ~c")
>>> net.forward([1, 1, 0])
array([1.])

Programs are sequences of assignments; every assigned name that does not start
with ``_`` becomes an output::

    _p = a ^ b
    sum = _p ^ cin
    carry = maj(a, b, cin)

Supported syntax: ``&  |  ^  ~``, ``and  or  not``, the constants ``0 1 True
False`` and the functions ``AND OR NOT NAND NOR XOR XNOR MAJ`` (any case).
"""

from __future__ import annotations

import ast

from .circuit import Circuit, Wire
from .compiler import compile_circuit
from .network import Network

_FUNCS = {
    "and": "and_",
    "or": "or_",
    "not": "not_",
    "nand": "nand",
    "nor": "nor",
    "xor": "xor",
    "xnor": "xnor",
    "maj": "maj",
}


class _Builder:
    def __init__(self, circuit: Circuit, inputs: list[str] | None):
        self.c = circuit
        self.env: dict[str, Wire] = {}
        self.fixed_inputs = inputs is not None
        for name in inputs or []:
            self.env[name] = circuit.input(name)

    def name(self, ident: str) -> Wire:
        if ident not in self.env:
            if self.fixed_inputs:
                raise NameError(f"unknown name {ident!r}")
            self.env[ident] = self.c.input(ident)
        return self.env[ident]

    def build(self, node: ast.AST) -> Wire:
        c = self.c
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.Constant) and node.value in (0, 1, True, False):
            return c.const(int(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
            return c.not_(self.build(node.operand))
        if isinstance(node, ast.BinOp):
            l, r = self.build(node.left), self.build(node.right)
            if isinstance(node.op, ast.BitAnd):
                return c.and_(l, r)
            if isinstance(node.op, ast.BitOr):
                return c.or_(l, r)
            if isinstance(node.op, ast.BitXor):
                return c.xor(l, r)
        if isinstance(node, ast.BoolOp):
            args = [self.build(v) for v in node.values]
            return c.and_(*args) if isinstance(node.op, ast.And) else c.or_(*args)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCS.get(node.func.id.lower().rstrip("_"))
            if fn is not None:
                return getattr(c, fn)(*[self.build(a) for a in node.args])
        raise SyntaxError(f"unsupported syntax: {ast.dump(node)}")


def compile_program(source: str, inputs: list[str] | None = None) -> Network:
    """Compile a straight-line boolean program (see module docstring)."""
    tree = ast.parse(source.strip(), mode="exec")
    c = Circuit()
    bld = _Builder(c, inputs)
    outputs: dict[str, Wire] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            wire = bld.build(stmt.value)
            bld.env[target] = wire
            if not target.startswith("_"):
                outputs[target] = wire
        elif isinstance(stmt, ast.Expr) and len(tree.body) == 1:
            outputs["out"] = bld.build(stmt.value)
        else:
            raise SyntaxError(f"unsupported statement: {ast.dump(stmt)}")
    return compile_circuit(c, outputs)


def compile_expression(expr: str, inputs: list[str] | None = None) -> Network:
    """Compile a single boolean expression to a network with one output ``out``.

    Inputs are ordered by first appearance unless ``inputs`` is given.
    """
    return compile_program(expr, inputs)

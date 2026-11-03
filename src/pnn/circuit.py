"""Circuits of threshold gates.

A :class:`Circuit` is a directed acyclic graph whose every internal node is a
*threshold neuron*::

    y = step(sum_i w_i * x_i + bias)

with integer weights and a half-integer bias, so that for binary inputs the
pre-activation is never exactly zero (there is always a margin of at least 0.5).

Classic gates (AND, OR, NOT, NAND, NOR, XOR, majority, half/full adders) are
just particular choices of ``(weights, bias)``. While the graph is being built,
the circuit performs a few cheap *peephole optimisations*:

* constant folding   - a constant input is absorbed into the bias;
* NOT folding        - ``w * NOT(x)`` is rewritten as ``w - w * x``;
* identity removal   - ``step(x - 0.5)`` is just ``x``;
* constant detection - a gate whose output cannot change becomes a constant;
* hash-consing       - structurally identical gates are created only once.

The circuit is later turned into layered weight matrices by
:func:`pnn.compiler.compile_circuit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Node:
    """A node of a circuit: an input, a constant, or a threshold gate."""

    id: int
    kind: str  # "input" | "const" | "gate"
    name: str | None = None
    inputs: tuple[int, ...] = ()
    weights: tuple[int, ...] = ()
    bias: float = 0.0
    value: int = 0  # only for constants


class Wire:
    """A handle to a node, with operator overloading for building logic.

    ``a & b``, ``a | b``, ``a ^ b`` and ``~a`` create AND, OR, XOR and NOT gates.
    """

    __slots__ = ("circuit", "id")

    def __init__(self, circuit: "Circuit", id: int):
        self.circuit = circuit
        self.id = id

    @property
    def node(self) -> Node:
        return self.circuit.nodes[self.id]

    def __and__(self, other: "Wire") -> "Wire":
        return self.circuit.and_(self, other)

    def __or__(self, other: "Wire") -> "Wire":
        return self.circuit.or_(self, other)

    def __xor__(self, other: "Wire") -> "Wire":
        return self.circuit.xor(self, other)

    def __invert__(self) -> "Wire":
        return self.circuit.not_(self)

    def __eq__(self, other: object) -> bool:  # identity semantics
        return isinstance(other, Wire) and other.circuit is self.circuit and other.id == self.id

    def __hash__(self) -> int:
        return hash((id(self.circuit), self.id))

    def __repr__(self) -> str:
        n = self.node
        return f"Wire({n.kind}:{n.name or n.id})"


def _is_half_integer(x: float) -> bool:
    return abs((x - 0.5) - round(x - 0.5)) < 1e-9


class Circuit:
    """A DAG of threshold gates with peephole optimisation while building."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._cache: dict[tuple, int] = {}
        self._names: dict[str, int] = {}

    # ------------------------------------------------------------------ nodes
    def _add(self, **kw) -> Wire:
        node = Node(id=len(self.nodes), **kw)
        self.nodes.append(node)
        return Wire(self, node.id)

    def input(self, name: str) -> Wire:
        """Declare a named binary input."""
        if name in self._names:
            raise ValueError(f"duplicate input name {name!r}")
        w = self._add(kind="input", name=name)
        self._names[name] = w.id
        return w

    def inputs(self, prefix: str, n: int) -> list[Wire]:
        """Declare ``n`` inputs ``prefix0 .. prefix{n-1}`` (LSB first for numbers)."""
        return [self.input(f"{prefix}{i}") for i in range(n)]

    def const(self, value: int | bool) -> Wire:
        value = int(bool(value))
        key = ("const", value)
        if key not in self._cache:
            self._cache[key] = self._add(kind="const", value=value).id
        return Wire(self, self._cache[key])

    @property
    def input_wires(self) -> list[Wire]:
        return [Wire(self, n.id) for n in self.nodes if n.kind == "input"]

    def _is_not(self, node: Node) -> bool:
        return node.kind == "gate" and node.weights == (-1,) and abs(node.bias - 0.5) < 1e-9

    # --------------------------------------------------------- threshold gate
    def threshold(
        self,
        inputs: Sequence[Wire],
        weights: Sequence[int],
        bias: float,
        name: str | None = None,
    ) -> Wire:
        """Create ``step(sum(w_i * x_i) + bias)``.

        Weights must be integers and the bias a half-integer (…, -1.5, -0.5,
        0.5, 1.5, …) so every pre-activation has a margin of at least 0.5.
        """
        if len(inputs) != len(weights):
            raise ValueError("inputs and weights must have the same length")
        if not _is_half_integer(bias):
            raise ValueError(f"bias must be a half-integer, got {bias}")
        acc: dict[int, int] = {}
        b = float(bias)
        for wire, w in zip(inputs, weights):
            if wire.circuit is not self:
                raise ValueError("wire belongs to a different circuit")
            if int(w) != w:
                raise ValueError(f"weights must be integers, got {w}")
            w = int(w)
            node = wire.node
            if node.kind == "const":  # constant folding
                b += w * node.value
            elif self._is_not(node):  # w*NOT(x) = w - w*x
                b += w
                acc[node.inputs[0]] = acc.get(node.inputs[0], 0) - w
            else:
                acc[node.id] = acc.get(node.id, 0) + w
        items = sorted((i, w) for i, w in acc.items() if w != 0)
        # constant detection: output cannot change for any binary input
        max_z = b + sum(w for _, w in items if w > 0)
        min_z = b + sum(w for _, w in items if w < 0)
        if max_z < 0:
            return self.const(0)
        if min_z > 0:
            return self.const(1)
        # identity removal: step(x - 0.5) == x
        if len(items) == 1 and items[0][1] == 1 and abs(b + 0.5) < 1e-9:
            return Wire(self, items[0][0])
        key = ("gate", tuple(items), b)
        if key in self._cache:
            return Wire(self, self._cache[key])
        wire = self._add(
            kind="gate",
            name=name,
            inputs=tuple(i for i, _ in items),
            weights=tuple(w for _, w in items),
            bias=b,
        )
        self._cache[key] = wire.id
        return wire

    # ------------------------------------------------------------ basic gates
    def not_(self, x: Wire) -> Wire:
        return self.threshold([x], [-1], 0.5)

    def and_(self, *xs: Wire) -> Wire:
        xs = _flat(xs)
        return self.threshold(xs, [1] * len(xs), -(len(xs) - 0.5))

    def or_(self, *xs: Wire) -> Wire:
        xs = _flat(xs)
        return self.threshold(xs, [1] * len(xs), -0.5)

    def nand(self, *xs: Wire) -> Wire:
        xs = _flat(xs)
        return self.threshold(xs, [-1] * len(xs), len(xs) - 0.5)

    def nor(self, *xs: Wire) -> Wire:
        xs = _flat(xs)
        return self.threshold(xs, [-1] * len(xs), 0.5)

    def maj(self, x: Wire, y: Wire, z: Wire) -> Wire:
        """Majority of three: 1 if at least two inputs are 1 (= full-adder carry)."""
        return self.threshold([x, y, z], [1, 1, 1], -1.5)

    def xor(self, *xs: Wire) -> Wire:
        """XOR is not linearly separable, so it takes two neurons:

        ``c = AND(x, y)`` and then ``step(x + y - 2c - 0.5)`` (a skip connection).
        """
        xs = _flat(xs)
        out = xs[0]
        for y in xs[1:]:
            c = self.and_(out, y)
            out = self.threshold([out, y, c], [1, 1, -2], -0.5)
        return out

    def xnor(self, x: Wire, y: Wire) -> Wire:
        c = self.and_(x, y)
        return self.threshold([x, y, c], [-1, -1, 2], 0.5)

    def half_adder(self, a: Wire, b: Wire) -> tuple[Wire, Wire]:
        """Return ``(sum, carry)``."""
        c = self.and_(a, b)
        s = self.threshold([a, b, c], [1, 1, -2], -0.5)
        return s, c

    def full_adder(self, a: Wire, b: Wire, cin: Wire) -> tuple[Wire, Wire]:
        """Return ``(sum, carry)`` using just two threshold neurons.

        ``carry = step(a + b + cin - 1.5)`` and
        ``sum   = step(a + b + cin - 2*carry - 0.5)``.
        """
        c = self.maj(a, b, cin)
        s = self.threshold([a, b, cin, c], [1, 1, 1, -2], -0.5)
        return s, c

    # ------------------------------------------------------------- reference
    def evaluate(self, assignment: dict[str, int], wires: Iterable[Wire]) -> list[int]:
        """Slow, pure-Python reference evaluation (used for testing the compiler)."""
        values: dict[int, int] = {}
        for node in self.nodes:
            if node.kind == "input":
                values[node.id] = int(assignment[node.name])
            elif node.kind == "const":
                values[node.id] = node.value
            else:
                z = node.bias + sum(w * values[i] for i, w in zip(node.inputs, node.weights))
                values[node.id] = int(z > 0)
        return [values[w.id] for w in wires]


def _flat(xs) -> list[Wire]:
    out: list[Wire] = []
    for x in xs:
        if isinstance(x, Wire):
            out.append(x)
        else:
            out.extend(_flat(x))
    if not out:
        raise ValueError("gate needs at least one input")
    return out

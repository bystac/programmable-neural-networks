"""Binary arithmetic circuits built from threshold neurons, and compiled networks.

Numbers are little-endian bit vectors (index 0 = least significant bit).

Two construction styles are provided:

``"ripple"``
    Textbook logic: ripple-carry adders made of two-neuron full adders.
    Small weights (|w| <= 2), depth grows linearly with the number of bits.

``"lookahead"``
    Threshold-logic carry-lookahead: the carry into bit ``i`` is a *single*
    neuron ``step(sum_{j<i} 2^j (a_j + b_j) + cin - 2^i + 0.5)``. An n-bit
    adder is then only 2 layers deep, at the price of weights up to 2^(n-1).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .circuit import Circuit, Wire
from .compiler import compile_circuit
from .network import Network

STYLES = ("ripple", "lookahead")


# --------------------------------------------------------------------- adders
def ripple_add(c: Circuit, xs: Sequence[Wire], ys: Sequence[Wire], cin: Wire | None = None):
    """Ripple-carry adder. Returns ``(sum_bits, carry_out)``."""
    assert len(xs) == len(ys)
    carry = cin if cin is not None else c.const(0)
    sums = []
    for x, y in zip(xs, ys):
        s, carry = c.full_adder(x, y, carry)
        sums.append(s)
    return sums, carry


def lookahead_add(c: Circuit, xs: Sequence[Wire], ys: Sequence[Wire], cin: Wire | None = None):
    """Two-layer threshold carry-lookahead adder. Returns ``(sum_bits, carry_out)``."""
    assert len(xs) == len(ys)
    n = len(xs)
    cin = cin if cin is not None else c.const(0)
    carries = [cin]
    for i in range(1, n + 1):
        ins, ws = [cin], [1]
        for j in range(i):
            ins += [xs[j], ys[j]]
            ws += [2**j, 2**j]
        carries.append(c.threshold(ins, ws, -(2**i) + 0.5))
    sums = [
        c.threshold([xs[i], ys[i], carries[i], carries[i + 1]], [1, 1, 1, -2], -0.5)
        for i in range(n)
    ]
    return sums, carries[n]


def add_bits(c: Circuit, xs, ys, cin=None, style: str = "ripple"):
    if style == "ripple":
        return ripple_add(c, xs, ys, cin)
    if style == "lookahead":
        return lookahead_add(c, xs, ys, cin)
    raise ValueError(f"unknown style {style!r}; choose from {STYLES}")


def sub_bits(c: Circuit, xs, ys, style: str = "ripple"):
    """Two's complement subtraction ``x - y = x + ~y + 1``. Returns ``(diff, borrow)``."""
    diff, carry = add_bits(c, xs, [c.not_(y) for y in ys], c.const(1), style)
    return diff, c.not_(carry)  # borrow = no carry out


def mul_bits(c: Circuit, xs, ys, style: str = "ripple"):
    """Shift-and-add (array) multiplier. Returns the ``len(xs)+len(ys)`` product bits."""
    n, m = len(xs), len(ys)
    width = n + m
    zero = c.const(0)
    acc = [c.and_(xs[i], ys[0]) for i in range(n)] + [zero] * (width - n)
    for j in range(1, m):
        row = [zero] * j + [c.and_(xs[i], ys[j]) for i in range(n)]
        row += [zero] * (width - len(row))
        # bits below j are already final; folding makes the zero columns free
        hi, _ = add_bits(c, acc[j:], row[j:], None, style)
        acc = acc[:j] + hi
    return acc


def less_than_bits(c: Circuit, xs, ys, style: str = "ripple") -> Wire:
    if style == "lookahead":
        # a < b  <=>  sum 2^j (b_j - a_j) > 0 : a single neuron
        n = len(xs)
        return c.threshold(list(xs) + list(ys), [-(2**j) for j in range(n)] + [2**j for j in range(n)], -0.5)
    _, borrow = sub_bits(c, xs, ys, style)
    return borrow


def equal_bits(c: Circuit, xs, ys, style: str = "ripple") -> Wire:
    if style == "lookahead":
        return c.nor(less_than_bits(c, xs, ys, style), less_than_bits(c, ys, xs, style))
    return c.and_(*[c.xnor(x, y) for x, y in zip(xs, ys)])


# ------------------------------------------------------------ compiled nets
OPS = {"+": "add", "-": "sub", "*": "mul", "<": "lt", "==": "eq"}


@lru_cache(maxsize=None)
def build_network(op: str, bits: int = 8, style: str = "ripple") -> Network:
    """Compile an ``bits``-bit calculator network for ``op`` in ``+ - * < ==``.

    Inputs are ``a0..a{n-1}, b0..b{n-1}`` (LSB first). Outputs:

    * ``+``  -> ``s0..s{n-1}, carry``
    * ``-``  -> ``d0..d{n-1}, borrow``   (result mod 2^n; borrow = a < b)
    * ``*``  -> ``p0..p{2n-1}``
    * ``<``  -> ``lt``
    * ``==`` -> ``eq``
    """
    op = OPS.get(op, op)
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    c = Circuit()
    a = c.inputs("a", bits)
    b = c.inputs("b", bits)
    if op == "add":
        s, carry = add_bits(c, a, b, None, style)
        outs = [(f"s{i}", w) for i, w in enumerate(s)] + [("carry", carry)]
    elif op == "sub":
        d, borrow = sub_bits(c, a, b, style)
        outs = [(f"d{i}", w) for i, w in enumerate(d)] + [("borrow", borrow)]
    elif op == "mul":
        p = mul_bits(c, a, b, style)
        outs = [(f"p{i}", w) for i, w in enumerate(p)]
    elif op == "lt":
        outs = [("lt", less_than_bits(c, a, b, style))]
    elif op == "eq":
        outs = [("eq", equal_bits(c, a, b, style))]
    else:
        raise ValueError(f"unknown op {op!r}")
    return compile_circuit(c, outs)


# ---------------------------------------------------------------- encoding
def to_bits(values, bits: int):
    """Encode integers (scalar or array) as little-endian bit arrays."""
    import numpy as np

    v = np.asarray(values, dtype=np.int64)
    return ((v[..., None] >> np.arange(bits)) & 1).astype(np.float64)


def from_bits(bit_array):
    """Decode little-endian bit arrays (thresholded at 0.5) back to integers."""
    import numpy as np

    B = (np.asarray(bit_array) > 0.5).astype(np.int64)
    return (B << np.arange(B.shape[-1])).sum(axis=-1)


def run(op: str, a, b, bits: int = 8, style: str = "ripple", mode: str = "step", steepness: float = 20.0):
    """Evaluate a calculator network on (arrays of) integers.

    Returns ``(value, flag)`` where ``flag`` is the carry/borrow bit for ``+``/``-``
    and ``None`` otherwise. ``value`` is an int array (or 0/1 for comparisons).
    """
    import numpy as np

    net = build_network(op, bits, style)
    a = np.asarray(a)
    b = np.asarray(b)
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= 2**bits) or np.any(b >= 2**bits):
        raise ValueError(f"operands must be in [0, {2**bits - 1}] for {bits}-bit mode")
    X = np.concatenate([to_bits(a, bits), to_bits(b, bits)], axis=-1)
    Y = net.forward(X, mode=mode, steepness=steepness)
    op = OPS.get(op, op)
    if op in ("add", "sub"):
        return from_bits(Y[..., :bits]), (Y[..., bits] > 0.5).astype(np.int64)
    return from_bits(Y), None

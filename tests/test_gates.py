"""Truth tables of every gate, each compiled into a real network."""

import itertools

import numpy as np
import pytest

from pnn import Circuit, compile_circuit

GATES = {
    "and": (2, lambda a, b: a & b),
    "or": (2, lambda a, b: a | b),
    "nand": (2, lambda a, b: 1 - (a & b)),
    "nor": (2, lambda a, b: 1 - (a | b)),
    "xor": (2, lambda a, b: a ^ b),
    "xnor": (2, lambda a, b: 1 - (a ^ b)),
    "maj": (3, lambda a, b, c: int(a + b + c >= 2)),
}


def _net(fn_name, n):
    c = Circuit()
    xs = c.inputs("x", n)
    method = {"and": "and_", "or": "or_"}.get(fn_name, fn_name)
    return compile_circuit(c, {"y": getattr(c, method)(*xs)})


@pytest.mark.parametrize("name", sorted(GATES))
def test_gate_truth_table(name):
    n, ref = GATES[name]
    net = _net(name, n)
    for bits in itertools.product([0, 1], repeat=n):
        assert net.forward(bits)[0] == ref(*bits), (name, bits)


def test_not():
    c = Circuit()
    x = c.input("x")
    net = compile_circuit(c, {"y": ~x})
    assert net.forward([0])[0] == 1 and net.forward([1])[0] == 0
    assert net.stats()["neurons"] == 1


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_wide_and_or_are_single_neurons(n):
    c = Circuit()
    xs = c.inputs("x", n)
    net = compile_circuit(c, {"and": c.and_(*xs), "or": c.or_(*xs)})
    if n > 1:
        assert net.stats() ["layers"] == 1 and net.stats()["neurons"] == 2
    for bits in itertools.product([0, 1], repeat=n):
        y = net.forward(bits)
        assert y[0] == all(bits) and y[1] == any(bits)


def test_half_adder():
    c = Circuit()
    a, b = c.inputs("x", 2)
    s, carry = c.half_adder(a, b)
    net = compile_circuit(c, {"s": s, "c": carry})
    for x, y in itertools.product([0, 1], repeat=2):
        assert list(net.forward([x, y])) == [(x + y) % 2, (x + y) // 2]


def test_full_adder_is_two_neurons():
    c = Circuit()
    a, b, cin = c.inputs("x", 3)
    s, carry = c.full_adder(a, b, cin)
    net = compile_circuit(c, {"s": s, "c": carry})
    for x, y, z in itertools.product([0, 1], repeat=3):
        assert list(net.forward([x, y, z])) == [(x + y + z) % 2, (x + y + z) // 2]
    st = net.stats()
    assert st["logic_neurons"] == 2 and st["layers"] == 2


def test_multi_input_xor_parity():
    c = Circuit()
    xs = c.inputs("x", 4)
    net = compile_circuit(c, {"p": c.xor(*xs)})
    for bits in itertools.product([0, 1], repeat=4):
        assert net.forward(bits)[0] == sum(bits) % 2


def test_operator_overloading():
    c = Circuit()
    a, b, d = c.inputs("x", 3)
    net = compile_circuit(c, {"y": (a & b) | (~d ^ a)})
    for x, y, z in itertools.product([0, 1], repeat=3):
        assert net.forward([x, y, z])[0] == ((x & y) | ((1 - z) ^ x))

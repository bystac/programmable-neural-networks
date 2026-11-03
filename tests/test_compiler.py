"""Compiler: folding, scheduling, relays, expressions, export."""

import itertools
import random

import numpy as np
import pytest

from pnn import Circuit, Network, compile_circuit, compile_expression, compile_program


def test_constant_and_not_folding():
    c = Circuit()
    x = c.input("x")
    assert c.and_(x, c.const(1)) == x  # AND(x,1) -> x
    assert c.and_(x, c.const(0)).node.kind == "const"
    assert c.not_(c.not_(x)) == x  # double negation disappears
    assert c.or_(x, ~x).node.value == 1  # tautology detected
    assert c.and_(x, ~x).node.value == 0  # contradiction detected


def test_hash_consing():
    c = Circuit()
    a, b = c.inputs("x", 2)
    assert c.and_(a, b) == c.and_(b, a)
    n = len(c.nodes)
    c.xor(a, b)
    c.xor(a, b)
    assert len(c.nodes) == n + 1  # xor reuses the existing AND neuron


def test_relays_carry_skip_connections():
    c = Circuit()
    a, b, d = c.inputs("x", 3)
    deep = c.and_(c.or_(a, b), c.nand(a, b))  # depth 2
    y = c.or_(deep, d)  # d must be relayed from layer 0 to layer 2
    net = compile_circuit(c, {"y": y, "a_copy": a, "one": c.const(1)})
    st = net.stats()
    assert st["layers"] == 3
    assert st["relay_neurons"] >= 1
    for bits in itertools.product([0, 1], repeat=3):
        x0, x1, x2 = bits
        assert list(net.forward(bits)) == [int((x0 ^ x1) or x2), x0, 1]


def _random_circuit(rng, n_inputs=4, n_gates=25):
    c = Circuit()
    wires = c.inputs("x", n_inputs)
    for _ in range(n_gates):
        kind = rng.choice(["and", "or", "xor", "not", "nand", "maj", "thr"])
        pick = lambda k: [rng.choice(wires) for _ in range(k)]
        if kind == "and":
            w = c.and_(*pick(2))
        elif kind == "or":
            w = c.or_(*pick(3))
        elif kind == "xor":
            w = c.xor(*pick(2))
        elif kind == "not":
            w = c.not_(*pick(1))
        elif kind == "nand":
            w = c.nand(*pick(2))
        elif kind == "maj":
            w = c.maj(*pick(3))
        else:
            ins = pick(3)
            w = c.threshold(ins, [rng.randint(-3, 3) for _ in ins], rng.randint(-3, 3) + 0.5)
        wires.append(w)
    outs = {f"y{i}": rng.choice(wires) for i in range(4)}
    return c, outs


@pytest.mark.parametrize("seed", range(30))
def test_compiled_network_matches_reference_evaluator(seed):
    rng = random.Random(seed)
    c, outs = _random_circuit(rng)
    net = compile_circuit(c, outs)
    names = [n.name for n in c.nodes if n.kind == "input"]
    X = np.array(list(itertools.product([0, 1], repeat=len(names))), dtype=float)
    Y = net.forward(X)
    for x, y in zip(X, Y):
        ref = c.evaluate(dict(zip(names, x.astype(int))), outs.values())
        assert list(y.astype(int)) == ref


def test_threshold_validation():
    c = Circuit()
    x = c.input("x")
    with pytest.raises(ValueError):
        c.threshold([x], [1], 0.0)  # bias must be a half-integer
    with pytest.raises(ValueError):
        c.threshold([x], [0.5], 0.5)  # weights must be integers


def test_compile_expression():
    net = compile_expression("(a & b) | ~c")
    assert net.input_names == ["a", "b", "c"]
    for a, b, cc in itertools.product([0, 1], repeat=3):
        assert net.forward([a, b, cc])[0] == ((a & b) | (1 - cc))


def test_compile_program_full_adder():
    net = compile_program(
        """
        _p = a ^ b
        sum = _p ^ cin
        carry = MAJ(a, b, cin)
        """.replace("        ", "")
    )
    assert net.output_names == ["sum", "carry"]
    for a, b, cin in itertools.product([0, 1], repeat=3):
        assert list(net.forward([a, b, cin])) == [(a + b + cin) % 2, (a + b + cin) // 2]


def test_compile_expression_keywords_and_functions():
    net = compile_expression("not (a and b) or nor(c, 0) or xnor(a, 1) & True", inputs=["a", "b", "c"])
    for a, b, cc in itertools.product([0, 1], repeat=3):
        ref = int((not (a and b)) or (not cc) or (a == 1))
        assert net.forward([a, b, cc])[0] == ref


def test_export_roundtrip(tmp_path):
    from pnn import build_network

    net = build_network("+", 4)
    X = np.array(list(itertools.product([0, 1], repeat=8)), dtype=float)
    Y = net.forward(X)
    net.save_npz(tmp_path / "add.npz")
    net.save_json(tmp_path / "add.json")
    for loaded in (Network.load_npz(tmp_path / "add.npz"), Network.load_json(tmp_path / "add.json")):
        assert loaded.output_names == net.output_names
        assert loaded.stats() == net.stats()
        np.testing.assert_array_equal(loaded.forward(X), Y)

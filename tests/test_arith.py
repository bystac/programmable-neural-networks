"""Exhaustive arithmetic tests against Python integers."""

import numpy as np
import pytest

from pnn import build_network, run

STYLES = ["ripple", "lookahead"]


@pytest.mark.parametrize("style", STYLES)
def test_add_8bit_exhaustive(all_pairs, style):
    a, b = all_pairs(8)  # all 65,536 pairs
    s, carry = run("+", a, b, bits=8, style=style)
    np.testing.assert_array_equal(s, (a + b) % 256)
    np.testing.assert_array_equal(carry, (a + b) >> 8)


@pytest.mark.parametrize("style", STYLES)
def test_sub_8bit_exhaustive(all_pairs, style):
    a, b = all_pairs(8)
    d, borrow = run("-", a, b, bits=8, style=style)
    np.testing.assert_array_equal(d, (a - b) % 256)
    np.testing.assert_array_equal(borrow, (a < b).astype(int))


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("style", STYLES)
def test_mul_exhaustive(all_pairs, bits, style):
    a, b = all_pairs(bits)
    p, _ = run("*", a, b, bits=bits, style=style)
    np.testing.assert_array_equal(p, a * b)


@pytest.mark.parametrize("style", STYLES)
def test_compare_8bit_exhaustive(all_pairs, style):
    a, b = all_pairs(8)
    lt, _ = run("<", a, b, bits=8, style=style)
    eq, _ = run("==", a, b, bits=8, style=style)
    np.testing.assert_array_equal(lt, (a < b).astype(int))
    np.testing.assert_array_equal(eq, (a == b).astype(int))


@pytest.mark.parametrize("bits", [1, 2, 3, 5])
@pytest.mark.parametrize("op", ["+", "-", "*", "<", "=="])
def test_small_widths_exhaustive(all_pairs, bits, op):
    a, b = all_pairs(bits)
    for style in STYLES:
        v, flag = run(op, a, b, bits=bits, style=style)
        if op == "+":
            np.testing.assert_array_equal(v + (flag << bits), a + b)
        elif op == "-":
            np.testing.assert_array_equal(v, (a - b) % 2**bits)
            np.testing.assert_array_equal(flag, (a < b).astype(int))
        elif op == "*":
            np.testing.assert_array_equal(v, a * b)
        elif op == "<":
            np.testing.assert_array_equal(v, (a < b).astype(int))
        else:
            np.testing.assert_array_equal(v, (a == b).astype(int))


@pytest.mark.parametrize("style", STYLES)
def test_sigmoid_mode_add_8bit_exhaustive(all_pairs, style):
    """With a steep sigmoid the network is smooth/differentiable and still exact after rounding."""
    a, b = all_pairs(8)
    s, carry = run("+", a, b, bits=8, style=style, mode="sigmoid", steepness=20.0)
    np.testing.assert_array_equal(s, (a + b) % 256)
    np.testing.assert_array_equal(carry, (a + b) >> 8)


def test_sigmoid_outputs_are_close_to_binary(all_pairs):
    from pnn import to_bits

    a, b = all_pairs(8)
    net = build_network("+", 8)
    X = np.concatenate([to_bits(a, 8), to_bits(b, 8)], axis=-1)
    Y = net.forward(X, mode="sigmoid", steepness=20.0)
    assert np.max(np.minimum(Y, 1 - Y)) < 1e-3


def test_16bit_add_random():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**16, 5000)
    b = rng.integers(0, 2**16, 5000)
    for style in STYLES:
        s, carry = run("+", a, b, bits=16, style=style)
        np.testing.assert_array_equal(s + (carry << 16), a + b)


def test_adder_stats():
    st = build_network("+", 8, "ripple").stats()
    assert st["inputs"] == 16 and st["outputs"] == 9
    assert st["layers"] == 9 and st["logic_neurons"] == 16  # 2 neurons per bit
    assert st["max_abs_weight"] == 2
    st2 = build_network("+", 8, "lookahead").stats()
    assert st2["layers"] == 2 and st2["logic_neurons"] == 16


def test_operand_range_checked():
    with pytest.raises(ValueError):
        run("+", 256, 1, bits=8)

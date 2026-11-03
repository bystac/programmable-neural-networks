"""A binary calculator whose every answer is computed by a programmed neural network.

Usage::

    python -m pnn.calc 13 + 29 --bits 8
    python -m pnn.calc 200 - 57
    python -m pnn.calc 12 '*' 13          # quote * and < for your shell
    python -m pnn.calc 7 '<' 9
    python -m pnn.calc 42 == 42 --style lookahead --mode sigmoid
"""

from __future__ import annotations

import argparse
import sys

from .arith import OPS, STYLES, build_network, run

_ALIASES = {"x": "*", "lt": "<", "eq": "==", "=": "==", "plus": "+", "minus": "-", "times": "*"}
_DESC = {
    ("+", "ripple"): "ripple-carry adder",
    ("+", "lookahead"): "threshold carry-lookahead adder",
    ("-", "ripple"): "two's-complement ripple subtractor",
    ("-", "lookahead"): "two's-complement lookahead subtractor",
    ("*", "ripple"): "shift-and-add array multiplier (ripple rows)",
    ("*", "lookahead"): "shift-and-add array multiplier (lookahead rows)",
    ("<", "ripple"): "comparator (subtractor borrow)",
    ("<", "lookahead"): "comparator (one weighted neuron)",
    ("==", "ripple"): "equality (XNOR + AND)",
    ("==", "lookahead"): "equality (two comparators + NOR)",
}


def _bin(v: int, width: int) -> str:
    return "0b" + format(int(v), f"0{width}b")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m pnn.calc",
        description="Binary calculator computed by a neural network with programmed (untrained) weights.",
    )
    p.add_argument("a", type=int)
    p.add_argument("op", help="one of + - * < ==  (aliases: x lt eq)")
    p.add_argument("b", type=int)
    p.add_argument("--bits", type=int, default=8, help="operand width (default 8)")
    p.add_argument("--style", choices=STYLES, default="ripple", help="circuit style (default ripple)")
    p.add_argument("--mode", choices=("step", "sigmoid"), default="step", help="activation (default step)")
    p.add_argument("--steepness", type=float, default=20.0, help="sigmoid steepness (default 20)")
    p.add_argument("--export", metavar="PATH", help="save the network weights (.npz or .json)")
    args = p.parse_args(argv)

    op = _ALIASES.get(args.op, args.op)
    if op not in OPS:
        p.error(f"unknown operator {args.op!r}; use one of + - * < ==")
    n = args.bits
    if not (1 <= n <= 32):
        p.error("--bits must be between 1 and 32")
    for v in (args.a, args.b):
        if not 0 <= v < 2**n:
            p.error(f"operands must be in [0, {2**n - 1}] for --bits {n}")

    net = build_network(op, n, args.style)
    value, flag = run(op, args.a, args.b, n, args.style, args.mode, args.steepness)
    value = int(value)

    print(f"pnn: {args.a} {op} {args.b}   [{n}-bit {_DESC[(op, args.style)]}, {args.mode} activation]")
    print(f"  a      = {args.a:>6}   {_bin(args.a, n)}")
    print(f"  b      = {args.b:>6}   {_bin(args.b, n)}")
    if op == "+":
        expected = args.a + args.b
        full = value + (int(flag) << n)
        print(f"  result = {value:>6}   {_bin(value, n)}   carry={int(flag)}")
        if flag:
            print(f"           (with carry: {full} = {_bin(full, n + 1)})")
        ok = full == expected
    elif op == "-":
        expected = (args.a - args.b) % 2**n
        print(f"  result = {value:>6}   {_bin(value, n)}   borrow={int(flag)}")
        if flag:
            print(f"           (borrow set: as signed, {value} - {2**n} = {value - 2**n})")
        ok = value == expected and int(flag) == int(args.a < args.b)
    elif op == "*":
        expected = args.a * args.b
        print(f"  result = {value:>6}   {_bin(value, 2 * n)}")
        ok = value == expected
    else:
        expected = int(args.a < args.b) if op == "<" else int(args.a == args.b)
        print(f"  result = {'true' if value else 'false':>6}   ({value})")
        ok = value == expected
    print(f"  check  : {'OK' if ok else 'MISMATCH'} (verified against Python integers)")
    s = net.stats()
    print(
        f"network: {s['inputs']} inputs -> {s['outputs']} output{'s' if s['outputs'] != 1 else ''} | {s['layers']} layers | "
        f"{s['neurons']} neurons ({s['logic_neurons']} logic + {s['relay_neurons']} relay)"
    )
    print(
        f"         {s['params']:,} params ({s['nonzero_params']:,} non-zero) | "
        f"max |w| = {s['max_abs_weight']:g} | trained: never"
    )
    if args.export:
        if args.export.endswith(".json"):
            net.save_json(args.export)
        else:
            net.save_npz(args.export)
        print(f"exported weights to {args.export}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

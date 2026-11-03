"""Compile a :class:`~pnn.circuit.Circuit` into a layered :class:`~pnn.network.Network`.

Pipeline:

1. **Prune** - keep only gates reachable from the requested outputs.
2. **Schedule** - give every gate a layer: ``depth = 1 + max(depth of inputs)``
   (inputs are layer 0). This is ASAP levelisation of the DAG.
3. **Relay** - a strictly layered network can only read the previous layer, so
   a value that is needed later than the next layer (a skip connection) is
   carried forward by *identity neurons* ``step(x - 0.5) == x``.
4. **Emit** - write each gate's integer weights and bias into a dense weight
   matrix per layer.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .circuit import Circuit, Wire
from .network import Layer, Network


def compile_circuit(
    circuit: Circuit,
    outputs: Mapping[str, Wire] | Sequence[tuple[str, Wire]],
) -> Network:
    """Compile ``circuit`` into a :class:`Network` computing ``outputs``.

    ``outputs`` maps output names to wires (order is preserved). The network's
    inputs are all of the circuit's declared inputs, in declaration order.
    """
    out_items = list(outputs.items()) if isinstance(outputs, Mapping) else list(outputs)
    if not out_items:
        raise ValueError("need at least one output")
    nodes = circuit.nodes
    input_ids = [n.id for n in nodes if n.kind == "input"]
    out_ids = [w.id for _, w in out_items]

    # 1. prune: nodes reachable from outputs
    reachable: set[int] = set()
    stack = list(out_ids)
    while stack:
        i = stack.pop()
        if i in reachable:
            continue
        reachable.add(i)
        stack.extend(nodes[i].inputs)

    # 2. schedule (node ids are already topologically ordered)
    depth: dict[int, int] = {}
    for i in sorted(reachable):
        n = nodes[i]
        if n.kind == "input":
            depth[i] = 0
        elif n.kind == "const":
            depth[i] = 1  # a bias-only neuron
        else:
            depth[i] = 1 + max(depth[j] for j in n.inputs)
    D = max(1, max(depth[i] for i in out_ids))

    # 3. how long must each value stay alive?  (last layer index it must appear in)
    need_until: dict[int, int] = {}
    for i in reachable:
        for j in nodes[i].inputs:
            need_until[j] = max(need_until.get(j, -1), depth[i] - 1)
    for i in out_ids:
        need_until[i] = D

    # 4. build layers
    pos: list[dict[int, int]] = [{i: k for k, i in enumerate(input_ids)}]
    layers: list[Layer] = []
    for L in range(1, D + 1):
        rows: list[tuple[str, int]] = []  # ("logic"|"relay", node id)
        if L < D:
            rows += [("logic", i) for i in sorted(reachable) if nodes[i].kind != "input" and depth[i] == L]
            rows += [
                ("relay", i)
                for i in sorted(reachable)
                if depth[i] < L <= need_until.get(i, -1)
            ]
        else:  # final layer: exactly one neuron per output, in order
            rows = [("logic" if depth[i] == D else "relay", i) for i in out_ids]
        W = np.zeros((len(rows), len(pos[-1])))
        b = np.zeros(len(rows))
        labels, kinds = [], []
        for r, (kind, i) in enumerate(rows):
            n = nodes[i]
            if kind == "relay":
                W[r, pos[-1][i]] = 1.0
                b[r] = -0.5
                labels.append(f"relay({_label(circuit, i)})")
            elif n.kind == "const":
                b[r] = 0.5 if n.value else -0.5
                labels.append(_label(circuit, i))
            else:
                for j, w in zip(n.inputs, n.weights):
                    W[r, pos[-1][j]] += w
                b[r] = n.bias
                labels.append(_label(circuit, i))
            kinds.append(kind)
        if L == D:
            labels = [name for name, _ in out_items]
        layers.append(Layer(W, b, labels, kinds))
        new_pos: dict[int, int] = {}
        for r, (_, i) in enumerate(rows):
            new_pos.setdefault(i, r)
        pos.append(new_pos)

    return Network(layers, [nodes[i].name for i in input_ids], [name for name, _ in out_items])


def _label(circuit: Circuit, i: int) -> str:
    n = circuit.nodes[i]
    if n.name:
        return n.name
    if n.kind == "const":
        return f"const{n.value}"
    return f"n{i}"

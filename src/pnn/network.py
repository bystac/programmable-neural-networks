"""A plain feed-forward network: dense layers of threshold neurons.

Inference is nothing but ``x <- act(x @ W.T + b)`` per layer, in NumPy.
No Python boolean logic runs at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Layer:
    W: np.ndarray  # shape (n_out, n_in)
    b: np.ndarray  # shape (n_out,)
    labels: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)  # "logic" | "relay"

    @property
    def size(self) -> int:
        return self.W.shape[0]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


class Network:
    """A layered network of neurons ``y = act(W x + b)``.

    Parameters are usually produced by :func:`pnn.compiler.compile_circuit`,
    but any weights can be loaded.
    """

    def __init__(self, layers: list[Layer], input_names: list[str], output_names: list[str]):
        self.layers = layers
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        if layers:
            assert layers[0].W.shape[1] == len(self.input_names)
            assert layers[-1].W.shape[0] == len(self.output_names)

    # -------------------------------------------------------------- inference
    def forward(
        self,
        X,
        mode: str = "step",
        steepness: float = 20.0,
        return_activations: bool = False,
    ):
        """Run the network on a batch of binary input vectors.

        ``mode="step"`` uses the Heaviside step (exact logic).
        ``mode="sigmoid"`` uses ``sigmoid(steepness * z)``: a smooth,
        differentiable network whose outputs round to the same answers.
        """
        A = np.asarray(X, dtype=np.float64)
        squeeze = A.ndim == 1
        if squeeze:
            A = A[None, :]
        if A.shape[1] != len(self.input_names):
            raise ValueError(f"expected {len(self.input_names)} inputs, got {A.shape[1]}")
        acts = [A]
        for layer in self.layers:
            Z = A @ layer.W.T + layer.b
            if mode == "step":
                A = (Z > 0).astype(np.float64)
            elif mode == "sigmoid":
                A = _sigmoid(steepness * Z)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            acts.append(A)
        out = A[0] if squeeze else A
        if return_activations:
            return out, acts
        return out

    __call__ = forward

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        neurons = sum(l.size for l in self.layers)
        relay = sum(k == "relay" for l in self.layers for k in l.kinds)
        params = sum(l.W.size + l.b.size for l in self.layers)
        nonzero = sum(int(np.count_nonzero(l.W)) + int(np.count_nonzero(l.b)) for l in self.layers)
        max_w = max((float(np.abs(l.W).max()) for l in self.layers if l.W.size), default=0.0)
        return {
            "inputs": len(self.input_names),
            "outputs": len(self.output_names),
            "layers": len(self.layers),
            "widths": [l.size for l in self.layers],
            "neurons": neurons,
            "logic_neurons": neurons - relay,
            "relay_neurons": relay,
            "params": params,
            "nonzero_params": nonzero,
            "max_abs_weight": max_w,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"{s['inputs']} inputs -> {s['outputs']} outputs | {s['layers']} layers | "
            f"{s['neurons']} neurons ({s['logic_neurons']} logic + {s['relay_neurons']} relay) | "
            f"{s['params']:,} params ({s['nonzero_params']:,} non-zero) | "
            f"max |w| = {s['max_abs_weight']:g}"
        )

    # ----------------------------------------------------------------- export
    def to_dict(self) -> dict:
        return {
            "format": "pnn-network/1",
            "input_names": self.input_names,
            "output_names": self.output_names,
            "layers": [
                {
                    "W": l.W.tolist(),
                    "b": l.b.tolist(),
                    "labels": l.labels,
                    "kinds": l.kinds,
                }
                for l in self.layers
            ],
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        layers = [
            Layer(
                W=np.asarray(l["W"], dtype=np.float64).reshape(len(l["b"]), -1),
                b=np.asarray(l["b"], dtype=np.float64),
                labels=list(l.get("labels", [])),
                kinds=list(l.get("kinds", [])),
            )
            for l in d["layers"]
        ]
        return cls(layers, d["input_names"], d["output_names"])

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load_json(cls, path: str | Path) -> "Network":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_npz(self, path: str | Path) -> None:
        arrays = {}
        for i, l in enumerate(self.layers):
            arrays[f"W{i}"] = l.W
            arrays[f"b{i}"] = l.b
        meta = {
            "input_names": self.input_names,
            "output_names": self.output_names,
            "labels": [l.labels for l in self.layers],
            "kinds": [l.kinds for l in self.layers],
        }
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> "Network":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            n = len(meta["labels"])
            layers = [
                Layer(f[f"W{i}"], f[f"b{i}"], meta["labels"][i], meta["kinds"][i]) for i in range(n)
            ]
        return cls(layers, meta["input_names"], meta["output_names"])

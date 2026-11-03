"""Programmed vs. trained: 8-bit addition.

Trains a plain NumPy MLP with *exactly the same layer shapes* as the
programmed 2-layer lookahead adder (16 -> H -> 9), using minibatch SGD with
momentum for a bounded wall-clock time, then compares exact-match accuracy on
all 65,536 input pairs with the programmed network (which is never trained).

    python examples/compare_training.py --seconds 30

This is a tiny illustration, not a benchmark: a bigger net, a better
optimiser or more time would get further. The point is that the programmed
weights are exact on step zero.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pnn import build_network, run, to_bits


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="training time budget")
    ap.add_argument("--train-frac", type=float, default=0.5, help="fraction of pairs used for training")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    a, b = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    a, b = a.ravel(), b.ravel()
    X = np.concatenate([to_bits(a, 8), to_bits(b, 8)], axis=1)
    total = a + b
    Y = to_bits(total, 9)

    # programmed network
    prog = build_network("+", 8, "lookahead")
    t0 = time.perf_counter()
    s, c = run("+", a, b, 8, "lookahead")
    prog_acc = np.mean(s + (c << 8) == total)
    prog_t = time.perf_counter() - t0
    widths = prog.stats()["widths"]
    H = widths[0]

    # trained network with the same shape: 16 -> H -> 9
    idx = rng.permutation(len(X))
    n_train = int(args.train_frac * len(X))
    tr, te = idx[:n_train], idx[n_train:]
    W1 = rng.normal(0, 1 / np.sqrt(16), (16, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 1 / np.sqrt(H), (H, 9)); b2 = np.zeros(9)
    vel = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    t0, epochs = time.perf_counter(), 0
    while time.perf_counter() - t0 < args.seconds:
        perm = rng.permutation(tr)
        for i in range(0, len(perm), args.batch):
            bi = perm[i : i + args.batch]
            x, y = X[bi], Y[bi]
            h = np.tanh(x @ W1 + b1)
            p = sigmoid(h @ W2 + b2)
            g2 = (p - y) / len(bi)  # BCE gradient w.r.t. logits
            gh = (g2 @ W2.T) * (1 - h**2)
            grads = (x.T @ gh, gh.sum(0), h.T @ g2, g2.sum(0))
            for k, (prm, g) in enumerate(zip((W1, b1, W2, b2), grads)):
                vel[k] = 0.9 * vel[k] - args.lr * g
                prm += vel[k]
        epochs += 1
    train_t = time.perf_counter() - t0

    def acc(rows):
        p = sigmoid(np.tanh(X[rows] @ W1 + b1) @ W2 + b2) > 0.5
        exact = np.all(p == (Y[rows] > 0.5), axis=1).mean()
        per_bit = (p == (Y[rows] > 0.5)).mean()
        return exact, per_bit

    tr_exact, tr_bit = acc(tr)
    te_exact, te_bit = acc(te)
    all_exact, all_bit = acc(np.arange(len(X)))
    n_params = W1.size + b1.size + W2.size + b2.size
    print(f"8-bit addition, all {len(X):,} pairs. Architecture for both: 16 -> {H} -> 9 ({n_params} params)")
    print(f"programmed (0 s training):      exact-match {prog_acc:7.2%}   (inference {prog_t*1000:.1f} ms)")
    print(
        f"trained    ({train_t:.0f} s SGD, {epochs} epochs on {args.train_frac:.0%} of pairs):"
        f" exact-match {all_exact:7.2%}  (train {tr_exact:.2%}, held-out {te_exact:.2%}, per-bit {all_bit:.2%})"
    )


if __name__ == "__main__":
    main()

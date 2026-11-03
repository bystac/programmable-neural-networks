# Contributing

Thanks for your interest! `pnn` is a small proof of concept, and contributions
that keep it **small, readable and exactly correct** are very welcome.

## Setup

```bash
git clone https://github.com/bystac/programmable-neural-networks.git
cd programmable-neural-networks
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

* **NumPy-only core.** No deep-learning frameworks in `src/pnn`. Optional
  exporters to other frameworks should live behind optional dependencies.
* **Exact by construction.** Every compiled network must be tested. Use
  exhaustive tests wherever the input space allows it (≤ ~2^16 cases runs
  in milliseconds), and use randomised tests against Python ints beyond that.
* **No training in the core.** Weights come from the compiler. Training
  experiments belong in `examples/`, are kept tiny, and are not run in CI.
* **Integer weights, half-integer biases.** This keeps a margin of at least 0.5
  on every neuron, which is what makes the sigmoid mode exact after rounding.
* Keep CI fast and cheap: one Python version, CPU only, seconds not minutes.

## Good first contributions

* Relay-minimising scheduling (ALAP, or an ILP for small circuits).
* New arithmetic blocks: division, shifters, carry-select or Wallace-tree multipliers.
* Exporters (PyTorch / ONNX) behind optional extras.
* Compiling bounded loops or a tiny imperative language.

## Pull requests

1. Fork, then create a feature branch.
2. Add or extend tests. `pytest` must pass at 100%.
3. Explain the idea in the PR description. If it changes network sizes,
   include the before/after numbers from `Network.summary()`.

By contributing you agree that your contributions are licensed under the MIT License.

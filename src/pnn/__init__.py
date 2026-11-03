"""Programmable Neural Networks: compile logic into neural-network weights.

Program the network instead of training it.
"""

from .arith import build_network, from_bits, run, to_bits
from .circuit import Circuit, Wire
from .compiler import compile_circuit
from .expr import compile_expression, compile_program
from .network import Layer, Network

__all__ = [
    "Circuit",
    "Wire",
    "Layer",
    "Network",
    "compile_circuit",
    "compile_expression",
    "compile_program",
    "build_network",
    "run",
    "to_bits",
    "from_bits",
]
__version__ = "0.1.0"

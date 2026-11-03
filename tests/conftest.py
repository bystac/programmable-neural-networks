import itertools

import numpy as np
import pytest


@pytest.fixture(scope="session")
def all_pairs():
    def make(bits):
        a, b = np.meshgrid(np.arange(2**bits), np.arange(2**bits), indexing="ij")
        return a.ravel(), b.ravel()

    return make


def truth_table(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)

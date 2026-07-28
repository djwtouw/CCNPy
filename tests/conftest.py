import numpy as np
import pytest

from ccnpy import generate_dataset


@pytest.fixture
def data():
    """A reproducible small multi-label dataset."""
    return generate_dataset(200, random_state=42)


@pytest.fixture
def fitted(data):
    from ccnpy import CCN
    X, Y = data
    return CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y), X, Y


def make_xy(n=150, m=3, L=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    W = rng.uniform(-2, 2, size=(m, L))
    P = 1 / (1 + np.exp(-(X @ W)))
    Y = (P >= rng.random((n, L))).astype(float)
    return X, Y

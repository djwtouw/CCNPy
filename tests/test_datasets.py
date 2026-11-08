import numpy as np
import pytest

from ccnpy import generate_dataset


def test_shapes_and_binary():
    X, Y = generate_dataset(100, random_state=0)
    assert X.shape == (100, 3)
    assert Y.shape == (100, 4)
    assert set(np.unique(Y)).issubset({0.0, 1.0})


def test_reproducible_with_seed():
    X1, Y1 = generate_dataset(50, random_state=7)
    X2, Y2 = generate_dataset(50, random_state=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(Y1, Y2)


def test_different_seeds_differ():
    _, Y1 = generate_dataset(200, random_state=1)
    _, Y2 = generate_dataset(200, random_state=2)
    assert not np.array_equal(Y1, Y2)


@pytest.mark.parametrize("n", [0, -5, 2.5])
def test_invalid_n(n):
    with pytest.raises(ValueError):
        generate_dataset(n)

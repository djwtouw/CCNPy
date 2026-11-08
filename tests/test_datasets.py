import numpy as np
import pytest

from ccnpy import generate_dataset, load_amphibians


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


def test_load_amphibians():
    data = load_amphibians()
    assert data.data.shape == (189, 25)
    assert data.target.shape == (189, 7)
    assert set(np.unique(data.target)) == {0, 1}
    assert data.feature_names[0] == "PC1"
    assert data.target_names == ["GreenFrog", "BrownFrog", "Toad", "FireToad",
                                 "TreeFrog", "Newt", "CrestNewt"]
    # Prevalence of the rarest and the most common species
    assert data.target.sum(axis=0)[6] == 21
    assert data.target.sum(axis=0)[1] == 148


def test_load_amphibians_return_X_y():
    X, Y = load_amphibians(return_X_y=True)
    data = load_amphibians()
    assert np.array_equal(X, data.data)
    assert np.array_equal(Y, data.target)

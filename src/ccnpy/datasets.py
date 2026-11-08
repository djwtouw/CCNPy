"""Synthetic multi-label data generation."""

import numpy as np
from sklearn.utils import check_random_state


def generate_dataset(n, random_state=None):
    """Generate a synthetic multi-label classification dataset.

    Generates three correlated explanatory variables (compound-symmetry
    covariance, variance 2, covariance 0.4) and four binary labels with
    autoregressive dependence. Each label uses a logistic model with
    coefficients drawn from ``U(-2, 2)`` and dependence on all previous labels
    via coefficients drawn from ``U(-4, 4)``.

    Parameters
    ----------
    n : int
        Number of observations (must be positive).
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.

    Returns
    -------
    X : ndarray of shape (n, 3)
        Explanatory variables.
    Y : ndarray of shape (n, 4)
        Binary multi-label outcomes (values in {0, 1}).
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = check_random_state(random_state)

    cov = np.array([[2.0, 0.4, 0.4],
                    [0.4, 2.0, 0.4],
                    [0.4, 0.4, 2.0]])
    X = rng.multivariate_normal(np.zeros(3), cov, size=n)

    Y = np.zeros((n, 4))
    for i in range(Y.shape[1]):
        beta = rng.uniform(-2.0, 2.0, size=X.shape[1])
        eta = X @ beta
        if i > 0:
            gamma = rng.uniform(-4.0, 4.0, size=i)
            eta = eta + Y[:, :i] @ gamma
        probs = 1.0 / (1.0 + np.exp(-eta))
        Y[:, i] = (rng.random(n) < probs).astype(float)

    return X, Y

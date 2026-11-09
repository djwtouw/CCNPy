"""Multi-label data sets: a synthetic generator and a bundled real data set."""

from importlib.resources import files

import numpy as np
from sklearn.utils import Bunch, check_random_state


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


def load_amphibians(return_X_y=False):
    """Load the amphibians multi-label data set.

    The data describe 189 water reservoirs in Poland. The explanatory variables
    summarize the site: the size and type of the reservoir, the surrounding
    land use, and the proximity of roads and buildings. The seven binary labels
    record whether a given amphibian species is present. Species prevalence
    differs considerably, from 78% for the brown frog down to 11% for the great
    crested newt, which makes the data set a realistic illustration of
    multi-label problems with rare labels.

    The raw data hold 15 site characteristics, 10 of which are categorical.
    After one-hot encoding those, 56 variables remain. Principal component
    analysis is applied to the centered and scaled variables, and the smallest
    number of components explaining at least 80% of the total variance is
    retained. This leaves 25 components at 82%.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, return ``(X, Y)`` instead of a Bunch.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        With the fields

        data : ndarray of shape (189, 25)
            Principal components ``PC1`` to ``PC25``.
        target : ndarray of shape (189, 7)
            Species presence (values in {0, 1}).
        feature_names : list of str
            ``["PC1", ..., "PC25"]``.
        target_names : list of str
            ``GreenFrog`` (green frogs), ``BrownFrog`` (brown frogs), ``Toad``
            (common toad), ``FireToad`` (fire-bellied toad), ``TreeFrog``
            (tree frog), ``Newt`` (common newt), and ``CrestNewt`` (great
            crested newt).
    (X, Y) : tuple of ndarray
        Returned instead of the Bunch when ``return_X_y`` is True.

    References
    ----------
    UCI Machine Learning Repository, Amphibians data set,
    doi:10.24432/C5H90N.

    Blachnik, M., Soltysiak, M., and Dabrowska, D. (2019). Predicting presence
    of amphibian species using features obtained from GIS and satellite
    images. ISPRS International Journal of Geo-Information, 8(3), 123,
    doi:10.3390/ijgi8030123.
    """
    path = files("ccnpy").joinpath("data").joinpath("amphibians.csv")
    with path.open("r") as f:
        names = f.readline().strip().split(",")
        values = np.loadtxt(f, delimiter=",")

    n_features = 25
    X = values[:, :n_features]
    Y = values[:, n_features:].astype(int)

    if return_X_y:
        return X, Y
    return Bunch(data=X, target=Y, feature_names=names[:n_features],
                 target_names=names[n_features:])

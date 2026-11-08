"""Guards that the C++ core behaves as specified.

Pins the predictions from the C++ core to an independent NumPy reference of the
chain formula, so any future change to the core or the binding that alters the
output is caught.
"""

import numpy as np

from ccnpy import CCN
from conftest import make_xy


def _reference_proba(X, b, W, C):
    """NumPy mirror of the core's compute_z_and_p (chain order)."""
    n, L = X.shape[0], b.size
    Z = X @ W + b
    P = np.zeros((n, L))
    for l1 in range(L):
        P[:, l1] = 1.0 / (1.0 + np.exp(-Z[:, l1]))
        for l2 in range(l1 + 1, L):
            Z[:, l2] += C[l2, l1] * P[:, l1]
    return P


def test_predict_matches_numpy_reference():
    X, Y = make_xy(seed=3)
    # Identity label order so chain order == column order.
    m = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)
    ref = _reference_proba(X, m.coef_["b"], m.coef_["W"], m.coef_["C"])
    assert np.allclose(m.predict_proba(X), ref, atol=1e-10)


def test_fitted_values_match_fresh_prediction():
    X, Y = make_xy(seed=4)
    m = CCN(random_state=0).fit(X, Y)
    assert np.allclose(m.fitted_proba_, m.predict_proba(X), atol=1e-12)


def test_param_vector_length():
    X, Y = make_xy(m=5, L=3, seed=5)
    m = CCN(random_state=0).fit(X, Y)
    L, mm = 3, 5
    assert m.coefficients_flat_.size == L + mm * L + (L * L - L) // 2

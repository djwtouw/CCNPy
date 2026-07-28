"""Cross-language parity / regression test.

Fits the model on frozen sample data with a deterministic informed start
(``n_starts=1``, no random restarts) and checks the fitted coefficients and
predicted probabilities against committed reference values, for both the
default label order and a custom one.

The identical fixtures (``parity_*.csv``) and reference values are shipped with
the R package and checked by its parity test with the same hyperparameters and
tolerance. Passing in both languages therefore guards against regressions *and*
proves the two implementations produce the same output. Regenerate the fixtures
with ``tests/fixtures/generate_fixtures.py`` if the algorithm intentionally
changes.
"""

from pathlib import Path

import numpy as np

from ccnpy import CCN

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Must match the R parity test (test-parity.R) and generate_fixtures.py.
Q = 2.0
ALPHA = 0.1
DECIMALS = 5  # required agreement; absolute tolerance 10**-DECIMALS
# 0-based permutation; the R test uses the 1-based equivalent c(4, 1, 3, 2).
LABEL_ORDER = [3, 0, 2, 1]


def _load(name):
    return np.loadtxt(FIXTURES / name, delimiter=",")


def _assert_matches(model, X, coef_file, proba_file):
    atol = 10.0 ** -DECIMALS
    np.testing.assert_allclose(model.coefficients_flat_, _load(coef_file),
                               atol=atol, rtol=0)
    np.testing.assert_allclose(model.predict_proba(X), _load(proba_file),
                               atol=atol, rtol=0)


def test_informed_start_parity():
    X, Y = _load("parity_x.csv"), _load("parity_y.csv")
    model = CCN(q=Q, alpha=ALPHA, n_starts=1).fit(X, Y)
    _assert_matches(model, X, "parity_coef.csv", "parity_proba.csv")


def test_informed_start_parity_custom_label_order():
    X, Y = _load("parity_x.csv"), _load("parity_y.csv")
    model = CCN(q=Q, alpha=ALPHA, n_starts=1, label_order=LABEL_ORDER).fit(X, Y)
    _assert_matches(model, X, "parity_coef_lo.csv", "parity_proba_lo.csv")

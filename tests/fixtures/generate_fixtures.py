"""Regenerate the cross-language parity fixtures.

Run this (``python tests/fixtures/generate_fixtures.py``) only when the
algorithm legitimately changes. It writes CSV files that are committed to
*both* the Python and R packages and used by their parity tests:

    parity_x.csv          sample predictors                (n x m)
    parity_y.csv          sample binary labels             (n x L)
    parity_coef.csv       expected flat coefficients       (n_params,)
    parity_proba.csv      expected predicted probabilities (n x L)
    parity_coef_lo.csv    expected flat coefficients, custom label order
    parity_proba_lo.csv   expected predicted probabilities, custom label order

The data is sampled once here and frozen as CSV, so R and Python read the exact
same inputs (no dependence on either language's RNG). The expected outputs are
produced from deterministic ``n_starts=1`` fits (informed start only), both with
the default label order and with a custom one.

The fixture hyperparameters below MUST match those used in the parity tests.
"""

from pathlib import Path

import numpy as np

from ccnpy import CCN, generate_dataset

# Fixture configuration (keep in sync with the parity tests).
N = 40
Q = 2.0
ALPHA = 0.1
# 0-based permutation for Python; the R parity test uses the 1-based
# equivalent c(4, 1, 3, 2).
LABEL_ORDER = [3, 0, 2, 1]
HERE = Path(__file__).resolve().parent


def main():
    # Sample the data once and freeze it.
    X, Y = generate_dataset(n=N, random_state=0)
    np.savetxt(HERE / "parity_x.csv", X, delimiter=",", fmt="%.12g")
    np.savetxt(HERE / "parity_y.csv", Y, delimiter=",", fmt="%d")

    # Reload from CSV so the golden outputs correspond to the exact values the
    # tests will read (not the full-precision in-memory array).
    X = np.loadtxt(HERE / "parity_x.csv", delimiter=",")
    Y = np.loadtxt(HERE / "parity_y.csv", delimiter=",")

    # Default label order.
    model = CCN(q=Q, alpha=ALPHA, n_starts=1).fit(X, Y)
    np.savetxt(HERE / "parity_coef.csv",
               model.coefficients_flat_, delimiter=",", fmt="%.10f")
    np.savetxt(HERE / "parity_proba.csv",
               model.predict_proba(X), delimiter=",", fmt="%.10f")

    # Custom label order.
    model_lo = CCN(q=Q, alpha=ALPHA, n_starts=1,
                   label_order=LABEL_ORDER).fit(X, Y)
    np.savetxt(HERE / "parity_coef_lo.csv",
               model_lo.coefficients_flat_, delimiter=",", fmt="%.10f")
    np.savetxt(HERE / "parity_proba_lo.csv",
               model_lo.predict_proba(X), delimiter=",", fmt="%.10f")

    print(f"wrote fixtures to {HERE} "
          f"(n={N}, q={Q}, alpha={ALPHA}, label_order={LABEL_ORDER})")


if __name__ == "__main__":
    main()

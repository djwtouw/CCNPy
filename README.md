# CCNPy: Classifier Chain Networks for Multi-Label Classification

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Fit a Classifier Chain Network (CCN) for multi-label binary classification. The
model jointly learns label dependencies through a chain structure and minimizes
an L*q* norm aggregated cross-entropy loss, estimated with a BFGS optimizer with
Wolfe line search implemented in C++ via [pybind11](https://pybind11.readthedocs.io)
and Eigen. CCNPy follows the scikit-learn estimator API, so it composes with
`Pipeline`, `GridSearchCV`, `cross_val_score`, and the rest of the ecosystem.

An R implementation of the same method is available as
[CCNR](https://github.com/djwtouw/CCNR).

More information on the method can be found in the following article:

D.J.W. Touw and M. Van de Velden (2025). *Classifier Chain Networks for
Multi-Label Classification*. Expert Systems with Applications, 286, 128048.
doi: [10.1016/j.eswa.2025.128048](https://doi.org/10.1016/j.eswa.2025.128048)

## Contents
- [Installation](#installation)
- [Example](#example)
- [Cross-validation](#cross-validation)
- [Label order](#label-order)
- [Development](#development)
- [Dependencies and licenses](#dependencies-and-licenses)
- [Citation](#citation)

## Installation

CCNPy is not on PyPI yet, so it is installed from source. Building compiles C++
and therefore requires a **C++17 compiler** (MSVC Build Tools on Windows, the
Xcode command line tools on macOS, or GCC/Clang on Linux). The
[Eigen](https://eigen.tuxfamily.org) headers are supplied automatically in the
ways shown below.

### Directly from GitHub

The reworked version currently lives on the `v0.2` branch:

```bash
pip install "ccnpy @ git+https://github.com/djwtouw/CCNPy@v0.2"
```

pip clones the repository and initializes its submodules, which pulls in the
pinned Eigen headers, then builds the extension. This needs network access to
GitHub and to gitlab.com (where Eigen is hosted).

### From a local clone

Clone with submodules so Eigen is present, then install:

```bash
git clone --branch v0.2 --recurse-submodules https://github.com/djwtouw/CCNPy
cd CCNPy
pip install .
```

If you already cloned without `--recurse-submodules`, fetch Eigen first:

```bash
git submodule update --init
pip install .
```

Alternatively, point the build at an Eigen you already have and skip the
submodule entirely:

```bash
EIGEN_INCLUDE_DIR=/path/to/eigen pip install .
```

## Example

The main entry point is the `CCN` estimator. It behaves like any scikit-learn
classifier: `fit`, `predict`, and `predict_proba`.

```python
import numpy as np
from sklearn.model_selection import train_test_split

from ccnpy import CCN, generate_dataset
from ccnpy.metrics import hamming_accuracy

# Generate sample data: 3 predictors, 4 binary labels
X, Y = generate_dataset(n=1000, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

# Fit a classifier chain network
model = CCN(q=1.0, alpha=0.01, random_state=1).fit(X_train, Y_train)
model.summary()

# Predicted classes and probabilities
Y_hat = model.predict(X_test)
Y_proba = model.predict_proba(X_test)

print(f"Hamming accuracy: {hamming_accuracy(Y_test, Y_hat):.4f}")
```

Learned coefficients are available in `model.coef_` (`"b"`, `"W"`, `"C"`).

Runnable scripts covering every part of the API live in
[`examples/`](examples) (`python examples/example_ccn.py`, and so on). They are
also executed by the test suite, so they stay in sync with the code.

## Cross-validation

Because `CCN` is a standard scikit-learn estimator, you can tune it with the
usual tools:

```python
from sklearn.model_selection import GridSearchCV
from ccnpy import CCN
from ccnpy.model_selection import MultilabelStratifiedKFold
from ccnpy.metrics import get_scorer

search = GridSearchCV(
    CCN(random_state=1),
    param_grid={"q": [1.0, 1.5, 2.0], "alpha": [0.01, 0.1, 1.0]},
    scoring=get_scorer("hamming_accuracy"),
    cv=MultilabelStratifiedKFold(n_splits=5, random_state=1),
).fit(X_train, Y_train)

print(search.best_params_)
```

For convenience, `ccn_cv` mirrors CCNR's `ccn_cv`: it runs the grid search and
refits the best model on the full data, returning the fitted `GridSearchCV`.

```python
from ccnpy import ccn_cv

cv = ccn_cv(X_train, Y_train, q=[1.0, 1.5, 2.0], alpha=[0.01, 0.1, 1.0],
            cv=5, scoring="hamming_accuracy", random_state=1)
Y_hat = cv.predict(X_test)              # uses the best refit model
print(cv.best_params_)
```

Available scorers (see `ccnpy.metrics.SCORERS`): `hamming_accuracy`,
`exact_match`, `micro_f1`, `macro_f1`, `log_likelihood`, `mean_auc`.

## Label order

Earlier labels in the chain are used as features for later ones, so the order
matters. By default the columns of `Y` are chained left to right. A different
order can be passed as a permutation, or as a rule that derives one from the
data:

```python
from ccnpy import CCN, conditional_entropy_matrix, entropy_label_order

H = conditional_entropy_matrix(Y_train)          # entry [i, j] is H(Y_j | Y_i)
order = entropy_label_order(Y_train, method="cebcc1")
model = CCN(q=1.0, alpha=0.01, label_order=order).fit(X_train, Y_train)
```

`entropy_label_order` implements the orderings of Jun et al. (2019), selected
with `method`: `cebcc1` (the default, and the best performer in that paper),
`cebcc2`, `cebcc3`, `cebcc4`, and `ebcc`, which ignores the dependence between
labels and sorts by marginal entropy alone.

`label_order` also accepts any callable `f(X, Y)` returning a permutation, which
is evaluated at the start of every `fit`. Inside cross-validation each fold then
derives its order from its own training rows, so candidate rules can be compared
without the order having seen the validation folds:

```python
def cebcc1_order(X, Y):
    return entropy_label_order(Y, method="cebcc1")

cv = ccn_cv(X_train, Y_train, q=[1.0, 2.0], alpha=[0.01, 0.1], cv=5,
            label_order=[None, cebcc1_order], random_state=1)
print(cv.best_params_["label_order"])
```

Predictions are always returned in the original column order of `Y`, whatever
the chain order. See [`examples/example_label_order.py`](examples/example_label_order.py)
for a fuller walkthrough.

X. Jun, Y. Lu, Z. Lei and D. Guolun (2019). *Conditional entropy based
classifier chains for multi-label classification*. Neurocomputing, 335, 185-194.
doi: [10.1016/j.neucom.2019.01.039](https://doi.org/10.1016/j.neucom.2019.01.039)

## Correspondence with CCNR

| CCNR (R) | CCNPy (Python) |
|----------|----------------|
| `ccn(x, y, q, lambda, ...)` | `CCN(q=..., alpha=..., ...).fit(X, Y)` |
| `predict(fit, type = "class")` | `model.predict(X)` |
| `predict(fit, type = "response")` | `model.predict_proba(X)` |
| `coef(fit)` | `model.coef_` |
| `ccn_cv(...)` | `ccn_cv(...)` or `GridSearchCV` |
| `ccn_hamming_accuracy`, ... | `ccnpy.metrics.*` |
| `generate_dataset(n)` | `generate_dataset(n)` |
| `entropy_label_order(y, method)` | `entropy_label_order(Y, method=...)` |
| `conditional_entropy_matrix(y)` | `conditional_entropy_matrix(Y)` |

The regularization penalty is named `lambda` in R and `alpha` in Python
(`lambda` is a reserved word; `alpha` is the scikit-learn convention).

## Development

The C++ core lives under `ccncpp/`, with the pybind11 binding in
`src/ccnpy/_binding.cpp`.

Eigen is located, in order, from: the `EIGEN_INCLUDE_DIR` environment variable, a
vendored copy at `extern/eigen` (a git submodule), then common system locations.
To work from a clone:

```bash
git submodule update --init   # populates extern/eigen
pip install -e ".[test]"
pytest
```

Wheels for the major platforms are built in CI with
[cibuildwheel](https://cibuildwheel.readthedocs.io).

## Dependencies and licenses

This project uses the Eigen C++ library for linear algebra, licensed under the
[MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/). CCNPy itself is licensed under
the GNU General Public License version 3 (GPLv3); see the `LICENSE` file.

## Citation

If you use this package in your research, please cite:

> Touw, D.J.W. and Van de Velden, M. (2025). Classifier chain networks for
> multi-label classification. *Expert Systems with Applications*, 286, 128048.
> doi: [10.1016/j.eswa.2025.128048](https://doi.org/10.1016/j.eswa.2025.128048)

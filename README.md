# CCNPy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

CCNPy implements the classifier chain network (CCN) for multi-label classification. To cite CCNPy in publications, please use:

D.J.W. Touw and M. van de Velden (2024). Classifier Chain Networks for Multi-Label Classification.

Note that the package is still in an early development stage and that various improvements are planned in the near future. For issues, please use [Github Issues](https://github.com/djwtouw/CCNPy/issues).

## Contents
- [Installation](#installation)
- [Examples](#examples)
- [Dependencies and Licenses](#dependencies-and-licenses)

## Installation
To install CCNPy, clone the repository and open a terminal in the top level directory and run
```bash
pip install .
```
Another option is to install the package directly from GitHub using
```bash
pip install "ccnpy @ git+https://github.com/djwtouw/CCNPy"
```

## Examples
First, generate some multi-labeled data. In this example, three explanatory variables and four labels are generated.
```Python
import numpy as np
from ccnpy import CCN
from ccnpy.metrics import accuracy


def generate_dataset(n):
    # Generate explanatory variables for multi-labeled data
    X = np.random.multivariate_normal(
        np.array([0.0, 0.0, 0.0]),
        np.array([[2.0, 0.4, 0.4], [0.4, 2.0, 0.4], [0.4, 0.4, 2.0]]),
        n
    )

    # Initialize Y
    Y = np.empty((X.shape[0], 4))

    # Compute multi-labeled outcomes
    for i in range(Y.shape[1]):
        # Compute X @ beta, where the elements of beta are uniform [-2, 2]
        Y[:, i] = X @ (np.random.rand(X.shape[1]) * 4 - 2.0)

        # Add effect of previous labels
        if i > 0:
            # The effects are random on the interval [-4, 4]
            Y[:, i] += Y[:, :i] @ (np.random.rand(i) * 8 - 4.0)

        Y[:, i] = 1.0 / (1.0 + np.exp(-Y[:, i]))

    # Transform Y into binary outcomes based on the computed probabilities
    Y = (Y >= np.random.rand(*Y.shape)).astype(float)

    return X, Y


# Set seed and generate training and testing data
np.random.seed(123)
X_train, Y_train = generate_dataset(200)
X_test, Y_test = generate_dataset(1000)
```
To fit a classifier chain network, use the following code.
```Python
# Fit a classifier chain network
model = CCN().fit(X_train, Y_train)

# Make predictions
Y_hat = model.predict(X_test)

# Compute accuracy on test set
print(f"Accuracy: {accuracy(Y_test, Y_hat):.4f}")
```
Cross-validation can be used to select optimal tuning parameters.
```Python
# Perform cross validation to tune the hyperparameters
model_cv = CCN().cv(X_train, Y_train, grid_search_type="random")

# Make predictions
Y_hat_cv = model_cv.predict(X_test)

# Compute accuracy on test set
print(f"Accuracy: {accuracy(Y_test, Y_hat_cv):.4f}")
```

## Dependencies and Licenses

This project uses the Eigen C++ library for linear algebra operations, which is licensed under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) license. The Eigen source files that are included in this repository can be found in the `cpp/include/Eigen/` subdirectory. For more information about Eigen, including documentation and usage examples, please visit the [Eigen website](http://eigen.tuxfamily.org/).

In addition, this project is licensed under the GNU General Public License version 3 (GPLv3). The GPLv3 is a widely-used free software license that requires derivative works to be licensed under the same terms. The full text of the GPLv3 can be found in the `ccmmpy/LICENSE` file.

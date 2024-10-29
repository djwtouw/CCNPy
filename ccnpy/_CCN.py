from copy import deepcopy

import numpy as np
import pandas as pd

from ._ccnpy import ccn_logistic, ccs_logistic
from ._ml_kfold import multilabel_kfold
from .metrics import hamming_loss, macro_F1, micro_F1, negloglik, zero_one_loss


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


class CCN:
    def __init__(
        self,
        q=2,
        alpha=0.01,
        tol=1e-6,
        init="auto",
        best_of=10,
        c1=1e-6,
        c2=0.9,
        logit_threshold=0.5,
        loss_type="cross_entropy",
    ):
        # Simple initializations
        self.q = q
        self.alpha = alpha
        self.tol = tol
        self.init = init
        self.best_of = best_of
        self.logit_threshold = logit_threshold
        self.loss_type = loss_type

        # Initializations that depend on each other
        if not (0 < c1 < c2 < 1):
            raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")
        self._c1 = c1
        self._c2 = c2

        # Fitted variables
        self.b, self.W, self.c, self.loss = None, None, None, None

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value < 1:
            raise ValueError("q must be greater or equal to 1")
        self._q = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("alpha must be nonnegative")
        self._alpha = value

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        if value <= 0:
            raise ValueError("tol must be positive")
        self._tol = value

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, value):
        if value not in ["informed", "random", "auto"]:
            raise ValueError(
                "init must be one of {'informed', 'random', 'auto'}"
            )
        self._init = value

    @property
    def best_of(self):
        return self._best_of

    @best_of.setter
    def best_of(self, value):
        if value < 1:
            raise ValueError("best_of should be positive")
        self._best_of = value

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, value):
        if not (0 < value < self.c2):
            raise ValueError("c1 must satisfy 0 < c1 < c2")
        self._c1 = value

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, value):
        if not (self.c1 < value < 1):
            raise ValueError("c2 must satisfy c1 < c2 < 1")
        self._c2 = value

    @property
    def logit_threshold(self):
        return self._logit_threshold

    @logit_threshold.setter
    def logit_threshold(self, value):
        if not (0 < value < 1):
            raise ValueError(
                "logit_threshold must satisfy 0 < " "logit_threshold < 1"
            )
        self._logit_threshold = value

    @staticmethod
    def _reshape_params(params, m, L):
        # Fill in the parameters for b
        b = params[:L]

        # Initialize W
        W = np.empty((m, L))

        # Fill in the parameters for W
        for label_l in range(L):
            w_idx0 = L + label_l * m
            w_idx1 = w_idx0 + m
            W[:, label_l] = params[w_idx0:w_idx1]

        # Fill in the parameters for c
        c = np.array([])
        if L > 1:
            c = params[-((L * L - L) // 2) :]

        return b, W, c

    def _params_init(self, X, Y, init_type):
        # Number of labels and the number of variables
        L = Y.shape[1]
        m = X.shape[1]

        # Initialize parameter containers
        b = np.empty(L)
        W = np.empty((m, L))
        c = np.empty((L * L - L) // 2)

        # If initialization is informed, use the sequential classifier chain to
        # initialize the parameters
        if init_type == "informed":
            # Create a copy of X to expand with the predicted label
            # probabilities
            X_ccs = X.copy()

            for label_l in range(L):
                # Minimize the sequential version of the classifier chain
                res = ccs_logistic(
                    X_ccs.T,
                    Y[:, label_l],
                    self.alpha,
                    self.c1,
                    self.c2,
                    self.tol,
                    self.loss_type,
                    0.0,
                    0.0,
                )

                # Fill in parameters
                b[label_l] = res[0]
                W[:, label_l] = res[1 : (m + 1)]
                c_idx0 = (label_l * label_l - label_l) // 2
                c_idx1 = c_idx0 + label_l
                c[c_idx0:c_idx1] = res[(m + 1) :]

                # Compute predicted probabilities for label l
                y_l = sigmoid(res[0] + X_ccs @ res[1:])

                # Add the probabilities to the X_ccs matrix
                X_ccs = np.c_[X_ccs, y_l]
        # If initialization is random, initialize the parameters at random, but
        # scale them in order to avoid numerical issues
        elif init_type == "random":
            b = np.random.normal(size=L)
            W = np.random.normal(size=(m, L))
            c = np.random.normal(size=(L * L - L) // 2)

            # Scale the parameter vectors
            for label_l in range(L):
                # Indices for the vector c, these values are used in the
                # prediction for label l
                idx_0 = (label_l * label_l - label_l) // 2
                idx_1 = idx_0 + label_l

                # Norm of the parameters used in predicting label l
                scale = (
                    b[label_l] ** 2
                    + sum(W[:, label_l] ** 2)
                    + sum(c[idx_0:idx_1] ** 2)
                )
                scale = np.sqrt(scale)

                # Scale
                b[label_l] /= scale
                W[:, label_l] /= scale
                c[idx_0:idx_1] /= scale

        return b, W, c

    def _informed_fit(self, X, Y):
        # Initialize parameters
        b, W, c = self._params_init(X, Y, "informed")

        # Proceed to fitting the classifier chain network
        params = np.concatenate((b, W.flatten("F"), c))
        res = ccn_logistic(
            X.T,
            Y.T,
            params,
            self.q,
            self.alpha,
            self.c1,
            self.c2,
            self.tol,
            self.loss_type,
            0.0,
            0.0,
        )

        # Extract parameter estimates
        b, W, c = self._reshape_params(res["params"], X.shape[1], Y.shape[1])

        return b, W, c, res["loss"]

    def _random_fit(self, X, Y):
        # Initialize list holding parameter estimates and losses
        param_estimates = []

        # Perform minimization best_of_n times with random starts
        for i in range(self.best_of):
            # Initialize parameters
            b, W, c = self._params_init(X, Y, "random")

            # Proceed to fitting the classifier chain network
            params = np.concatenate((b, W.flatten("F"), c))
            res = ccn_logistic(
                X.T,
                Y.T,
                params,
                self.q,
                self.alpha,
                self.c1,
                self.c2,
                self.tol,
                self.loss_type,
                0.0,
                0.0,
            )

            # Add result to the ist
            param_estimates.append(res)

        # Find index with the lowest value for the loss
        idx = np.argmin(
            [param_estimates[i]["loss"] for i in range(self.best_of)]
        )

        # Select result
        res = param_estimates[idx]

        # Extract parameter estimates
        b, W, c = self._reshape_params(res["params"], X.shape[1], Y.shape[1])

        return b, W, c, res["loss"]

    def fit(self, X, Y):
        # Best of n random starts
        if self.init == "random":
            self.b, self.W, self.c, self.loss = self._random_fit(X, Y)
        # Informed start using classifier chain
        elif self.init == "informed":
            self.b, self.W, self.c, self.loss = self._informed_fit(X, Y)
        # Best of the above
        elif self.init == "auto":
            b_r, W_r, c_r, loss_r = self.loss = self._random_fit(X, Y)
            b_i, W_i, c_i, loss_i = self.loss = self._informed_fit(X, Y)

            # Select model with the lowest loss
            if loss_r < loss_i:
                self.b, self.W, self.c, self.loss = b_r, W_r, c_r, loss_r
            else:
                self.b, self.W, self.c, self.loss = b_i, W_i, c_i, loss_i

        return self

    def cv(
        self,
        X,
        Y,
        grid=None,
        metric="hamming",
        k=5,
        folds=None,
        grid_search_type="random",
        q_range=(1.0, 5.0),
        alpha_range=(1e-5, 0.5),
        num_draws=40,
    ):
        # Check which type of grid search is done: complete, random, or custom
        if grid_search_type == "random":
            # Initialize dataframe that holds the grid and resulting score
            grid = pd.DataFrame(
                np.zeros((num_draws, 3)), columns=["alpha", "q", "score"]
            )

            # Select random values for q
            grid["q"] = np.random.rand(num_draws)
            grid["q"] *= q_range[1] - q_range[0]
            grid["q"] += q_range[0]

            # Select random values for alpha
            grid["alpha"] = np.random.laplace(
                loc=alpha_range[0], scale=alpha_range[1] / 2, size=num_draws
            )

            # Make sure values for alpha are within the specified range
            for i in range(num_draws):
                while not (
                    alpha_range[0] <= grid["alpha"][i] <= alpha_range[1]
                ):
                    grid.loc[i, "alpha"] = np.random.laplace(
                        loc=alpha_range[0], scale=alpha_range[1] / 2
                    )
        elif grid_search_type == "manual":
            if grid is None:
                raise ValueError("grid needs to be defined")

            # Dataframe that holds grid
            grid = deepcopy(grid)

            # Add column for score
            grid["score"] = 0.0
        else:
            raise ValueError("grid_search_type must be 'manual' or 'random'")

        # Check input for metrics
        if metric not in [
            "hamming",
            "zero_one",
            "micro_F1",
            "macro_F1",
            "nll",
        ]:
            raise ValueError(
                (
                    "metric should be one of 'hamming', 'zero_one', "
                    "'micro_F1', 'macro_F1', or 'nll'"
                )
            )

        # Get folds for k fold cross validation
        if folds is None:
            folds = multilabel_kfold(Y, k)
        else:
            k = len(folds)

        # Do the grid search
        for grid_index in range(grid.shape[0]):
            # Set hyperparameters
            self.alpha = grid.loc[grid_index, "alpha"]
            self.q = grid.loc[grid_index, "q"]

            # Cross validation score
            score = 0

            # Loop through the folds
            for i in range(k):
                # Test and train indices
                test = folds[i]
                train = np.concatenate(folds[:i] + folds[(i + 1) :])

                # Fit the model on the train set
                self.fit(X[train, :], Y[train, :])

                # Make predictions for the test set
                Y_hat = self.predict(X[test, :])
                Y_hat_proba = self.predict_proba(X[test, :])

                # Fold weight for the score
                fold_weight = Y_hat.size / Y.size

                # Add score
                if metric == "hamming":
                    score -= hamming_loss(Y[test, :], Y_hat) * fold_weight
                elif metric == "zero_one":
                    score -= zero_one_loss(Y[test, :], Y_hat) * fold_weight
                elif metric == "macro_F1":
                    score += macro_F1(Y[test, :], Y_hat) * fold_weight
                elif metric == "micro_F1":
                    score += micro_F1(Y[test, :], Y_hat) * fold_weight
                elif metric == "nll":
                    score -= negloglik(Y[test, :], Y_hat_proba) * fold_weight

            # Add the score to the grid with the hyperparameters
            grid.loc[grid_index, "score"] = score

        # Select the best set of hyperparameters
        idx_best = np.argmax(grid["score"])

        # Save cross validation results
        self.cv_results = grid.copy()

        # Set the hyperparameters and train the model on all the data
        self.q = grid.loc[idx_best, "q"]
        self.alpha = grid.loc[idx_best, "alpha"]
        self.fit(X, Y)

        return self

    def predict_proba(self, X):
        # Start with X * W
        P = X @ self.W

        # Transform using the activation function and add the result to
        # subsequent labels
        for l1 in range(self.b.size):
            P[:, l1] = sigmoid(self.b[l1] + P[:, l1])

            for l2 in range(l1 + 1, self.b.size):
                # The parameter that governs the effect of the prediction for
                # label 1 on the prediction for label 2
                c_l2l1 = self.c[(l2 * l2 - l2) // 2 + l1]
                P[:, l2] += c_l2l1 * P[:, l1]

        return P

    def predict(self, X):
        P = self.predict_proba(X)

        return (P >= self.logit_threshold).astype(int)

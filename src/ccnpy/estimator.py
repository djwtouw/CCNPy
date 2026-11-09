"""The Classifier Chain Network estimator.

Follows scikit-learn conventions: parameters are stored verbatim in
``__init__``, validated in ``fit``, and learned attributes carry a trailing
underscore.
"""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

from ._core import ccn_fit, ccn_predict, ccs_fit


def _n_params(m, L):
    """Length of the flat parameter vector for ``m`` features and ``L`` labels."""
    return L + m * L + (L * L - L) // 2


def _column_names(X):
    """String column names of a data frame ``X``, or None when it has none."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(list(columns), dtype=object)
    if not all(isinstance(name, str) for name in names):
        return None
    return names


def _reshape(flat, m, L):
    """Split a flat parameter vector into intercepts ``b``, weights ``W`` and
    the lower-triangular chain matrix ``C`` (all in chain order)."""
    b = np.asarray(flat[:L], dtype=float)

    W = np.empty((m, L))
    for label in range(L):
        start = L + label * m
        W[:, label] = flat[start:start + m]

    C = np.zeros((L, L))
    if L > 1:
        c = flat[-((L * L - L) // 2):]
        idx = 0
        for l2 in range(L):
            for l1 in range(l2):
                C[l2, l1] = c[idx]
                idx += 1
    return b, W, C


class CCN(ClassifierMixin, BaseEstimator):
    """Classifier Chain Network for multi-label binary classification.

    The model jointly learns label dependencies through a chain structure and
    minimizes an Lq-norm aggregated cross-entropy loss with a BFGS optimizer
    (Wolfe line search), implemented in C++.

    Parameters
    ----------
    q : float, default=2.0
        Exponent of the per-observation Lq norm used to aggregate label losses,
        must be >= 1. With ``q = 1`` all labels contribute equally; as ``q``
        grows, larger label losses dominate.
    alpha : float, default=0.01
        Non-negative regularization penalty. The penalty acts on all weights
        except the intercepts, so its effect depends on the scale of the
        explanatory variables. Consider standardizing ``X``, for example with
        :class:`~sklearn.preprocessing.StandardScaler` in a
        :class:`~sklearn.pipeline.Pipeline`.
    tol : float, default=1e-6
        Convergence tolerance for the optimizer.
    n_starts : int, default=1
        Number of optimizer restarts; the fit with the lowest training loss is
        kept. The first start uses an informed initialization from a sequential
        classifier chain; starts 2..n_starts draw from ``N(0, restart_scale^2)``.
    restart_scale : float, default=0.1
        Standard deviation of the random restarts. Ignored when ``n_starts=1``.
    label_order : array-like of int, callable or None, default=None
        Zero-based permutation giving the order in which labels are chained.
        Earlier labels are used as features for later ones. ``None`` uses the
        natural column order. Predictions are always returned in the original
        column order of ``Y``.

        A callable is called as ``label_order(X, Y)`` on the validated training
        data at the start of every ``fit`` and must return such a permutation.
        This defers the choice of order to fit time, so inside cross-validation
        each fold derives its order from its own training rows. See
        :func:`ccnpy.entropy_label_order` for ready made rules. The resolved
        permutation is stored in ``label_order_``.
    c1, c2 : float, default=1e-6, 0.9
        Wolfe line-search constants, must satisfy ``0 < c1 < c2 < 1``.
    random_state : int, RandomState instance or None, default=None
        Controls the random restarts.

    Attributes
    ----------
    coef_ : dict
        ``{"b": (L,), "W": (m, L), "C": (L, L)}`` in chain order.
    coefficients_flat_ : ndarray of shape (n_params,)
        Raw flat parameter vector returned by the optimizer.
    loss_ : float
        Training loss of the retained fit.
    label_order_ : ndarray of shape (L,)
        The chain order actually used.
    n_features_in_ : int
        Number of predictors seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names of ``X`` seen during fit. Defined only when ``X`` has
        string column names, for example a pandas DataFrame.
    n_labels_ : int
        Number of labels seen during fit.
    fitted_proba_ : ndarray of shape (n, L)
        In-sample predicted probabilities, in original column order.
    """

    _parameter_constraints = {
        "q": [Interval(numbers.Real, 1, None, closed="left")],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "tol": [Interval(numbers.Real, 0, None, closed="neither")],
        "n_starts": [Interval(numbers.Integral, 1, None, closed="left")],
        "restart_scale": [Interval(numbers.Real, 0, None, closed="neither")],
        "label_order": ["array-like", callable, None],
        "c1": [Interval(numbers.Real, 0, 1, closed="neither")],
        "c2": [Interval(numbers.Real, 0, 1, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(self, q=2.0, alpha=0.01, tol=1e-6, n_starts=1,
                 restart_scale=0.1, label_order=None, c1=1e-6, c2=0.9,
                 random_state=None):
        self.q = q
        self.alpha = alpha
        self.tol = tol
        self.n_starts = n_starts
        self.restart_scale = restart_scale
        self.label_order = label_order
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state

    # -- sklearn tag declaration (this is a multi-label classifier) ----------
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_label = True
        return tags

    # -- helpers -------------------------------------------------------------
    def _informed_start(self, X, Y):
        """Sequential-chain initialization (the deterministic first start)."""
        n, m = X.shape
        L = Y.shape[1]
        flat = np.zeros(_n_params(m, L))
        X_ccs = X.copy()

        for label in range(L):
            try:
                res = ccs_fit(X_ccs.T, Y[:, label], self.alpha, self.c1,
                              self.c2, self.tol, "cross_entropy")
            except Exception:
                # Fall back to zeros for this link if the sub-fit fails.
                res = np.zeros(m + label + 1)

            flat[label] = res[0]
            flat[L + label * m:L + label * m + m] = res[1:m + 1]
            if label > 0:
                c0 = (label * label - label) // 2
                flat[L + m * L + c0:L + m * L + c0 + label] = res[m + 1:]

            # Append this link's predicted probabilities as a feature.
            eta = res[0] + X_ccs @ res[1:]
            X_ccs = np.c_[X_ccs, 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))]

        return flat

    def _fit_once(self, X, Y, start):
        coef, loss = ccn_fit(X.T, Y.T, start, self.q, self.alpha, self.c1,
                             self.c2, self.tol, "cross_entropy")
        return np.asarray(coef, dtype=float), float(loss)

    # -- API -----------------------------------------------------------------
    def fit(self, X, Y):
        """Fit the classifier chain network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_labels), values in {0, 1}
            A 1D array of shape (n_samples,) is read as a single label.

        Returns
        -------
        self
        """
        self._validate_params()
        if not (0 < self.c1 < self.c2 < 1):
            raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")

        names = _column_names(X)
        X, Y = check_X_y(X, Y, multi_output=True, dtype=np.float64,
                         ensure_2d=True)
        # A 1D Y holds a single label, one entry per observation.
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2 or Y.shape[1] < 1:
            raise ValueError("Y must be a 2D array with at least one column")
        if not np.all(np.isin(Y, (0.0, 1.0))):
            raise ValueError("Y must contain only 0 and 1")

        n, m = X.shape
        L = Y.shape[1]

        # Resolve chain order. A callable is evaluated on this fit's data, so
        # the order follows the training rows it is given.
        if self.label_order is None:
            order = np.arange(L)
        else:
            spec = self.label_order
            source = "label_order"
            if callable(spec):
                spec = spec(X, Y)
                source = "the label_order callable"
            # Only whole numbers qualify: casting 3.5 or "3" to int would
            # silently accept them.
            order = np.asarray(spec).ravel()
            if order.dtype.kind not in "iuf" or not np.all(
                    order == np.round(order)):
                order = np.empty(0, dtype=int)
            order = order.astype(int)
            if order.shape[0] != L or not np.array_equal(np.sort(order),
                                                         np.arange(L)):
                raise ValueError(
                    f"{source} must be a permutation of 0..{L - 1}")
        Y_chain = Y[:, order]

        # Build the list of starting vectors.
        rng = check_random_state(self.random_state)
        starts = [self._informed_start(X, Y_chain)]
        for _ in range(self.n_starts - 1):
            starts.append(rng.normal(scale=self.restart_scale,
                                     size=_n_params(m, L)))

        best_flat, best_loss = None, np.inf
        for i, start in enumerate(starts):
            try:
                flat, loss = self._fit_once(X, Y_chain, start)
            except Exception as exc:  # pragma: no cover - rare optimizer failure
                import warnings
                warnings.warn(f"Start {i + 1} failed: {exc}")
                continue
            if loss < best_loss:
                best_flat, best_loss = flat, loss

        if best_flat is None:
            raise RuntimeError(
                "All optimization starts failed. Check data and hyperparameters.")

        # Store learned state (chain order for coefficients).
        b, W, C = _reshape(best_flat, m, L)
        self.coefficients_flat_ = best_flat
        self.coef_ = {"b": b, "W": W, "C": C}
        self.loss_ = best_loss
        self.label_order_ = order
        self.n_features_in_ = m
        if names is not None:
            self.feature_names_in_ = names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_labels_ = L
        # Per-label binary classes, as multi-label classifiers report them.
        self.classes_ = [np.array([0, 1]) for _ in range(L)]

        # In-sample fitted values, restored to original column order.
        self.fitted_proba_ = self._predict_proba_flat(X)
        return self

    def _predict_proba_flat(self, X):
        """Probabilities via the shared C++ core, in original column order."""
        probs = np.asarray(
            ccn_predict(X.T, self.coefficients_flat_, self.n_labels_)).T
        inv = np.argsort(self.label_order_)
        return probs[:, inv]

    def predict_proba(self, X):
        """Predicted label probabilities, shape (n_samples, n_labels).

        Columns are matched by position. When both the training data and ``X``
        have string column names, these must be identical and in the same
        order.
        """
        check_is_fitted(self)
        names = _column_names(X)
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        trained = getattr(self, "feature_names_in_", None)
        if (trained is not None and names is not None
                and not np.array_equal(names, trained)):
            raise ValueError(
                "The column names of X must match those seen during fit, in "
                "the same order: " + ", ".join(trained))
        return self._predict_proba_flat(X)

    def predict(self, X, threshold=0.5):
        """Predicted binary labels, shape (n_samples, n_labels).

        ``threshold`` may be a scalar applied to all labels or an array of
        length ``n_labels`` for per-label thresholds.
        """
        proba = self.predict_proba(X)
        threshold = np.asarray(threshold)
        if (threshold.dtype.kind not in "iuf"
                or not np.all((threshold > 0) & (threshold < 1))):
            raise ValueError(
                "threshold must be numeric and strictly between 0 and 1")
        if threshold.ndim != 0 and threshold.shape != (self.n_labels_,):
            raise ValueError(
                f"threshold must be a scalar or length {self.n_labels_}")
        return (proba >= threshold).astype(int)

    def score(self, X, Y, sample_weight=None):
        """Mean per-label (Hamming) accuracy."""
        Y = np.atleast_2d(np.asarray(Y))
        return float((self.predict(X) == Y).mean())

    def summary(self):
        """Print a human-readable summary of the fitted model."""
        check_is_fitted(self)
        order = self.label_order_
        print("Classifier Chain Network - Summary")
        print("----------------------------------")
        print(f"Labels (L)    : {self.n_labels_}")
        print(f"Predictors (m): {self.n_features_in_}")
        print(f"Parameters    : {self.coefficients_flat_.size}")
        print(f"\nHyperparameters:\n  q      : {self.q}\n  alpha  : {self.alpha}")
        print(f"\nTraining loss : {round(self.loss_, 6)}")
        print(f"Label order   : {order.tolist()}")
        with np.printoptions(precision=4, suppress=True):
            print("\nIntercepts (b):"); print(self.coef_["b"])
            print("\nWeight matrix (W), m x L:"); print(self.coef_["W"])
            if self.n_labels_ > 1:
                print("\nChain matrix (C), L x L:"); print(self.coef_["C"])

"""Cross-validation utilities for the Classifier Chain Network.

Provides a multi-label stratified k-fold splitter (a drop-in scikit-learn CV
splitter) and :func:`ccn_cv`, a convenience that runs a hyperparameter grid
search by delegating to scikit-learn's :class:`GridSearchCV`.
"""

import numbers

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state

from .estimator import CCN
from .metrics import SCORERS, get_scorer


def _break_tie(candidates, rng):
    """One of ``candidates``, drawn uniformly at random when there are several."""
    if candidates.size == 1:
        return int(candidates[0])
    return int(candidates[rng.randint(candidates.size)])


def _iterative_stratification(y, k, order, rng):
    """Assign each row to one of ``k`` folds by iterative stratification.

    Implements the algorithm of Sechidis, Tsoumakas and Vlahavas (2011), "On
    the Stratification of Multi-Label Data". Each fold is grown so that the
    proportion of positive examples of every label is kept as close as possible
    to the proportion in the full data, considering the rarest labels first.

    Parameters
    ----------
    y : ndarray of shape (n_samples, n_labels)
        Binary label matrix.
    k : int
        Number of folds.
    order : ndarray of shape (n_samples,)
        Order in which rows are visited.
    rng : RandomState
        Source of randomness for breaking ties.

    Returns
    -------
    fold : ndarray of shape (n_samples,)
        Fold index in ``0 .. k - 1`` for every row.

    Notes
    -----
    Every tie (the rarest label, the target fold) is broken at random, as in
    the original paper, so the result depends on ``rng`` as well as on ``y``,
    ``k`` and ``order``.
    """
    n, L = y.shape
    fold = np.full(n, -1, dtype=int)
    assigned = np.zeros(n, dtype=bool)

    # Desired number of examples (overall, and per label) still to place in
    # each fold. Both are float targets that we decrement as rows are placed.
    desired_fold = np.full(k, n / k, dtype=float)
    label_counts = y.sum(axis=0).astype(float)
    desired_label_fold = np.tile(label_counts / k, (k, 1))  # shape (k, L)

    label_remaining = label_counts.copy()  # unplaced positives per label
    remaining = n

    while remaining > 0:
        positive = np.where(label_remaining > 0)[0]
        if positive.size == 0:
            # Only label-free rows are left; spread them by the overall target.
            for idx in order:
                if assigned[idx]:
                    continue
                j = _break_tie(
                    np.where(desired_fold == desired_fold.max())[0], rng)
                fold[idx] = j
                assigned[idx] = True
                desired_fold[j] -= 1.0
            break

        # Rarest remaining label first; ties broken at random.
        scarce = label_remaining[positive]
        label = _break_tie(positive[scarce == scarce.min()], rng)

        for idx in order:
            if assigned[idx] or y[idx, label] == 0:
                continue
            # Fold that most needs an example of this label; ties are settled
            # by the overall target and any remaining tie at random.
            col = desired_label_fold[:, label]
            cand = np.where(col == col.max())[0]
            if cand.size > 1:
                overall = desired_fold[cand]
                cand = cand[overall == overall.max()]
            j = _break_tie(cand, rng)

            fold[idx] = j
            assigned[idx] = True
            row_labels = np.where(y[idx] == 1)[0]
            desired_label_fold[j, row_labels] -= 1.0
            desired_fold[j] -= 1.0
            label_remaining[row_labels] -= 1.0
            remaining -= 1

    return fold


def _multilabel_folds(y, k, random_state):
    """Fold index array (``0 .. k - 1``) for each row via iterative stratification."""
    n = y.shape[0]
    if k > n:
        raise ValueError(
            f"n_splits={k} cannot be greater than the number of samples ({n}).")

    rng = check_random_state(random_state)
    order = rng.permutation(n)
    return _iterative_stratification(y, k, order, rng)


class MultilabelStratifiedKFold:
    """Stratified k-fold cross-validator for multi-label targets.

    Uses iterative stratification (Sechidis, Tsoumakas and Vlahavas, 2011) to
    keep the proportion of positive examples of every label roughly constant
    across folds, so that every fold contains both classes of each label where
    the data allow. Conforms to the scikit-learn splitter interface
    (``split`` / ``get_n_splits``).

    Parameters
    ----------
    n_splits : int, default=5
    random_state : int, RandomState instance or None, default=None
        Controls the order in which rows are visited and the random
        tie-breaking; the split is stratified either way. ``None`` draws a fresh
        split (not reproducible); an integer gives a reproducible one.
    """

    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        y = np.atleast_2d(np.asarray(y))
        n = y.shape[0]
        all_idx = np.arange(n)
        fold = _multilabel_folds(y, self.n_splits, self.random_state)
        for i in range(self.n_splits):
            test = all_idx[fold == i]
            train = all_idx[fold != i]
            yield train, test


def _check_label_orders(candidates):
    """Validate a list of candidate chain orders for the grid."""
    if isinstance(candidates, np.ndarray):
        candidates = list(candidates)
    if not isinstance(candidates, (list, tuple)):
        raise ValueError(
            "label_order must be a sequence of candidate orders, for example "
            "label_order=[None, [2, 0, 1]]")

    for candidate in candidates:
        if candidate is None or callable(candidate):
            continue
        if np.ndim(candidate) != 1:
            raise ValueError(
                "each label_order candidate must be None, a callable, or a "
                "permutation; wrap a single order in a list, for example "
                "label_order=[[2, 0, 1]]")
    return list(candidates)


def ccn_cv(X, Y, q, alpha, cv=5, scoring="hamming_accuracy", label_order=None,
           random_state=None, n_jobs=None, **ccn_kwargs):
    """Grid-search hyperparameters for a Classifier Chain Network.

    Evaluates every ``(q, alpha)`` combination with k-fold cross-validation and
    refits the best model on the full data. Returns a fitted
    :class:`~sklearn.model_selection.GridSearchCV`, so the result exposes
    ``best_estimator_`` (the refit CCN), ``best_params_`` and ``cv_results_``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_labels), values in {0, 1}
    q : float or sequence of float
        Candidate Lq-norm exponents.
    alpha : float or sequence of float
        Candidate regularization penalties.
    cv : int or cross-validation splitter, default=5
        An integer ``k`` uses :class:`MultilabelStratifiedKFold`; a splitter is
        used as given.
    scoring : str or scorer, default="hamming_accuracy"
        A key of :data:`ccnpy.metrics.SCORERS`, any scikit-learn scoring name,
        or a scorer object. Higher is better.
    label_order : sequence or None, default=None
        Candidate chain orders to search over, alongside ``q`` and ``alpha``.
        Each candidate is ``None`` (the column order of ``Y``), a zero-based
        permutation, or a callable ``f(X, Y)`` returning one. A callable is
        evaluated inside every fit, so each fold derives its order from its own
        training rows, which a precomputed permutation cannot do. ``None``
        searches no orders and leaves the choice to ``ccn_kwargs``.
    random_state : int, RandomState instance or None, default=None
        Seed for the default fold generation.
    n_jobs : int or None, default=None
        Passed to :class:`GridSearchCV`.
    **ccn_kwargs
        Extra keyword arguments forwarded to :class:`~ccnpy.CCN` (e.g.
        ``n_starts``).

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted search object, refit on the full data.

    Notes
    -----
    Any :class:`~ccnpy.CCN` parameter can be searched by using
    :class:`GridSearchCV` directly; this function only wires up the common case.

    Examples
    --------
    >>> from ccnpy import ccn_cv, entropy_label_order, generate_dataset
    >>> X, Y = generate_dataset(200, random_state=0)
    >>> def cebcc1(X, Y):
    ...     return entropy_label_order(Y, method="cebcc1")
    >>> search = ccn_cv(X, Y, q=1.0, alpha=0.01, cv=3,
    ...                 label_order=[None, cebcc1])
    >>> search.best_estimator_.label_order_.shape
    (4,)
    """
    q = np.atleast_1d(q).astype(float).tolist()
    alpha = np.atleast_1d(alpha).astype(float).tolist()
    param_grid = {"q": q, "alpha": alpha}
    if label_order is not None:
        param_grid["label_order"] = _check_label_orders(label_order)

    if isinstance(cv, numbers.Integral):
        cv = MultilabelStratifiedKFold(n_splits=int(cv),
                                       random_state=random_state)
    if isinstance(scoring, str) and scoring in SCORERS:
        scoring = get_scorer(scoring)

    search = GridSearchCV(
        CCN(random_state=random_state, **ccn_kwargs),
        param_grid=param_grid,
        scoring=scoring, cv=cv, refit=True, n_jobs=n_jobs,
    )
    search.fit(X, Y)
    return search


__all__ = ["MultilabelStratifiedKFold", "ccn_cv"]

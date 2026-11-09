import numpy as np
import pytest

from ccnpy import CCN, ccn_cv, generate_dataset
from ccnpy.model_selection import (
    MultilabelStratifiedKFold,
    _iterative_stratification,
    _multilabel_folds,
)

from conftest import make_xy


def test_splitter_partitions_indices():
    X, Y = generate_dataset(200, random_state=0)
    skf = MultilabelStratifiedKFold(n_splits=5)  # default: a fresh split
    assert skf.get_n_splits() == 5
    seen_test = []
    for train, test in skf.split(X, Y):
        assert len(np.intersect1d(train, test)) == 0
        assert len(train) + len(test) == X.shape[0]
        seen_test.append(test)
    all_test = np.concatenate(seen_test)
    # Every observation is in exactly one test fold.
    assert np.array_equal(np.sort(all_test), np.arange(X.shape[0]))


def test_splitter_folds_have_label_variation():
    X, Y = generate_dataset(200, random_state=1)
    skf = MultilabelStratifiedKFold(n_splits=5, random_state=1)
    for _, test in skf.split(X, Y):
        col_sums = Y[test].sum(axis=0)
        assert np.all(col_sums > 0)
        assert np.all(col_sums < len(test))


def test_splitter_keeps_label_proportions():
    X, Y = generate_dataset(300, random_state=3)
    skf = MultilabelStratifiedKFold(n_splits=5, random_state=3)
    overall = Y.mean(axis=0)
    for _, test in skf.split(X, Y):
        # Each fold's per-label prevalence is close to the whole-data prevalence.
        assert np.all(np.abs(Y[test].mean(axis=0) - overall) < 0.1)


def test_splitter_reproducible_with_random_state():
    X, Y = generate_dataset(200, random_state=4)
    a = list(MultilabelStratifiedKFold(n_splits=5, random_state=0).split(X, Y))
    b = list(MultilabelStratifiedKFold(n_splits=5, random_state=0).split(X, Y))
    for (_, ta), (_, tb) in zip(a, b):
        assert np.array_equal(ta, tb)


def test_splitter_random_state_changes_assignment():
    X, Y = generate_dataset(200, random_state=5)
    # Different seeds give a different (but still valid) assignment.
    assert not np.array_equal(_multilabel_folds(Y, 5, 0),
                              _multilabel_folds(Y, 5, 1))


def test_splitter_rejects_too_many_splits():
    with pytest.raises(ValueError):
        _multilabel_folds(np.array([[1, 0], [0, 1], [1, 1]]), 5, None)


@pytest.mark.parametrize("n_splits", [1, 0, 2.5])
def test_splitter_rejects_invalid_n_splits(n_splits):
    X, Y = generate_dataset(30, random_state=0)
    with pytest.raises(ValueError, match="at least 2"):
        next(MultilabelStratifiedKFold(n_splits).split(X, Y))


@pytest.mark.parametrize("bad", [2.0, 3.0, np.nan])
def test_splitter_rejects_non_binary_y(bad):
    X, Y = generate_dataset(30, random_state=0)
    Y = np.where(Y == 1, bad, 0.0)
    with pytest.raises(ValueError):
        next(MultilabelStratifiedKFold(3).split(X, Y))


def test_splitter_reads_one_dimensional_y_as_a_single_label():
    X, Y = generate_dataset(60, random_state=0)
    a = MultilabelStratifiedKFold(3, random_state=1).split(X, Y[:, 0])
    b = MultilabelStratifiedKFold(3, random_state=1).split(X, Y[:, :1])
    for (_, test_a), (_, test_b) in zip(a, b):
        assert np.array_equal(test_a, test_b)


def test_ccn_cv_rejects_non_binary_y():
    X, Y = make_xy(n=60, seed=1)
    with pytest.raises(ValueError, match="only 0 and 1"):
        ccn_cv(X, 2 * Y, q=1.0, alpha=0.1, cv=3)


def test_ties_are_broken_randomly():
    """Ties vary the split even when the visiting order is held fixed."""
    # All candidates stay tied, because every row carries every label.
    Y = np.ones((40, 2))
    order = np.arange(len(Y))
    a = _iterative_stratification(Y, 5, order, np.random.RandomState(0))
    b = _iterative_stratification(Y, 5, order, np.random.RandomState(1))
    assert not np.array_equal(a, b)
    # Both are still valid, balanced partitions.
    for folds in (a, b):
        assert np.array_equal(np.bincount(folds, minlength=5), np.full(5, 8))


def test_ccn_cv_selects_from_grid():
    X, Y = make_xy(n=200, seed=2)
    search = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01, 0.1], cv=3,
                    scoring="hamming_accuracy", random_state=0)
    assert search.best_params_["q"] in (1.0, 2.0)
    assert search.best_params_["alpha"] in (0.01, 0.1)
    assert isinstance(search.best_estimator_, CCN)
    # 2 x 2 grid evaluated.
    assert len(search.cv_results_["params"]) == 4
    assert np.isfinite(search.best_score_)


def test_ccn_cv_probability_scorer():
    X, Y = make_xy(n=200, seed=3)
    search = ccn_cv(X, Y, q=[1.0], alpha=[0.05, 0.1], cv=3,
                    scoring="mean_auc", random_state=0)
    assert np.isfinite(search.best_score_)


def test_ccn_cv_forwards_ccn_kwargs():
    X, Y = make_xy(n=150, seed=4)
    search = ccn_cv(X, Y, q=[1.0], alpha=[0.1], cv=3, n_starts=2,
                    random_state=0)
    assert search.best_estimator_.n_starts == 2


def test_ccn_cv_does_not_select_a_grid_point_with_a_failed_fit():
    from sklearn.exceptions import FitFailedWarning
    from sklearn.model_selection import PredefinedSplit
    X, Y = make_xy(n=40, m=2, L=2, seed=5)
    # Unequal folds make the training size identify the fold, so this order
    # fails in the first fold only
    cv = PredefinedSplit(np.repeat([0, 1, 2], [10, 15, 15]))

    def flaky(X, Y):
        if X.shape[0] == 30:
            raise RuntimeError("fails in fold 0")
        return np.arange(Y.shape[1])

    with pytest.warns(FitFailedWarning, match="fits failed"), \
            pytest.warns(UserWarning, match="non-finite"):
        search = ccn_cv(X, Y, q=1.0, alpha=0.01, cv=cv,
                        label_order=[None, flaky])
    scores = search.cv_results_["mean_test_score"]
    assert np.isnan(scores[1])
    assert search.best_params_["label_order"] is None


def test_ccn_cv_rejects_neg_log_loss():
    X, Y = make_xy(n=60, seed=1)
    with pytest.raises(ValueError, match="log_likelihood"):
        ccn_cv(X, Y, q=1.0, alpha=0.1, cv=3, scoring="neg_log_loss")


def test_ccn_cv_stops_when_no_grid_point_can_be_scored():
    from ccnpy.metrics import hamming_accuracy
    X, Y = make_xy(n=60, seed=1)
    # A plain metric function is not a scorer, so every fold scores NaN.
    with pytest.warns(UserWarning), \
            pytest.raises(ValueError, match="make_scorer"):
        ccn_cv(X, Y, q=1.0, alpha=[0.01, 0.1], cv=3,
               scoring=hamming_accuracy)


def test_ccn_cv_stops_when_every_fit_fails():
    X, Y = make_xy(n=40, m=2, L=2, seed=5)

    def broken(X, Y):
        raise RuntimeError("always fails")

    with pytest.raises(ValueError, match="fits failed"):
        ccn_cv(X, Y, q=1.0, alpha=0.01, cv=3, label_order=[broken])

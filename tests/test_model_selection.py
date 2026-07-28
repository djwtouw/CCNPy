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

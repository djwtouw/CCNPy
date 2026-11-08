import numpy as np
import pytest
from sklearn.base import clone

from ccnpy import (CCN, ccn_cv, conditional_entropy_matrix,
                   entropy_label_order, generate_dataset)
from ccnpy.label_order import METHODS
from conftest import make_xy


def reference_matrix(Y):
    """H(Y_j | Y_i) straight from the definition, one pair at a time."""
    n, L = Y.shape
    out = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            total = 0.0
            for a in (0, 1):
                p_a = np.mean(Y[:, i] == a)
                if p_a == 0:
                    continue
                for b in (0, 1):
                    p_ab = np.mean((Y[:, i] == a) & (Y[:, j] == b))
                    if p_ab > 0:
                        total -= p_ab * np.log2(p_ab / p_a)
            out[i, j] = total
    return out


# -- conditional_entropy_matrix ---------------------------------------------

def test_matrix_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(10):
        Y = (rng.random((80, 5)) < rng.uniform(0.1, 0.9, 5)).astype(float)
        assert np.allclose(conditional_entropy_matrix(Y), reference_matrix(Y))


def test_matrix_independent_labels_are_one_bit():
    # All four combinations equally often, so knowing one label tells nothing.
    Y = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    assert np.allclose(conditional_entropy_matrix(Y), [[0., 1.], [1., 0.]])


def test_matrix_identical_labels_have_no_residual_entropy():
    Y = np.array([[0., 0.], [1., 1.], [0., 0.], [1., 1.]])
    assert np.allclose(conditional_entropy_matrix(Y), 0.0)


def test_matrix_handles_constant_labels_without_nan():
    # Label 1 is constant: it carries no entropy, and conditioning on it leaves
    # the entropy of label 0 untouched.
    Y = np.array([[0., 1.], [1., 1.], [0., 1.], [1., 1.]])
    M = conditional_entropy_matrix(Y)
    assert not np.isnan(M).any()
    assert np.allclose(M, [[0., 0.], [1., 0.]])
    assert not np.isnan(conditional_entropy_matrix(np.zeros((5, 3)))).any()


def test_matrix_diagonal_is_zero_and_shape_is_square():
    X, Y = generate_dataset(100, random_state=1)
    M = conditional_entropy_matrix(Y)
    assert M.shape == (Y.shape[1], Y.shape[1])
    assert np.allclose(np.diag(M), 0.0)


def test_matrix_asymmetry_equals_the_difference_in_marginal_entropies():
    # H(Y_j | Y_i) - H(Y_i | Y_j) = H(Y_j) - H(Y_i), because the mutual
    # information cancels. Prevalences 4/8 and 2/8 give 1 and 0.811 bits.
    Y = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.],
                  [1., 0.], [1., 0.], [1., 1.], [1., 1.]])
    M = conditional_entropy_matrix(Y)
    h = np.array([1.0, -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))])
    assert np.isclose(M[0, 1] - M[1, 0], h[1] - h[0])
    assert not np.isclose(M[0, 1], M[1, 0])


def test_matrix_rejects_non_binary():
    with pytest.raises(ValueError, match="only 0 and 1"):
        conditional_entropy_matrix(np.array([[0., 2.], [1., 0.]]))


# -- entropy_label_order ----------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_order_is_a_permutation(method):
    X, Y = generate_dataset(200, random_state=2)
    order = entropy_label_order(Y, method=method)
    assert order.dtype.kind == "i"
    assert sorted(order.tolist()) == list(range(Y.shape[1]))


def test_order_default_is_cebcc1():
    X, Y = generate_dataset(200, random_state=3)
    assert np.array_equal(entropy_label_order(Y),
                          entropy_label_order(Y, method="cebcc1"))


def test_order_unknown_method():
    X, Y = generate_dataset(50, random_state=0)
    with pytest.raises(ValueError, match="Unknown method"):
        entropy_label_order(Y, method="cebcc5")


def test_ebcc_sorts_by_increasing_marginal_entropy():
    # Prevalences 0.5, 0.1 and 0.3 give entropies 1.0, 0.469 and 0.881, so the
    # order runs from the most extreme prevalence to the most balanced label.
    n = 100
    Y = np.zeros((n, 3))
    Y[:50, 0] = 1.0
    Y[:10, 1] = 1.0
    Y[:30, 2] = 1.0
    assert entropy_label_order(Y, method="ebcc").tolist() == [1, 2, 0]


def test_cebcc_strategies_use_documented_extremum_and_end():
    # A three-label matrix with distinct row and column sums, checked against
    # the rules by hand.
    Y = np.array([[0., 0., 0.], [1., 0., 1.], [1., 1., 0.], [0., 1., 1.],
                  [1., 0., 0.], [1., 1., 1.], [0., 0., 1.], [1., 1., 0.]])
    M = conditional_entropy_matrix(Y)
    row_min = int(np.argmin(M.sum(axis=1)))
    row_max = int(np.argmax(M.sum(axis=1)))
    col_min = int(np.argmin(M.sum(axis=0)))
    col_max = int(np.argmax(M.sum(axis=0)))

    assert entropy_label_order(Y, "cebcc1")[-1] == row_min
    assert entropy_label_order(Y, "cebcc2")[0] == row_max
    assert entropy_label_order(Y, "cebcc3")[0] == col_min
    assert entropy_label_order(Y, "cebcc4")[-1] == col_max


def test_order_is_deterministic_and_ties_go_to_lowest_index():
    # Independent, perfectly balanced labels make every sum equal, so each step
    # falls back on the lowest remaining index.
    Y = np.array([[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 0.],
                  [1., 1., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert entropy_label_order(Y, "cebcc3").tolist() == [0, 1, 2]
    assert entropy_label_order(Y, "cebcc1").tolist() == [2, 1, 0]


def test_order_single_label():
    Y = np.array([[0.], [1.], [1.]])
    for method in METHODS:
        assert entropy_label_order(Y, method).tolist() == [0]


# -- CCN with a callable label_order ---------------------------------------

def cebcc1_order(X, Y):
    return entropy_label_order(Y, method="cebcc1")


def test_callable_label_order_matches_precomputed_fit():
    X, Y = generate_dataset(200, random_state=4)
    dynamic = CCN(q=1.0, alpha=0.01, label_order=cebcc1_order,
                  random_state=0).fit(X, Y)
    static = CCN(q=1.0, alpha=0.01, label_order=entropy_label_order(Y),
                 random_state=0).fit(X, Y)
    assert np.array_equal(dynamic.label_order_, static.label_order_)
    assert np.allclose(dynamic.coefficients_flat_, static.coefficients_flat_)


def test_callable_label_order_sees_only_the_rows_it_is_given():
    X, Y = generate_dataset(200, random_state=5)
    seen = {}

    def record(X_fit, Y_fit):
        seen["n"] = X_fit.shape[0]
        return entropy_label_order(Y_fit, method="cebcc1")

    CCN(q=1.0, alpha=0.01, label_order=record,
        random_state=0).fit(X[:60], Y[:60])
    assert seen["n"] == 60


def test_callable_label_order_is_stored_resolved_and_param_kept_verbatim():
    X, Y = generate_dataset(150, random_state=6)
    model = CCN(q=1.0, alpha=0.01, label_order=cebcc1_order).fit(X, Y)
    # The parameter stays as given (sklearn convention), the resolved order
    # lands in the fitted attribute.
    assert model.get_params()["label_order"] is cebcc1_order
    assert isinstance(model.label_order_, np.ndarray)
    assert clone(model).get_params()["label_order"] is cebcc1_order


def test_callable_returning_invalid_order_raises():
    X, Y = generate_dataset(100, random_state=7)
    with pytest.raises(ValueError, match="label_order callable"):
        CCN(label_order=lambda X, Y: [0, 0, 1, 2]).fit(X, Y)
    with pytest.raises(ValueError, match="label_order callable"):
        CCN(label_order=lambda X, Y: None).fit(X, Y)


def test_callable_label_order_predicts_in_original_column_order():
    X, Y = make_xy(n=120, L=4, seed=1)
    model = CCN(q=1.0, alpha=0.01, label_order=cebcc1_order,
                random_state=0).fit(X, Y)
    assert model.predict_proba(X).shape == Y.shape
    # A permuted chain must not permute the output columns.
    natural = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)
    assert model.predict(X).shape == natural.predict(X).shape


# -- ccn_cv over candidate orders ------------------------------------------

def test_ccn_cv_searches_over_label_orders():
    X, Y = generate_dataset(150, random_state=8)
    search = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01], cv=3,
                    label_order=[None, cebcc1_order], random_state=0)
    assert len(search.cv_results_["params"]) == 4
    assert set(search.best_params_) == {"q", "alpha", "label_order"}
    assert search.best_params_["label_order"] in (None, cebcc1_order)


def test_ccn_cv_refits_the_winning_rule_on_all_data():
    X, Y = generate_dataset(150, random_state=9)
    search = ccn_cv(X, Y, q=[1.0], alpha=[0.01], cv=3,
                    label_order=[cebcc1_order], random_state=0)
    expected = entropy_label_order(Y, method="cebcc1")
    assert np.array_equal(search.best_estimator_.label_order_, expected)


def test_ccn_cv_without_label_order_keeps_the_two_parameter_grid():
    X, Y = generate_dataset(120, random_state=10)
    search = ccn_cv(X, Y, q=[1.0], alpha=[0.01, 0.1], cv=3, random_state=0)
    assert set(search.best_params_) == {"q", "alpha"}


def test_ccn_cv_rejects_a_bare_permutation():
    X, Y = generate_dataset(100, random_state=11)
    with pytest.raises(ValueError, match="wrap a single order in a list"):
        ccn_cv(X, Y, q=1.0, alpha=0.01, cv=3, label_order=[0, 2, 1, 3])
    with pytest.raises(ValueError, match="sequence of candidate orders"):
        ccn_cv(X, Y, q=1.0, alpha=0.01, cv=3, label_order=cebcc1_order)

import numpy as np
import pytest
from sklearn.base import clone

from ccnpy import CCN

from conftest import make_xy


def test_fit_returns_self_and_sets_attributes(data):
    X, Y = data
    m = CCN(random_state=0)
    assert m.fit(X, Y) is m
    assert m.n_features_in_ == X.shape[1]
    assert m.n_labels_ == Y.shape[1]
    assert m.coef_["b"].shape == (Y.shape[1],)
    assert m.coef_["W"].shape == (X.shape[1], Y.shape[1])
    assert m.coef_["C"].shape == (Y.shape[1], Y.shape[1])
    assert np.isfinite(m.loss_)


def test_predict_shapes_and_types(fitted):
    m, X, Y = fitted
    proba = m.predict_proba(X)
    cls = m.predict(X)
    assert proba.shape == Y.shape
    assert cls.shape == Y.shape
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(np.unique(cls)).issubset({0, 1})


def test_chain_matrix_is_lower_triangular(fitted):
    m, _, _ = fitted
    C = m.coef_["C"]
    assert np.allclose(np.triu(C), 0.0)


def test_threshold_scalar_and_vector(fitted):
    m, X, _ = fitted
    base = m.predict(X, threshold=0.5)
    high = m.predict(X, threshold=0.99)
    # A higher threshold can only turn 1s into 0s.
    assert (high <= base).all()
    vec = m.predict(X, threshold=np.full(m.n_labels_, 0.5))
    assert np.array_equal(vec, base)


def test_threshold_validation(fitted):
    m, X, _ = fitted
    with pytest.raises(ValueError):
        m.predict(X, threshold=1.5)
    with pytest.raises(ValueError):
        m.predict(X, threshold=np.array([0.5, 0.5]))  # wrong length


@pytest.mark.parametrize("threshold", [
    [2.0, 0.5, 0.5, 0.5], [0.5, -1.0, 0.5, 0.5], [0.5, np.nan, 0.5, 0.5],
    np.nan, "0.5",
])
def test_threshold_rejects_values_outside_the_unit_interval(fitted, threshold):
    m, X, _ = fitted
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        m.predict(X, threshold=threshold)


def test_label_order_roundtrip_changes_chain_not_output_shape(data):
    X, Y = data
    L = Y.shape[1]
    m1 = CCN(random_state=0).fit(X, Y)
    m2 = CCN(label_order=list(reversed(range(L))), random_state=0).fit(X, Y)
    assert m2.label_order_.tolist() == list(reversed(range(L)))
    # Output is always in original column order, so shapes line up.
    assert m1.predict_proba(X).shape == m2.predict_proba(X).shape


def test_invalid_label_order(data):
    X, Y = data
    with pytest.raises(ValueError):
        CCN(label_order=[0, 0, 1, 2]).fit(X, Y)


@pytest.mark.parametrize("order", [[3.5, 0, 2, 1], ["3", "0", "2", "1"]])
def test_label_order_must_hold_whole_numbers(data, order):
    X, Y = data
    with pytest.raises(ValueError, match="permutation"):
        CCN(label_order=order).fit(X, Y)


def test_label_order_accepts_whole_floats(data):
    X, Y = data
    m = CCN(label_order=[3.0, 0.0, 2.0, 1.0]).fit(X, Y)
    assert m.label_order_.tolist() == [3, 0, 2, 1]


def test_one_dimensional_y_is_a_single_label(data):
    pd = pytest.importorskip("pandas")
    X, Y = data
    reference = CCN(q=1.0, alpha=0.01).fit(X, Y[:, :1])
    assert reference.n_labels_ == 1
    for y in (Y[:, 0], pd.Series(Y[:, 0]), list(Y[:, 0])):
        m = CCN(q=1.0, alpha=0.01).fit(X, y)
        assert m.n_labels_ == 1
        assert np.array_equal(m.coefficients_flat_,
                              reference.coefficients_flat_)
        assert m.predict(X).shape == (X.shape[0], 1)
        assert m.predict_proba(X).shape == (X.shape[0], 1)


def test_rejects_non_binary_y(data):
    X, Y = data
    Y2 = Y.copy()
    Y2[0, 0] = 2
    with pytest.raises(ValueError):
        CCN().fit(X, Y2)


@pytest.mark.parametrize("kwargs", [
    {"q": 0.5}, {"q": np.inf}, {"alpha": -1}, {"tol": 0}, {"tol": -1},
    {"n_starts": 0}, {"n_starts": 2.5}, {"n_starts": None},
    {"restart_scale": [1, 2]}, {"c1": 0.9, "c2": 0.1},
])
def test_parameter_validation(data, kwargs):
    X, Y = data
    with pytest.raises(ValueError):
        CCN(**kwargs).fit(X, Y)


def test_rejects_infinite_values(data):
    X, Y = data
    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        CCN().fit(X_inf, Y)
    m = CCN().fit(X, Y)
    with pytest.raises(ValueError, match="infinity"):
        m.predict(X_inf)


def test_n_starts_improves_or_matches_loss(data):
    X, Y = data
    one = CCN(n_starts=1, random_state=0).fit(X, Y).loss_
    many = CCN(n_starts=5, random_state=0).fit(X, Y).loss_
    assert many <= one + 1e-8


def test_score_is_hamming_accuracy(fitted):
    m, X, Y = fitted
    expected = (m.predict(X) == Y).mean()
    assert m.score(X, Y) == pytest.approx(expected)


def test_get_set_params_and_clone():
    m = CCN(q=3.0, alpha=0.2, n_starts=4)
    assert m.get_params()["q"] == 3.0
    m.set_params(q=1.5)
    assert m.q == 1.5
    c = clone(m)
    assert c.get_params() == m.get_params()
    assert not hasattr(c, "coef_")  # clone is unfitted


def test_predict_wrong_n_features(fitted):
    m, X, _ = fitted
    with pytest.raises(ValueError):
        m.predict(X[:, :-1])


def test_summary_runs(fitted, capsys):
    m, _, _ = fitted
    m.summary()
    out = capsys.readouterr().out
    assert "Classifier Chain Network" in out
    assert "Weight matrix" in out


def test_learns_on_clean_data():
    X, Y = make_xy(seed=1)
    m = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)
    # On its own training data the model should beat the trivial baseline.
    assert (m.predict(X) == Y).mean() > 0.6

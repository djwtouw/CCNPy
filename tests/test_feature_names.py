import numpy as np
import pytest

from ccnpy import CCN

from conftest import make_xy

pd = pytest.importorskip("pandas")


@pytest.fixture
def frames():
    X, Y = make_xy(n=100, m=3, L=3, seed=1)
    return pd.DataFrame(X, columns=["u", "v", "w"]), Y


def test_feature_names_are_stored(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y)
    assert list(m.feature_names_in_) == ["u", "v", "w"]


def test_no_feature_names_for_arrays(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y).fit(X.to_numpy(), Y)
    assert not hasattr(m, "feature_names_in_")


def test_same_column_names_are_accepted(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y)
    np.testing.assert_allclose(m.predict_proba(X), m.fitted_proba_)


def test_column_names_in_a_different_order_are_rejected(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y)
    with pytest.raises(ValueError, match="same order: u, v, w"):
        m.predict(X[["w", "u", "v"]])


def test_different_column_names_are_rejected(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y)
    with pytest.raises(ValueError, match="column names of X"):
        m.predict(X.rename(columns={"v": "z"}))


def test_columns_are_matched_by_position_without_names(frames):
    X, Y = frames
    m = CCN(random_state=0).fit(X, Y)
    np.testing.assert_allclose(m.predict_proba(X.to_numpy()),
                               m.predict_proba(X))
    m_arr = CCN(random_state=0).fit(X.to_numpy(), Y)
    np.testing.assert_allclose(m_arr.predict_proba(X[["w", "u", "v"]]),
                               m_arr.predict_proba(X.to_numpy()[:, [2, 0, 1]]))

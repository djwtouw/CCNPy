import numpy as np
import pytest

from ccnpy import metrics


@pytest.fixture
def labels():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    return y_true, y_pred


def test_hamming_accuracy(labels):
    y_true, y_pred = labels
    # 7 of 9 entries match.
    assert metrics.hamming_accuracy(y_true, y_pred) == pytest.approx(7 / 9)


def test_exact_match(labels):
    y_true, y_pred = labels
    # Only the middle row matches fully.
    assert metrics.exact_match(y_true, y_pred) == pytest.approx(1 / 3)


def test_micro_macro_f1_bounds(labels):
    y_true, y_pred = labels
    for fn in (metrics.micro_f1, metrics.macro_f1):
        v = fn(y_true, y_pred)
        assert 0.0 <= v <= 1.0


def test_perfect_prediction_scores_one():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    assert metrics.hamming_accuracy(y, y) == 1.0
    assert metrics.exact_match(y, y) == 1.0
    assert metrics.micro_f1(y, y) == 1.0
    assert metrics.macro_f1(y, y) == 1.0


def test_log_likelihood_higher_for_confident_correct():
    y = np.array([[1, 0]])
    good = metrics.log_likelihood(y, np.array([[0.9, 0.1]]))
    bad = metrics.log_likelihood(y, np.array([[0.6, 0.4]]))
    assert good > bad


def test_log_likelihood_averages_over_label_observation_pairs():
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.8, 0.3], [0.4, 0.7]])
    expected = np.mean(y_true * np.log(y_prob) +
                       (1 - y_true) * np.log(1 - y_prob))
    assert metrics.log_likelihood(y_true, y_prob) == pytest.approx(expected)


def test_log_likelihood_is_finite_at_the_probability_bounds():
    y_true = np.array([[1, 0]])
    assert np.isfinite(metrics.log_likelihood(y_true, np.array([[0.0, 1.0]])))


def test_mean_auc_perfect_separation():
    y_true = np.array([[0], [0], [1], [1]])
    y_prob = np.array([[0.1], [0.2], [0.8], [0.9]])
    assert metrics.mean_auc(y_true, y_prob) == pytest.approx(1.0)


def test_mean_auc_skips_single_class_labels():
    # Second label is constant -> skipped; first is perfectly separable.
    y_true = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    y_prob = np.array([[0.1, 0.5], [0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])
    assert metrics.mean_auc(y_true, y_prob) == pytest.approx(1.0)


def test_scorer_registry():
    assert set(metrics.SCORERS) == {
        "hamming_accuracy", "exact_match", "micro_f1", "macro_f1",
        "log_likelihood", "mean_auc",
    }
    assert metrics.get_scorer("micro_f1") is metrics.SCORERS["micro_f1"]
    with pytest.raises(ValueError):
        metrics.get_scorer("nope")

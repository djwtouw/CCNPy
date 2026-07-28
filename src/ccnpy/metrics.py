"""Multi-label scoring functions.

Each metric is provided as a plain function and, via :data:`SCORERS` /
:func:`get_scorer`, as a scikit-learn scorer object usable directly with
``GridSearchCV``/``cross_val_score``. Higher is better for every scorer here.
"""

import numpy as np
from sklearn.metrics import f1_score, make_scorer, roc_auc_score


def _as2d(a):
    return np.atleast_2d(np.asarray(a))


def hamming_accuracy(y_true, y_pred):
    """Proportion of correctly predicted individual labels (1 - Hamming loss)."""
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    return float((y_true == y_pred).mean())


def exact_match(y_true, y_pred):
    """Proportion of observations with all labels predicted correctly."""
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    return float((y_true == y_pred).all(axis=1).mean())


def micro_f1(y_true, y_pred):
    """Micro-averaged F1 (confusion matrix pooled across labels)."""
    return float(f1_score(_as2d(y_true), _as2d(y_pred),
                          average="micro", zero_division=0))


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 (F1 computed per label, then averaged)."""
    return float(f1_score(_as2d(y_true), _as2d(y_pred),
                          average="macro", zero_division=0))


def log_likelihood(y_true, y_prob):
    """Mean per-label log-likelihood under the predicted probabilities."""
    eps = 1e-15
    y_true, y_prob = _as2d(y_true), _as2d(y_prob)
    p = np.clip(y_prob, eps, 1 - eps)
    return float((y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def mean_auc(y_true, y_prob):
    """Mean per-label ROC AUC; labels with a single class present are skipped."""
    y_true, y_prob = _as2d(y_true), _as2d(y_prob)
    aucs = []
    for l in range(y_true.shape[1]):
        if len(np.unique(y_true[:, l])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, l], y_prob[:, l]))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


# scikit-learn scorer objects (estimator, X, y) -> float, higher is better.
SCORERS = {
    "hamming_accuracy": make_scorer(hamming_accuracy),
    "exact_match": make_scorer(exact_match),
    "micro_f1": make_scorer(micro_f1),
    "macro_f1": make_scorer(macro_f1),
    "log_likelihood": make_scorer(log_likelihood, response_method="predict_proba"),
    "mean_auc": make_scorer(mean_auc, response_method="predict_proba"),
}


def get_scorer(name):
    """Return the scikit-learn scorer registered under ``name``."""
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}")


__all__ = [
    "hamming_accuracy", "exact_match", "micro_f1", "macro_f1",
    "log_likelihood", "mean_auc", "SCORERS", "get_scorer",
]

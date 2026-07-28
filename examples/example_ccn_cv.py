"""Cross-validation and hyperparameter selection.

Run with:  python examples/example_ccn_cv.py
"""

import numpy as np
from sklearn.metrics import make_scorer

from ccnpy import ccn_cv, generate_dataset
from ccnpy.model_selection import MultilabelStratifiedKFold

# Generate sample data
X, Y = generate_dataset(n=200, random_state=2)

# Cross-validate over a small hyperparameter grid. ccn_cv returns a fitted
# scikit-learn GridSearchCV, refit on the full data with the best parameters.
search = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01, 0.1], cv=3, random_state=1)
print("best params:", search.best_params_)

# The best model (refit on all data) and predictions straight from the search
best = search.best_estimator_
print(best)
print("classes:\n", search.predict(X)[:5])
print("probabilities:\n", search.predict_proba(X)[:5].round(3))
print("intercepts:", best.coef_["b"].round(3))

# Use a different built-in scorer for selection
search_f1 = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01, 0.1], cv=3,
                   scoring="macro_f1", random_state=1)
print("best params (macro_f1):", search_f1.best_params_)

# Use a custom scorer. Build it with scikit-learn's make_scorer; this one
# rewards predicted probabilities close to the true labels, so it needs
# predict_proba (higher is better).
def soft_accuracy(y_true, y_prob):
    return float(np.mean(1.0 - np.abs(y_true - y_prob)))

soft_scorer = make_scorer(soft_accuracy, response_method="predict_proba")
search_custom = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01, 0.1], cv=3,
                       scoring=soft_scorer, random_state=1)
print("best params (custom):", search_custom.best_params_)

# Reuse the same folds for a fair comparison between settings: build the
# splitter once with a fixed random_state and pass it to each search.
folds = MultilabelStratifiedKFold(n_splits=3, random_state=1)
cmp_a = ccn_cv(X, Y, q=[1.0], alpha=[0.01, 0.1], cv=folds, random_state=1)
cmp_b = ccn_cv(X, Y, q=[2.0], alpha=[0.01, 0.1], cv=folds, random_state=1)
print("q=1 best:", cmp_a.best_params_, "| q=2 best:", cmp_b.best_params_)

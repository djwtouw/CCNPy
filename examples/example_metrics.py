"""Evaluating predictions with the built-in scoring functions.

Run with:  python examples/example_metrics.py
"""

from ccnpy import CCN, ccn_cv, generate_dataset
from ccnpy import metrics

# Generate sample data
X, Y = generate_dataset(n=200, random_state=2)

# Fit a classifier chain network and obtain predictions
model = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)
y_class = model.predict(X)
y_prob = model.predict_proba(X)

# Class-based scorers take (y_true, y_pred); probability-based scorers take
# (y_true, y_prob).
print("hamming_accuracy:", round(metrics.hamming_accuracy(Y, y_class), 4))
print("exact_match     :", round(metrics.exact_match(Y, y_class), 4))
print("micro_f1        :", round(metrics.micro_f1(Y, y_class), 4))
print("macro_f1        :", round(metrics.macro_f1(Y, y_class), 4))
print("log_likelihood  :", round(metrics.log_likelihood(Y, y_prob), 4))
print("mean_auc        :", round(metrics.mean_auc(Y, y_prob), 4))

# Use a scorer by name to select hyperparameters in cross-validation
search = ccn_cv(X, Y, q=[1.0], alpha=[0.01, 0.1], cv=3,
                scoring="mean_auc", random_state=1)
print("best params (mean_auc):", search.best_params_)

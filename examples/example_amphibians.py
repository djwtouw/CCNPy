"""A complete multi-label analysis of the amphibians data set.

Fits a classifier chain network, chooses the label order, selects the tuning
parameters by cross-validation, evaluates predictions on held-out data, and
inspects the estimated label interdependencies.

Run with:  python examples/example_amphibians.py
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ccnpy import CCN, ccn_cv, entropy_label_order, load_amphibians
from ccnpy.metrics import (exact_match, hamming_accuracy, log_likelihood,
                           macro_f1, mean_auc, micro_f1)
from ccnpy.model_selection import MultilabelStratifiedKFold


def show(matrix, row_names, col_names, digits=3):
    """Print a matrix with row and column names, leaving NaN entries blank."""
    width = max(len(name) for name in list(row_names) + list(col_names)) + 1
    print(" " * width + "".join(f"{name:>{width}}" for name in col_names))
    for name, row in zip(row_names, matrix):
        cells = ["" if np.isnan(value) else f"{value:.{digits}f}"
                 for value in row]
        print(f"{name:<{width}}" + "".join(f"{cell:>{width}}" for cell in cells))


def chain_matrix(C):
    """Rows 2..L and columns 1..L-1 of C, with the empty upper part blank."""
    M = C[1:, :-1].copy()
    M[np.triu_indices_from(M, k=1)] = np.nan
    return M


# Data: presence of seven amphibian species at 189 water reservoirs, with the
# site characteristics summarized by 25 principal components
amphibians = load_amphibians()
X, Y = amphibians.data, amphibians.target
names = np.array(amphibians.target_names)
print("X:", X.shape, " Y:", Y.shape)
show(Y.mean(axis=0, keepdims=True), ["prevalence"], names, digits=2)

# Set aside 49 sites to evaluate the final model on data it has not seen
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=140, random_state=1)

# Fit a classifier chain network. q controls how the losses of the labels are
# aggregated and alpha is the ridge penalty.
model = CCN(q=1.5, alpha=0.5).fit(X_train, Y_train)
print("\nTraining loss:", round(model.loss_, 5))

# Coefficients are stored in chain order. The coefficients of the principal
# components are of little interest here, so only b and C are shown.
chain = names[model.label_order_]
print("\nIntercepts (b):")
show(model.coef_["b"][None, :], ["b"], chain)
print("\nCoefficients of the label predictions (C):")
show(chain_matrix(model.coef_["C"]), chain[1:], chain[:-1])

# Choose the label order. entropy_label_order derives an order from the
# labels themselves. Passed as a function, the order is recomputed on the
# data used in every fit, so during cross-validation each set of training
# folds derives its own order.
print("\ncebcc1 order:", names[entropy_label_order(Y_train, method="cebcc1")])


def cebcc1_order(X, Y):
    return entropy_label_order(Y, method="cebcc1")


# Stratified folds keep the proportion of every label close to its proportion
# in the full training data, which matters for rare labels such as CrestNewt
folds = MultilabelStratifiedKFold(n_splits=5, random_state=1)
per_fold = [Y_train[test].mean(axis=0) for _, test in folds.split(X_train, Y_train)]
print("\nLabel proportions per fold:")
show(per_fold, [f"fold {k + 1}" for k in range(5)], names, digits=2)

# Select q, alpha, and the label order by cross-validation. None stands for
# the column order of Y. Combinations are scored with the Hamming accuracy.
cv = ccn_cv(X_train, Y_train, q=[1.0, 2.0], alpha=[0.1, 0.5, 1.0],
            label_order=[None, cebcc1_order], cv=folds)
best = cv.best_params_
order_name = "default" if best["label_order"] is None else "cebcc1"
print(f"\nSelected: q = {best['q']}, alpha = {best['alpha']}, "
      f"label order = {order_name}")

# Predict on the held-out sites with the model refitted on all training data.
# Besides the default threshold of 0.5, each label can get its own threshold;
# setting it to the prevalence of the label helps to detect rare labels.
Y_prob = cv.predict_proba(X_test)
Y_class = cv.best_estimator_.predict(X_test)
Y_class_prev = cv.best_estimator_.predict(X_test,
                                          threshold=Y_train.mean(axis=0))

scores = np.array([
    [score(Y_test, Y_class), score(Y_test, Y_class_prev)]
    for score in (hamming_accuracy, exact_match, micro_f1, macro_f1)
])
print("\nScores on the held-out sites:")
show(scores, ["hamming", "exact_match", "micro_f1", "macro_f1"],
     ["threshold_0.5", "prevalence"])

# The log-likelihood and the mean area under the ROC curve use the
# probabilities and do not depend on a threshold
print("\nlog-likelihood:", round(log_likelihood(Y_test, Y_prob), 4))
print("mean AUC:      ", round(mean_auc(Y_test, Y_prob), 4))

# Estimated label interdependencies of the selected model. Rows are the
# predicted labels and columns the labels preceding them in the chain. A
# positive entry means that a higher predicted presence of the earlier species
# increases the predicted presence of the later one, after accounting for the
# site characteristics.
selected = cv.best_estimator_
chain = names[selected.label_order_]
print("\nCoefficients of the label predictions (C) of the selected model:")
show(chain_matrix(selected.coef_["C"]), chain[1:], chain[:-1])

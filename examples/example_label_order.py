"""Choosing the order in which labels are chained.

Run with:  python examples/example_label_order.py
"""

import numpy as np

from ccnpy import (CCN, ccn_cv, conditional_entropy_matrix,
                   entropy_label_order, generate_dataset)

X, Y = generate_dataset(n=200, random_state=2)

# Pairwise conditional entropies of the labels, in bits: entry [i, j] is
# H(Y_j | Y_i), the uncertainty left in label j once label i is known. Row sums
# say how much a label explains the others, column sums how much the others
# explain it, and the ordering rules below aggregate one or the other.
H = conditional_entropy_matrix(Y)
print("conditional entropy matrix:\n", H.round(3))
print("row sums   :", H.sum(axis=1).round(3))
print("column sums:", H.sum(axis=0).round(3))

# Each rule turns that matrix into a chain order. Different rules often agree
# on a given dataset, and they need not agree on a subsample of it.
for method in ("cebcc1", "cebcc2", "cebcc3", "cebcc4", "ebcc"):
    print(f"{method}: {entropy_label_order(Y, method=method)}")

# Fit with an order computed up front
order = entropy_label_order(Y, method="cebcc1")
fixed = CCN(q=1.0, alpha=0.01, label_order=order).fit(X, Y)
print("fixed order  :", fixed.label_order_, "loss", round(fixed.loss_, 4))

# Or hand CCN the rule itself. label_order then accepts any callable f(X, Y)
# returning a permutation, and it is evaluated at the start of every fit, so
# the order always comes from the rows that fit actually sees.
def cebcc1_order(X, Y):
    return entropy_label_order(Y, method="cebcc1")

rule = CCN(q=1.0, alpha=0.01, label_order=cebcc1_order).fit(X, Y)
print("rule on all data:", rule.label_order_, "loss", round(rule.loss_, 4))

# On a subsample the same rule can pick a different order, which is exactly
# what makes it worth deferring the choice to fit time.
sub = CCN(q=1.0, alpha=0.01, label_order=cebcc1_order).fit(X[:60], Y[:60])
print("rule on 60 rows :", sub.label_order_)

# Cross-validate the rules against each other and against the column order of
# Y (the None candidate). Passing rules rather than permutations keeps the
# comparison honest: a permutation computed on all of Y has already seen the
# validation folds, which flatters it. Any callable works here, so a rule of
# your own is just another candidate.
def cebcc3_order(X, Y):
    return entropy_label_order(Y, method="cebcc3")

def reverse_order(X, Y):
    return np.arange(Y.shape[1])[::-1]

search = ccn_cv(X, Y, q=[1.0, 2.0], alpha=[0.01], cv=3, random_state=1,
                label_order=[None, cebcc1_order, cebcc3_order, reverse_order])

names = {None: "column order", cebcc1_order: "cebcc1",
         cebcc3_order: "cebcc3", reverse_order: "reversed"}
for params, score in zip(search.cv_results_["params"],
                         search.cv_results_["mean_test_score"]):
    print(f"q={params['q']:<4} {names[params['label_order']]:<13}"
          f" score={score:.4f}")

print("best:", names[search.best_params_["label_order"]],
      "with q =", search.best_params_["q"])
print("order of the refitted model:", search.best_estimator_.label_order_)

"""Fitting a Classifier Chain Network and inspecting the result.

Run with:  python examples/example_ccn.py
"""

from ccnpy import CCN, generate_dataset

# Generate sample data: 3 predictors, 4 binary labels
X, Y = generate_dataset(n=200, random_state=2)

# Fit a classifier chain network
model = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)
print(model)

# Inspect the estimated coefficients: intercepts (b), predictor weights (W),
# and chain weights (C)
coef = model.coef_
print("b:", coef["b"].round(3))
print("W:\n", coef["W"].round(3))
print("C:\n", coef["C"].round(3))

# In-sample predicted classes and probabilities (first rows)
print("classes:\n", model.predict(X)[:5])
print("probabilities:\n", model.predict_proba(X)[:5].round(3))

# Fit with a custom label order: labels earlier in the chain are used as
# features for later labels, while outputs remain in the original column
# order. label_order is a 0-based permutation of the label columns.
model_lo = CCN(q=1.0, alpha=0.01, label_order=[3, 0, 2, 1],
               random_state=0).fit(X, Y)
print("label order used:", model_lo.label_order_.tolist())

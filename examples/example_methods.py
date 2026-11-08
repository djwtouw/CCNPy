"""Inspection methods: repr, summary, coefficients, fitted values, residuals.

Run with:  python examples/example_methods.py
"""

from ccnpy import CCN, generate_dataset

# Generate sample data
X, Y = generate_dataset(n=200, random_state=2)

# Fit a classifier chain network
model = CCN(q=1.0, alpha=0.01, random_state=0).fit(X, Y)

# Compact overview (repr) and a full summary of the model
print(model)
model.summary()

# Coefficients: intercepts (b), predictor weights (W), chain weights (C)
print(model.coef_)

# Fitted classes and fitted probabilities (in-sample, first rows)
print("classes:\n", model.predict(X)[:5])
print("probabilities:\n", model.predict_proba(X)[:5].round(3))

# Response residuals: y minus the fitted probabilities
residuals = Y - model.fitted_proba_
print("residuals:\n", residuals[:5].round(3))

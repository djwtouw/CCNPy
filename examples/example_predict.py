"""Prediction: classes, probabilities, and decision thresholds.

Run with:  python examples/example_predict.py
"""

from ccnpy import CCN, generate_dataset

# Generate sample data and split into training and test sets
X, Y = generate_dataset(n=250, random_state=2)
X_train, Y_train = X[:200], Y[:200]
X_test = X[200:]

# Fit a classifier chain network on the training data
model = CCN(q=1.0, alpha=0.01, random_state=0).fit(X_train, Y_train)

# Predicted classes and probabilities for the test data (first rows)
print("classes:\n", model.predict(X_test)[:5])
print("probabilities:\n", model.predict_proba(X_test)[:5].round(3))

# Custom decision threshold: a scalar applied to all labels, or one
# threshold per label
print("threshold 0.3:\n", model.predict(X_test, threshold=0.3)[:5])
print("per-label threshold:\n",
      model.predict(X_test, threshold=[0.3, 0.5, 0.5, 0.7])[:5])

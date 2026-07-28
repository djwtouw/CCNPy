"""Generating a synthetic multi-label dataset.

Run with:  python examples/example_generate_dataset.py
"""

from ccnpy import generate_dataset

# Generate sample data
X, Y = generate_dataset(n=100, random_state=123)

# Explanatory variables (n x 3) and binary labels (n x 4)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("X head:\n", X[:5].round(3))
print("Y head:\n", Y[:5])

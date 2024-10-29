import numpy as np
from ccnpy import CCN
from ccnpy.metrics import accuracy


def generate_dataset(n):
    # Generate explanatory variables for multi-labeled data
    X = np.random.multivariate_normal(
        np.array([0.0, 0.0, 0.0]),
        np.array([[2.0, 0.4, 0.4], [0.4, 2.0, 0.4], [0.4, 0.4, 2.0]]),
        n
    )

    # Initialize Y
    Y = np.empty((X.shape[0], 4))

    # Compute multi-labeled outcomes
    for i in range(Y.shape[1]):
        # Compute X @ beta, where the elements of beta are uniform [-2, 2]
        Y[:, i] = X @ (np.random.rand(X.shape[1]) * 4 - 2.0)

        # Add effect of previous labels
        if i > 0:
            # The effects are random on the interval [-4, 4]
            Y[:, i] += Y[:, :i] @ (np.random.rand(i) * 8 - 4.0)

        Y[:, i] = 1.0 / (1.0 + np.exp(-Y[:, i]))

    # Transform Y into binary outcomes based on the computed probabilities
    Y = (Y >= np.random.rand(*Y.shape)).astype(float)

    return X, Y


# %%

# Set seed and generate training and testing data
np.random.seed(123)
X_train, Y_train = generate_dataset(200)
X_test, Y_test = generate_dataset(1000)

# Fit a classifier chain network
model = CCN().fit(X_train, Y_train)

# Make predictions
Y_hat = model.predict(X_test)

# Compute accuracy on test set
print(f"Accuracy: {accuracy(Y_test, Y_hat):.4f}")

# Perform cross validation to tune the hyperparameters
model_cv = CCN().cv(X_train, Y_train, grid_search_type="random")

# Make predictions
Y_hat_cv = model_cv.predict(X_test)

# Compute accuracy on test set
print(f"Accuracy: {accuracy(Y_test, Y_hat_cv):.4f}")

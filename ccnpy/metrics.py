import numpy as np


def accuracy(Y, Y_hat):
    return (Y == Y_hat).mean()


def hamming_loss(Y, Y_hat):
    """
    Compute the Hamming loss for the predictions.

    :param Y: True labels.
    :param Y_hat: Predicted labels.
    :return: Hamming loss.
    """
    return (Y != Y_hat).mean()


def zero_one_loss(Y, Y_hat):
    """
    Compute the zero-one loss for the predictions.

    :param Y: True labels.
    :param Y_hat: Predicted labels.
    :return: Zero-one loss.
    """
    return np.any(Y_hat != Y, axis=1).mean()


def _confusion_matrix(y, y_hat):
    """
    Compute the confusion matrix.

    :param y: True labels.
    :param y_hat: Predicted labels.
    :return: Confusion matrix.
    """
    # Initialize confusion matrix
    cm = np.zeros((2, 2))

    # True positives
    cm[0, 0] = np.logical_and(y == 1, y_hat == 1).sum()
    # False negatives
    cm[0, 1] = np.logical_and(y == 1, y_hat == 0).sum()
    # False positives
    cm[1, 0] = np.logical_and(y == 0, y_hat == 1).sum()
    # True negatives
    cm[1, 1] = y.size - cm.sum()

    return cm


def micro_F1(Y, Y_hat):
    """
    Compute the micro-F1 score for the predictions.

    :param Y: True labels.
    :param Y_hat: Predicted labels.
    :return: Micro-F1 score.
    """
    # Initialize confusion matrix
    cm = np.zeros((2, 2))

    # Ensure 2D input
    Y = np.atleast_2d(Y)
    Y_hat = np.atleast_2d(Y_hat)

    # Number of labels
    L = Y.shape[1]

    # Compute micro confusion matrix
    for i in range(L):
        cm += _confusion_matrix(Y[:, i], Y_hat[:, i])

    # Compute micro F1
    result = 2 * cm[0, 0] / (2 * cm[0, 0] + cm[0, 1] + cm[1, 0])

    return result


def macro_F1(Y, Y_hat):
    """
    Compute the macro-F1 score for the predictions.

    :param Y: True labels.
    :param Y_hat: Predicted labels.
    :return: Macro-F1 score.
    """
    # Initialize macro F1
    result = 0

    # Ensure 2D input
    Y = np.atleast_2d(Y)
    Y_hat = np.atleast_2d(Y_hat)

    # Number of labels
    L = Y.shape[1]

    # Compute confusion matrices
    for i in range(L):
        cm = _confusion_matrix(Y[:, i], Y_hat[:, i])

        # Compute F1
        result += 2 * cm[0, 0] / (2 * cm[0, 0] + cm[0, 1] + cm[1, 0])

    # Take the average
    result /= L

    return result


def negloglik(Y, Y_hat):
    """
    Compute the negative log-likelihood score for the predictions.

    :param Y: True labels.
    :param Y_hat: Predicted label probabilities.
    :return: Negative log-likelihood score.
    """
    # Copy Y_hat and clip values to prevent numerical instability
    Y_hat_c = np.clip(Y_hat, 1e-12, 1 - 1e-12)

    return -(Y * np.log(Y_hat_c) + (1 - Y) * np.log(1 - Y_hat_c)).mean()

import numpy as np
from sklearn.model_selection import train_test_split


def _generate_kfold(Y, k, stratify):
    # Number of observations
    n = Y.shape[0]

    # Initialize folds
    folds = []

    # Get the indices that are not in a fold yet
    remainder = np.arange(n)

    # Divide the remaining indices into a fold and a new remainder
    for i in range(k - 1):
        # Fold size
        if n % k == 0:
            fold_size = n // k
        else:
            fold_size = 1 / k / (1 - i / k)

        # Get the split
        remainder, fold = train_test_split(
            remainder,
            test_size=fold_size,
            stratify=Y[np.ix_(remainder, stratify)],
        )

        # Add the fold to the folds
        folds.append(fold)

    # After the final iteration, add the remainder to the folds
    folds.append(remainder)

    return folds


def _find_candidate_folds(Y, k, imbalance_order):
    # Number of labels
    L = Y.shape[1]

    # The outer loop reduces the number of columns that are considered for
    # stratifying the sample if no valid folds can be found
    for label_l in range(L):
        # Try ten times to find folds that use the specified columns of Y for
        # stratifying
        for i in range(10):
            try:
                folds = _generate_kfold(Y, k, imbalance_order[: (L - label_l)])

                # If a valid set of folds is found, immediately return
                return folds
            except ValueError:
                pass

    return None


def multilabel_kfold(Y, k):
    # Get label proportions and find the most imbalanced ones
    label_prop = Y.mean(axis=0)
    imbalance = np.minimum(label_prop, 1 - label_prop)

    # Lower means more imbalance, get the order of the labels
    imbalance_order = np.argsort(imbalance)

    # Try ten times to find valid folds that have label heterogeneity
    for i in range(10):
        # Flag to indicate success
        success = True

        # Find candidate folds
        folds = _find_candidate_folds(Y, k, imbalance_order)

        if folds is not None:
            # Check if all folds have different outcomes for each of the labels
            for fold in folds:
                if not (
                    np.all(Y[fold, :].sum(axis=0) > 0)
                    and np.all(Y[fold, :].sum(axis=0) < len(fold))
                ):
                    success = False
        else:
            success = False

        # If a valid set of folds was found, break out of the loop
        if success:
            break

    if not success:
        raise ValueError(
            "It is not possible to construct a valid set of "
            "folds with variation in each of the labels for each "
            "of the folds. Try setting k to a lower value"
        )

    return folds

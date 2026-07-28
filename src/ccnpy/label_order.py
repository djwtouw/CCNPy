"""Entropy based label orders for the classifier chain.

The chain order matters: earlier labels are used as features for later ones, so
a poor order propagates errors down the chain. The functions here derive an
order from the labels alone, following Jun et al. (2019), and return a
permutation that can be handed straight to the ``label_order`` parameter of
:class:`~ccnpy.CCN`.

References
----------
Jun, X., Lu, Y., Lei, Z. and Guolun, D. (2019). Conditional entropy based
classifier chains for multi-label classification. *Neurocomputing*, 335,
185-194.
"""

import numpy as np
from sklearn.utils import check_array

METHODS = ("cebcc1", "cebcc2", "cebcc3", "cebcc4", "ebcc")


def _check_labels(Y):
    """Validated (n, L) binary float array."""
    Y = check_array(Y, dtype=np.float64, ensure_2d=True)
    if not np.all(np.isin(Y, (0.0, 1.0))):
        raise ValueError("Y must contain only 0 and 1")
    return Y


def _contribution(count, n, margin):
    """Term ``p log2(margin / p)`` of a conditional entropy, elementwise.

    ``count`` holds joint counts for one cell of every label pair, ``margin``
    the marginal probability that cell is conditioned on, broadcast along rows.
    Cells with zero probability contribute nothing (0 log 0 = 0), which also
    covers a zero margin, since that forces a zero joint count.
    """
    p = count / n
    ratio = np.divide(margin, p, out=np.ones_like(p), where=p > 0)
    return p * np.log2(ratio)


def conditional_entropy_matrix(Y):
    """Pairwise conditional entropies of the labels.

    Entry ``[i, j]`` is ``H(Y_j | Y_i)`` in bits, estimated by plugging the
    joint frequency table of the two labels into the definition. The diagonal
    is zero.

    Parameters
    ----------
    Y : array-like of shape (n_samples, n_labels), values in {0, 1}

    Returns
    -------
    ndarray of shape (n_labels, n_labels)

    Notes
    -----
    A label that is constant contributes zero entropy, and conditioning on it
    leaves the other label's entropy unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from ccnpy import conditional_entropy_matrix
    >>> Y = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    >>> conditional_entropy_matrix(Y)   # independent labels
    array([[0., 1.],
           [1., 0.]])
    """
    Y = _check_labels(Y)
    n = Y.shape[0]
    Y0 = 1.0 - Y

    # Joint counts for the four cells of every label pair: entry [i, j] of
    # n_ab counts the rows with Y_i = a and Y_j = b.
    n11 = Y.T @ Y
    n10 = Y.T @ Y0
    n01 = Y0.T @ Y
    n00 = Y0.T @ Y0

    # H(Y_j | Y_i) conditions on Y_i, so the margins vary along rows.
    p1 = Y.mean(axis=0)[:, None]
    result = (_contribution(n11, n, p1) + _contribution(n10, n, p1)
              + _contribution(n01, n, 1.0 - p1)
              + _contribution(n00, n, 1.0 - p1))
    np.fill_diagonal(result, 0.0)
    return result


def _marginal_entropy(Y):
    """Per label binary entropy H(Y_l) in bits."""
    p = Y.mean(axis=0)
    inner = (p > 0.0) & (p < 1.0)
    h = np.zeros_like(p)
    h[inner] = -(p[inner] * np.log2(p[inner])
                 + (1.0 - p[inner]) * np.log2(1.0 - p[inner]))
    return h


def _greedy_order(CE, method):
    """Greedy chain order from a conditional entropy matrix.

    Repeatedly places the extreme label at one end of the chain and drops its
    row and column, so later steps aggregate only over the labels still
    unplaced. Ties go to the lowest remaining label index.
    """
    use_rows = method in ("cebcc1", "cebcc2")
    take_max = method in ("cebcc2", "cebcc4")
    from_end = method in ("cebcc1", "cebcc4")

    M = CE.copy()
    labels = list(range(CE.shape[0]))
    picks = []

    while labels:
        totals = M.sum(axis=1) if use_rows else M.sum(axis=0)
        idx = int(np.argmax(totals) if take_max else np.argmin(totals))
        picks.append(labels.pop(idx))
        M = np.delete(np.delete(M, idx, axis=0), idx, axis=1)

    # Labels destined for the end of the chain were picked first.
    return np.array(picks[::-1] if from_end else picks, dtype=int)


def entropy_label_order(Y, method="cebcc1"):
    """Chain order derived from the entropy of the labels.

    Implements the orderings of Jun et al. (2019). All of them place labels that
    are more certain earlier in the chain, so that error propagates from
    well-determined labels toward poorly determined ones rather than the other
    way around. The four conditional entropy rules differ in whether a label is
    judged by how much it explains the others or by how much the others explain
    it, and in which end of the chain is filled first.

    Parameters
    ----------
    Y : array-like of shape (n_samples, n_labels), values in {0, 1}
    method : {"cebcc1", "cebcc2", "cebcc3", "cebcc4", "ebcc"}, default="cebcc1"
        Ordering rule. With ``H`` the matrix returned by
        :func:`conditional_entropy_matrix`:

        - ``"cebcc1"``: the label with the smallest row sum of ``H`` goes to the
          end of the chain. The best performer in Jun et al. (2019), hence the
          default.
        - ``"cebcc2"``: the label with the largest row sum goes to the
          beginning.
        - ``"cebcc3"``: the label with the smallest column sum goes to the
          beginning.
        - ``"cebcc4"``: the label with the largest column sum goes to the end.
        - ``"ebcc"``: labels sorted by increasing marginal entropy
          ``H(Y_l)``, ignoring the dependence between labels.

        The conditional entropy rules apply their criterion greedily,
        recomputing the sums over the labels not yet placed.

    Returns
    -------
    ndarray of shape (n_labels,)
        Zero-based permutation, suitable for the ``label_order`` parameter of
        :class:`~ccnpy.CCN`.

    Notes
    -----
    Row sums and column sums answer different questions. The row sum
    ``sum_j H(Y_j | Y_i)`` is small when label ``i`` explains the others well,
    the column sum ``sum_i H(Y_j | Y_i)`` is small when the others explain
    label ``j`` well. Ties are resolved toward the lowest label index, so the
    result can depend on the column order of ``Y``.

    Deriving an order from the same labels that are later used for evaluation
    leaks information. Passing this function to ``label_order`` as a callable
    rather than precomputing the permutation avoids that, because every fit
    then derives its order from its own training rows.

    References
    ----------
    Jun, X., Lu, Y., Lei, Z. and Guolun, D. (2019). Conditional entropy based
    classifier chains for multi-label classification. *Neurocomputing*, 335,
    185-194.

    Examples
    --------
    >>> from ccnpy import CCN, entropy_label_order, generate_dataset
    >>> X, Y = generate_dataset(200, random_state=0)
    >>> order = entropy_label_order(Y, method="cebcc1")
    >>> model = CCN(q=1.0, alpha=0.01, label_order=order).fit(X, Y)

    Let each fit derive its own order, which is what cross-validation wants:

    >>> def cebcc1(X, Y):
    ...     return entropy_label_order(Y, method="cebcc1")
    >>> model = CCN(q=1.0, alpha=0.01, label_order=cebcc1).fit(X, Y)
    >>> model.label_order_.shape
    (4,)
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {list(METHODS)}")

    Y = _check_labels(Y)
    if method == "ebcc":
        # Removing a label does not change the marginal entropy of any other,
        # so a single stable sort is equivalent to placing them greedily.
        return np.argsort(_marginal_entropy(Y), kind="stable").astype(int)

    return _greedy_order(conditional_entropy_matrix(Y), method)


__all__ = ["conditional_entropy_matrix", "entropy_label_order", "METHODS"]

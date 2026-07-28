"""CCNPy: Classifier Chain Networks for multi-label classification.

The main entry point is the :class:`CCN` estimator, which follows the
scikit-learn API (``fit`` / ``predict`` / ``predict_proba``).
"""

from . import metrics
from .datasets import generate_dataset
from .estimator import CCN
from .label_order import conditional_entropy_matrix, entropy_label_order
from .model_selection import MultilabelStratifiedKFold, ccn_cv

__version__ = "0.2.0"

__all__ = [
    "CCN",
    "ccn_cv",
    "MultilabelStratifiedKFold",
    "conditional_entropy_matrix",
    "entropy_label_order",
    "generate_dataset",
    "metrics",
]

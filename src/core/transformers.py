import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays for dense-only estimators/transformers."""

    def fit(self, X, y=None):
        """Execute fit as part of the module workflow.

        Encapsulates a focused unit of pipeline logic for reuse and testing.
        """
        return self

    def transform(self, X):
        """Execute transform as part of the module workflow.

        Encapsulates a focused unit of pipeline logic for reuse and testing.
        """
        if sparse.issparse(X):
            return X.toarray()
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

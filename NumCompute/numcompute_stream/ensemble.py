"""Ensemble classifiers that learn from data chunk by chunk"""

import numpy as np

from numcompute.utils import validate_array_like, validate_non_empty_array, validate_options
from numcompute_stream.tree import DecisionTreeClassifier


class BaggingClassifier:
    """Bagging over decision trees — each tree sees a bootstrap sample per chunk"""

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        """Set up empty trees ready for streaming partial_fit

        n_estimators - how many trees to train
        max_depth - passed through to each DecisionTreeClassifier
        min_samples_split - passed through to each DecisionTreeClassifier
        criterion - 'gini' or 'entropy' for each tree
        max_features - passed through to each DecisionTreeClassifier
        random_state - seed for bootstrap sampling, None means non-reproducible
        """
        validate_options(criterion, ("gini", "entropy"), x_name="criterion")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.classes_ = None
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                max_features=max_features,
            )
            for _ in range(n_estimators)
        ]
        self._rng = np.random.default_rng(random_state)

    def partial_fit(self, X, y):
        """Bootstrap each tree on this chunk and partial_fit it

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns self so you can chain partial_fit calls

        Raises ValueError if X/y empty or shapes don't match
        """
        # step 1: sanity check inputs
        X = validate_array_like(X, name="X").astype(float)
        y = validate_array_like(y, name="y")
        validate_non_empty_array(X, name="X")
        validate_non_empty_array(y, name="y")

        if X.ndim != 2:
            raise ValueError("BaggingClassifier expects 2D input for X")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # step 2: remember all label classes seen so far
        chunk_classes = np.unique(y)
        if self.classes_ is None:
            self.classes_ = chunk_classes
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, chunk_classes]))

        # step 3: bootstrap sample rows and update each tree
        n_samples = X.shape[0]
        for tree in self.estimators_:
            indices = self._rng.integers(0, n_samples, size=n_samples)
            tree.partial_fit(X[indices], y[indices])

        return self

    def predict(self, X):
        """Predict class label for each row using a majority vote

        X - 2d array (n_samples, n_features)

        Returns 1d array of predicted labels

        Raises ValueError if no tree has been fit yet
        """
        if self.classes_ is None:
            raise ValueError("BaggingClassifier is not fitted")

        X = validate_array_like(X, name="X").astype(float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("BaggingClassifier expects 2D input for X")

        votes = np.array([tree.predict(X) for tree in self.estimators_])
        return self._majority_vote(votes)

    def fit(self, X, y):
        """Learn from all of X at once (calls partial_fit internally)

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns self
        """
        return self.partial_fit(X, y)

    def _majority_vote(self, votes):
        n_samples = votes.shape[1]
        predictions = np.empty(n_samples, dtype=votes.dtype)

        for j in range(n_samples):
            labels, counts = np.unique(votes[:, j], return_counts=True)
            predictions[j] = labels[counts.argmax()]

        return predictions

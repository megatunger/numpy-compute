"""Decision tree classifier with streaming partial_fit"""

import numpy as np

from numcompute.utils import validate_array_like, validate_non_empty_array, validate_options


class _Node:
    """One node in the tree — either a leaf holding samples or an internal split"""

    def __init__(self, depth=0):
        self.depth = depth
        self.n_samples = 0
        self.class_counts = {}
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self._X = []
        self._y = []

    @property
    def is_leaf(self):
        return self.left is None

    def majority_class(self):
        return max(self.class_counts, key=self.class_counts.get)


class DecisionTreeClassifier:
    """Simple decision tree that grows as new chunks arrive"""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
    ):
        """Set up an empty tree

        max_depth - max levels below the root (root depth is 0)
        min_samples_split - min samples in a leaf before we try to split it
        criterion - 'gini' or 'entropy' for split scoring
        max_features - how many features to try at each split, None means all
        """
        validate_options(criterion, ("gini", "entropy"), x_name="criterion")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.tree_ = None
        self.classes_ = None

    def partial_fit(self, X, y):
        """Route this chunk through the tree and grow splits if needed

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
            raise ValueError("DecisionTreeClassifier expects 2D input for X")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # step 2: boot the tree on first chunk
        if self.tree_ is None:
            self.tree_ = _Node(depth=0)

        # step 3: remember all label classes seen so far
        chunk_classes = np.unique(y)
        if self.classes_ is None:
            self.classes_ = chunk_classes
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, chunk_classes]))

        # step 4: send each row down to a leaf and stash it there
        for xi, yi in zip(X, y):
            leaf = self._find_leaf(self.tree_, xi)
            leaf._X.append(xi)
            leaf._y.append(yi)
            leaf.class_counts[yi] = leaf.class_counts.get(yi, 0) + 1
            leaf.n_samples += 1

        self.tree_.n_samples += X.shape[0]
        return self

    def predict(self, X):
        """Predict class label for each row

        X - 2d array (n_samples, n_features)

        Returns 1d array of predicted labels

        Raises ValueError if the tree hasn't been fit yet
        """
        raise NotImplementedError

    def fit(self, X, y):
        """Learn from all of X at once (calls partial_fit internally)

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns self
        """
        return self.partial_fit(X, y)

    def _find_leaf(self, node, x):
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

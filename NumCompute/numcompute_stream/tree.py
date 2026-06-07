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

        # step 5: try splitting any leaf that has enough data
        self._grow(self.tree_)
        return self

    def predict(self, X):
        """Predict class label for each row

        X - 2d array (n_samples, n_features)

        Returns 1d array of predicted labels

        Raises ValueError if the tree hasn't been fit yet
        """
        if self.tree_ is None:
            raise ValueError("DecisionTreeClassifier is not fitted")

        X = validate_array_like(X, name="X").astype(float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("DecisionTreeClassifier expects 2D input for X")

        return np.array([self._predict_one(xi) for xi in X])

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

    def _predict_one(self, x):
        leaf = self._find_leaf(self.tree_, x)
        return leaf.majority_class()

    def _grow(self, node):
        if node.is_leaf:
            self._maybe_split(node)
        else:
            self._grow(node.left)
            self._grow(node.right)

    def _maybe_split(self, node):
        if node.depth >= self.max_depth:
            return
        if node.n_samples < self.min_samples_split:
            return

        X = np.asarray(node._X, dtype=float)
        y = np.asarray(node._y)

        if len(np.unique(y)) == 1:
            return

        feature_indices = self._feature_indices(X.shape[1])
        best = self._best_split(X, y, feature_indices)
        if best is None:
            return

        feature_index, threshold = best
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = _Node(depth=node.depth + 1)
        node.right = _Node(depth=node.depth + 1)

        for xi, yi, is_left in zip(X, y, left_mask):
            child = node.left if is_left else node.right
            child._X.append(xi)
            child._y.append(yi)
            child.class_counts[yi] = child.class_counts.get(yi, 0) + 1
            child.n_samples += 1

        node._X = []
        node._y = []

        self._grow(node.left)
        self._grow(node.right)

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return list(range(n_features))
        return list(range(min(self.max_features, n_features)))

    def _best_split(self, X, y, feature_indices):
        parent_impurity = self._impurity(y)
        best_score = np.inf
        best_split = None

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])
            if values.size <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                if left_mask.all() or (~left_mask).all():
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]
                weighted = (
                    (y_left.size / y.size) * self._impurity(y_left)
                    + (y_right.size / y.size) * self._impurity(y_right)
                )
                gain = parent_impurity - weighted

                if gain <= 0:
                    continue

                if best_split is None:
                    best_score = weighted
                    best_split = (feature_index, threshold)
                    continue

                if weighted < best_score - 1e-12:
                    best_score = weighted
                    best_split = (feature_index, threshold)
                elif np.isclose(weighted, best_score):
                    if feature_index < best_split[0] or (
                        feature_index == best_split[0]
                        and threshold < best_split[1]
                    ):
                        best_score = weighted
                        best_split = (feature_index, threshold)

        return best_split

    def _impurity(self, y):
        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)

        entropy = -np.sum(probs * np.log2(probs))
        return entropy

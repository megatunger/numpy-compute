"""Decision tree classifier with streaming partial_fit"""


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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.tree_ = None  # root node once we have data
        self.classes_ = None  # label classes seen so far

    def partial_fit(self, X, y):
        """Route this chunk through the tree and grow splits if needed

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns self so you can chain partial_fit calls

        Raises ValueError if X/y empty or shapes don't match
        """
        raise NotImplementedError

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

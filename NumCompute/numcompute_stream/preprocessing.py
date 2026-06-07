"""Preprocessing that learns from data chunk by chunk"""

import numpy as np

from numcompute.preprocessing import OneHotEncoder as BatchOneHotEncoder
from numcompute.preprocessing import SimpleImputer as BatchSimpleImputer
from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute.utils import validate_array_like, validate_non_empty_array
from numcompute_stream.stats import RunningStats


class StandardScaler(BatchStandardScaler):
    """StandardScaler you can train one chunk at a time"""

    def __init__(self):
        """Start unfitted with no samples seen yet"""
        super().__init__()
        self.n_samples_seen_ = 0  # running count of rows fed in
        self._running_stats = RunningStats(ddof=0)

    def partial_fit(self, X, y=None):
        """Update mean and scale from one chunk of data

        X - 2d array (n_samples, n_features)
        y - not used, just here to match sklearn-style API

        Returns self so you can chain partial_fit calls

        Raises ValueError if X is empty or not 2d
        """
        # step 0: same checks as batch fit — 2d, non-empty, float
        X = validate_array_like(X, name="X").astype(float)
        validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("StandardScaler expects 2D input.")

        # step 1: fold this chunk into running mean/var (Welford via RunningStats)
        self._running_stats.update(X)

        # step 2: pull mean and std out, fix zero std so transform won't divide by 0
        self.mean_, scale = self._running_stats.result()
        self.scale_ = scale
        self.scale_[self.scale_ == 0.0] = 1.0

        # step 3: keep track of how many rows we've seen
        self.n_samples_seen_ = self._running_stats.n
        return self

    def fit(self, X, y=None):
        """Reset stats and learn from X in one go (calls partial_fit internally)

        X - 2d array (n_samples, n_features)
        y - not used

        Returns self
        """
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.scale_ = None
        self._running_stats.reset()
        return self.partial_fit(X, y)


class SimpleImputer(BatchSimpleImputer):
    """SimpleImputer you can train one chunk at a time"""

    def __init__(self, strategy="mean"):
        """Start unfitted

        strategy - only 'mean' supported for now (same as batch version)
        """
        super().__init__(strategy=strategy)
        self.n_samples_seen_ = 0  # running count of rows fed in
        self._sum_ = None  # running sum per column (ignoring NaN)
        self._count_ = None  # how many non-NaN values per column

    def partial_fit(self, X, y=None):
        """Update fill values from one chunk of data

        X - 2d array (n_samples, n_features), can contain NaN
        y - not used

        Returns self so you can chain partial_fit calls

        Raises ValueError if X is empty or not 2d
        """
        # step 1: same checks as batch fit — 2d, non-empty, float
        X = validate_array_like(X, name="X").astype(float)
        validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")

        n_features = X.shape[1]

        # step 2: first chunk? set up empty sum/count arrays
        if self._sum_ is None:
            self._sum_ = np.zeros(n_features, dtype=float)
            self._count_ = np.zeros(n_features, dtype=float)

        # step 3: add non-NaN values from this chunk into running totals
        for j in range(n_features):
            col = X[:, j]
            valid = ~np.isnan(col)
            self._sum_[j] += np.sum(col[valid])
            self._count_[j] += valid.sum()

        # step 4: fill value per column = sum / count (NaN if column had no values yet)
        self.statistics_ = np.divide(
            self._sum_,
            self._count_,
            out=np.full(n_features, np.nan, dtype=float),
            where=self._count_ > 0,
        )

        self.n_samples_seen_ += X.shape[0]
        return self

    def fit(self, X, y=None):
        """Reset stats and learn from X in one go (calls partial_fit internally)

        X - 2d array (n_samples, n_features)
        y - not used

        Returns self
        """
        self.n_samples_seen_ = 0
        self.statistics_ = None
        self._sum_ = None
        self._count_ = None
        return self.partial_fit(X, y)


class OneHotEncoder(BatchOneHotEncoder):
    """OneHotEncoder that grows categories as new chunks arrive"""

    def __init__(self, handle_unknown="error"):
        """Start unfitted

        handle_unknown - 'error' or 'ignore' (same as batch version)
        """
        super().__init__(handle_unknown=handle_unknown)

    def partial_fit(self, X, y=None):
        """Add categories seen in this chunk

        X - 2d categorical array (n_samples, n_features)
        y - not used

        Returns self so you can chain partial_fit calls

        Raises ValueError if X is empty or not 2d
        """
        # step 1: same checks as batch fit — 2d, non-empty
        X = validate_array_like(X, name="X")
        validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")

        n_features = X.shape[1]

        # step 2: first chunk? store sorted unique categories per column
        if self.categories_ is None:
            self.n_features_in_ = n_features
            self.categories_ = [np.unique(X[:, i]) for i in range(n_features)]
        else:
            # later chunks must have same number of columns
            if n_features != self.n_features_in_:
                raise ValueError(
                    "Input has different number of features than fit data."
                )

            # step 3: merge in any new category values and keep them sorted
            for i in range(n_features):
                self.categories_[i] = np.unique(
                    np.concatenate([self.categories_[i], np.unique(X[:, i])])
                )

        return self

    def fit(self, X, y=None):
        """Reset categories and learn from X in one go (calls partial_fit internally)

        X - 2d categorical array (n_samples, n_features)
        y - not used

        Returns self
        """
        self.categories_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y)

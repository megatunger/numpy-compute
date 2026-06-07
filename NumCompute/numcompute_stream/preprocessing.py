"""Preprocessing that learns from data chunk by chunk"""

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

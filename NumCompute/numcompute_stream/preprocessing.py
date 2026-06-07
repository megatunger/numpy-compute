"""Preprocessing that learns from data chunk by chunk"""

from numcompute.preprocessing import StandardScaler as BatchStandardScaler


class StandardScaler(BatchStandardScaler):
    """StandardScaler you can train one chunk at a time"""

    def __init__(self):
        """Start unfitted with no samples seen yet"""
        super().__init__()
        self.n_samples_seen_ = 0  # running count of rows fed in

    def partial_fit(self, X, y=None):
        """Update mean and scale from one chunk of data

        X - 2d array (n_samples, n_features)
        y - not used, just here to match sklearn-style API

        Returns self so you can chain partial_fit calls

        Raises ValueError if X is empty or not 2d
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """Reset stats and learn from X in one go (calls partial_fit internally)

        X - 2d array (n_samples, n_features)
        y - not used

        Returns self
        """
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.scale_ = None
        return self.partial_fit(X, y)

"""Streaming statistics with incremental chunk updates."""


class RunningStats:
    """Per-feature running mean and variance (Welford's algorithm)."""

    def __init__(self, ddof=0):
        self.ddof = ddof
        self.n = 0
        self.mean_ = None
        self.var_ = None

    def update(self, X_chunk):
        """Incorporate a batch of samples. X_chunk shape: (n_samples, n_features)."""
        raise NotImplementedError

    def result(self):
        """Return mean_ and std_ per feature."""
        raise NotImplementedError

    def reset(self):
        """Clear accumulated state."""
        self.n = 0
        self.mean_ = None
        self.var_ = None
        return self


class StreamingHistogram:
    """Incrementally accumulate a histogram over streamed data."""

    def __init__(self, bins=10, range=None):
        self.bins = bins
        self.range = range
        self.bin_edges_ = None
        self.counts_ = None

    def update(self, X_chunk):
        """Add samples from a chunk to the running histogram."""
        raise NotImplementedError

    def result(self):
        """Return accumulated counts and bin edges."""
        raise NotImplementedError

    def reset(self):
        """Clear accumulated counts and bin edges."""
        self.bin_edges_ = None
        self.counts_ = None
        return self


def update_stats(stats, X_chunk):
    """Update a streaming stats object that exposes ``update``."""
    raise NotImplementedError

"""Streaming statistics with incremental chunk updates."""

import numpy as np

from numcompute.utils import validate_array_like, validate_non_empty_array


class RunningStats:
    """Per-feature running mean and variance (Welford's algorithm)."""

    def __init__(self, ddof=0):
        self.ddof = ddof
        self.n = 0  # how many rows we've seen total
        self.mean_ = None  # running average per column
        self.var_ = None  # running variance per column
        self._m2 = None  # hidden helper — sum of squared gaps from the mean

    def update(self, X_chunk):
        """Incorporate a batch of samples. X_chunk shape: (n_samples, n_features)."""
        # step 1: sanity check — needs to be a real non-empty array
        X = validate_array_like(X_chunk, name="X_chunk")
        validate_non_empty_array(X, name="X_chunk")

        # step 2: normalise shape — 1d list becomes one column, reject weird dims
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(
                f"Expect X_chunk to be 1-D or 2-D, got {X.ndim} dimensions"
            )

        # step 3: work in float64 so big numbers don't blow up the math
        X = X.astype(np.float64, copy=False)
        chunk_n, n_features = X.shape

        # step 4: figure out this chunk's own mean and spread (per feature)
        chunk_mean = X.mean(axis=0)
        chunk_m2 = np.sum((X - chunk_mean) ** 2, axis=0)

        # step 5: first chunk? just adopt its stats directly
        if self.n == 0:
            self.mean_ = chunk_mean
            self._m2 = chunk_m2
            self.n = chunk_n
        else:
            # every chunk after must have the same number of columns
            if n_features != self.mean_.shape[0]:
                raise ValueError(
                    "Expect all chunks to have the same number of features, "
                    f"got {n_features} and {self.mean_.shape[0]}"
                )

            # step 6: merge old stats + new chunk into one combined picture
            # (like you dumped both piles of data together without storing them)
            new_n = self.n + chunk_n
            delta = chunk_mean - self.mean_
            self.mean_ = self.mean_ + delta * chunk_n / new_n
            self._m2 = (
                self._m2
                + chunk_m2
                + delta ** 2 * self.n * chunk_n / new_n
            )
            self.n = new_n

        # step 7: turn the hidden _m2 into actual variance
        self._refresh_var()
        return self

    def result(self):
        """Return mean_ and std_ per feature."""
        if self.n == 0:
            raise ValueError("No samples have been added")

        # variance -> standard deviation (just take the square root)
        std_ = np.sqrt(self.var_)
        return self.mean_.copy(), std_.copy()

    def reset(self):
        """Clear accumulated state."""
        self.n = 0
        self.mean_ = None
        self.var_ = None
        self._m2 = None
        return self

    def _refresh_var(self):
        # spread = total squared gaps / (n - ddof)
        # ddof=0 -> population var (numpy default), ddof=1 -> sample var
        divisor = self.n - self.ddof
        if divisor <= 0:
            self.var_ = np.zeros_like(self.mean_)
        else:
            self.var_ = self._m2 / divisor


class StreamingHistogram:
    """Incrementally accumulate a histogram over streamed data."""

    def __init__(self, bins=10, range=None):
        self.bins = bins
        self.range = range
        self.bin_edges_ = None  # where the bucket walls are
        self.counts_ = None  # how many values landed in each bucket

    def update(self, X_chunk):
        """Add samples from a chunk to the running histogram."""
        # step 1: validate and flatten to a simple 1d list of values
        X = validate_array_like(X_chunk, name="X_chunk")
        validate_non_empty_array(X, name="X_chunk")
        X = X.ravel()

        # step 2: first time? lock in the bin edges so later chunks use same buckets
        if self.bin_edges_ is None:
            self.bin_edges_ = np.histogram_bin_edges(
                X, bins=self.bins, range=self.range
            )
            self.counts_ = np.zeros(len(self.bin_edges_) - 1, dtype=np.int64)

        # step 3: count this chunk's values and add to the running total
        chunk_counts, _ = np.histogram(X, bins=self.bin_edges_)
        self.counts_ += chunk_counts
        return self

    def result(self):
        """Return accumulated counts and bin edges."""
        if self.counts_ is None:
            raise ValueError("No samples have been added")

        return self.counts_.copy(), self.bin_edges_.copy()

    def reset(self):
        """Clear accumulated counts and bin edges."""
        self.bin_edges_ = None
        self.counts_ = None
        return self


def update_stats(stats, X_chunk):
    """Update a streaming stats object that exposes ``update``."""
    # just a shortcut — hand the chunk to whatever stats object you have
    return stats.update(X_chunk)

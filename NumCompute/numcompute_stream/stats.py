"""Stats that update chunk by chunk instead of needing all the data at once"""

import numpy as np

from numcompute.utils import validate_array_like, validate_non_empty_array


class RunningStats:
    """Keeps track of mean and variance as new data comes in (uses Welford's method)"""

    def __init__(self, ddof=0):
        """Start with empty stats

        ddof - basically controls variance formula, 0 is same as np.var, 1 is
        same as when you do np.std(..., ddof=1) for sample std
        """
        self.ddof = ddof
        self.n = 0  # how many rows we've seen total
        self.mean_ = None  # running average per column
        self.var_ = None  # running variance per column
        self._m2 = None  # hidden helper — sum of squared gaps from the mean

    def update(self, X_chunk):
        """Feed in one chunk and update the running stats

        X_chunk - your new data, can be 2d (rows x features) or just a 1d list
        (we treat that as one feature column)

        Returns self so you can do stats.update(a).update(b)

        Raises ValueError if chunk is empty, wrong shape, or different number
        of columns than before
        """
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
        """Give back mean and std for each feature

        Returns (mean_, std_) as two arrays

        Raises ValueError if you haven't called update yet
        """
        if self.n == 0:
            raise ValueError("No samples have been added")

        # variance -> standard deviation (just take the square root)
        std_ = np.sqrt(self.var_)
        return self.mean_.copy(), std_.copy()

    def reset(self):
        """Clear everything and go back to empty"""
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
    """Histogram that you can build up over multiple chunks"""

    def __init__(self, bins=10, range=None):
        """Start an empty histogram

        bins - how many bins, or pass bin edges like np.histogram
        range - optional (min, max), stuff outside gets ignored, if None we
        pick edges from the first chunk
        """
        self.bins = bins
        self.range = range
        self.bin_edges_ = None  # where the bucket walls are
        self.counts_ = None  # how many values landed in each bucket

    def update(self, X_chunk):
        """Add more values into the histogram

        X_chunk - array of values, any shape is fine we flatten it

        Returns self

        Raises ValueError if empty
        """
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
        """Return the histogram so far

        Returns (counts_, bin_edges_)

        Raises ValueError if no data added yet
        """
        if self.counts_ is None:
            raise ValueError("No samples have been added")

        return self.counts_.copy(), self.bin_edges_.copy()

    def reset(self):
        """Reset counts and bin edges"""
        self.bin_edges_ = None
        self.counts_ = None
        return self


def update_stats(stats, X_chunk):
    """Quick helper - just calls stats.update(X_chunk) for you

    stats - any object with an update method (RunningStats, StreamingHistogram, etc)
    X_chunk - the new data to add

    Returns the stats object back
    """
    # just a shortcut — hand the chunk to whatever stats object you have
    return stats.update(X_chunk)

"""Metrics that build up over chunks instead of needing all labels at once"""

from numcompute.utils import validate_metrics_array


class StreamingAccuracy:
    """Keeps a running accuracy score as new predictions come in"""

    def __init__(self):
        """Start with zero correct and zero total"""
        self.correct_ = 0  # how many predictions matched
        self.total_ = 0  # how many predictions we've seen

    def update(self, y_true, y_pred):
        """Add one chunk of labels and predictions

        y_true - ground truth labels for this chunk
        y_pred - model predictions for this chunk, same shape as y_true

        Returns self so you can chain updates

        Raises ValueError if arrays are empty, wrong shape, or incompatible
        """
        # step 1: sanity check — same shape, non-empty, compatible dtypes
        y_true, y_pred = validate_metrics_array(y_true, y_pred)

        # step 2: count how many got it right in this chunk
        chunk_correct = (y_true == y_pred).sum()

        # step 3: add chunk totals to the running counts
        self.correct_ += int(chunk_correct)
        self.total_ += y_true.size
        return self

    def result(self):
        """Get accuracy so far as a float between 0 and 1

        Returns correct / total across all chunks

        Raises ValueError if you haven't called update yet
        """
        if self.total_ == 0:
            raise ValueError("No samples have been added")

        return self.correct_ / self.total_

    def reset(self):
        """Clear counts and go back to zero"""
        self.correct_ = 0
        self.total_ = 0
        return self

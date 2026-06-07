"""Metrics that build up over chunks instead of needing all labels at once"""


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
        raise NotImplementedError

    def result(self):
        """Get accuracy so far as a float between 0 and 1

        Returns correct / total across all chunks

        Raises ValueError if you haven't called update yet
        """
        raise NotImplementedError

    def reset(self):
        """Clear counts and go back to zero"""
        self.correct_ = 0
        self.total_ = 0
        return self

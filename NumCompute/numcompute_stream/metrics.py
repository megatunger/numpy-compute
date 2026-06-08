"""Metrics that build up over chunks instead of needing all labels at once"""

import numpy as np

from numcompute.utils import validate_array_like, validate_metrics_array


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


class StreamingConfusionMatrix:
    """Builds a confusion matrix as chunks of predictions arrive"""

    def __init__(self, labels=None):
        """Start with an empty count table

        labels - optional fixed class order for result(), None means infer from data
        """
        self.labels_ = None if labels is None else validate_array_like(labels).ravel()
        self.counts_ = {}

    def update(self, y_true, y_pred):
        """Add one chunk to the running confusion counts

        y_true - ground truth labels for this chunk
        y_pred - model predictions for this chunk, same shape as y_true

        Returns self so you can chain updates

        Raises ValueError if arrays are empty, wrong shape, or incompatible
        """
        y_true, y_pred = validate_metrics_array(y_true, y_pred)

        for true_label, pred_label in zip(y_true.ravel(), y_pred.ravel()):
            key = (true_label, pred_label)
            self.counts_[key] = self.counts_.get(key, 0) + 1

        return self

    def result(self):
        """Get the confusion matrix built from all chunks so far

        Returns a 2d array — rows are true labels, columns are predicted labels

        Raises ValueError if you haven't called update yet
        """
        if not self.counts_:
            raise ValueError("No samples have been added")

        if self.labels_ is None:
            true_labels = np.array([key[0] for key in self.counts_])
            pred_labels = np.array([key[1] for key in self.counts_])
            labels = np.union1d(true_labels, pred_labels)
        else:
            labels = self.labels_

        n_labels = labels.size
        matrix = np.zeros((n_labels, n_labels), dtype=np.float32)
        label_to_index = {label: index for index, label in enumerate(labels)}

        for (true_label, pred_label), count in self.counts_.items():
            row = label_to_index[true_label]
            col = label_to_index[pred_label]
            matrix[row, col] += count

        return matrix

    def reset(self):
        """Clear counts and go back to empty"""
        self.counts_ = {}
        return self


class StreamingPrecision:
    """Running precision for one positive class (same idea as numcompute.metrics)"""

    def __init__(self, pos_label=1):
        """Track tp and fn for the chosen positive label

        pos_label - which class counts as positive (default 1)
        """
        self.pos_label = pos_label
        self.tp_ = 0
        self.fn_ = 0

    def update(self, y_true, y_pred):
        """Add one chunk of labels and predictions

        y_true - ground truth labels for this chunk
        y_pred - model predictions for this chunk, same shape as y_true

        Returns self so you can chain updates

        Raises ValueError if arrays are empty, wrong shape, or incompatible
        """
        y_true, y_pred = validate_metrics_array(y_true, y_pred)

        true_pos = y_true == self.pos_label
        pred_pos = y_pred == self.pos_label
        self.tp_ += int((true_pos & pred_pos).sum())
        self.fn_ += int((true_pos & ~pred_pos).sum())
        return self

    def result(self):
        """Get precision so far

        Raises ValueError if you haven't called update yet
        """
        if self.tp_ == 0 and self.fn_ == 0:
            raise ValueError("No samples have been added")

        if self.tp_ + self.fn_ == 0:
            return 0.0

        return self.tp_ / (self.tp_ + self.fn_)

    def reset(self):
        """Clear counts and go back to zero"""
        self.tp_ = 0
        self.fn_ = 0
        return self


class StreamingRecall:
    """Running recall for one positive class (same idea as numcompute.metrics)"""

    def __init__(self, pos_label=1):
        """Track tp and fp for the chosen positive label

        pos_label - which class counts as positive (default 1)
        """
        self.pos_label = pos_label
        self.tp_ = 0
        self.fp_ = 0

    def update(self, y_true, y_pred):
        """Add one chunk of labels and predictions

        y_true - ground truth labels for this chunk
        y_pred - model predictions for this chunk, same shape as y_true

        Returns self so you can chain updates

        Raises ValueError if arrays are empty, wrong shape, or incompatible
        """
        y_true, y_pred = validate_metrics_array(y_true, y_pred)

        true_pos = y_true == self.pos_label
        pred_pos = y_pred == self.pos_label
        self.tp_ += int((true_pos & pred_pos).sum())
        self.fp_ += int((~true_pos & pred_pos).sum())
        return self

    def result(self):
        """Get recall so far

        Raises ValueError if you haven't called update yet
        """
        if self.tp_ == 0 and self.fp_ == 0:
            raise ValueError("No samples have been added")

        if self.tp_ + self.fp_ == 0:
            return 0.0

        return self.tp_ / (self.tp_ + self.fp_)

    def reset(self):
        """Clear counts and go back to zero"""
        self.tp_ = 0
        self.fp_ = 0
        return self


class StreamingF1:
    """Running F1 for one positive class, from streaming precision and recall"""

    def __init__(self, pos_label=1):
        """Track counts needed for F1 on the chosen positive label

        pos_label - which class counts as positive (default 1)
        """
        self.pos_label = pos_label
        self.tp_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def update(self, y_true, y_pred):
        """Add one chunk of labels and predictions

        y_true - ground truth labels for this chunk
        y_pred - model predictions for this chunk, same shape as y_true

        Returns self so you can chain updates

        Raises ValueError if arrays are empty, wrong shape, or incompatible
        """
        y_true, y_pred = validate_metrics_array(y_true, y_pred)

        true_pos = y_true == self.pos_label
        pred_pos = y_pred == self.pos_label
        self.tp_ += int((true_pos & pred_pos).sum())
        self.fp_ += int((~true_pos & pred_pos).sum())
        self.fn_ += int((true_pos & ~pred_pos).sum())
        return self

    def result(self):
        """Get F1 so far

        Raises ValueError if you haven't called update yet
        """
        if self.tp_ == 0 and self.fp_ == 0 and self.fn_ == 0:
            raise ValueError("No samples have been added")

        precision = 0.0 if self.tp_ + self.fn_ == 0 else self.tp_ / (self.tp_ + self.fn_)
        recall = 0.0 if self.tp_ + self.fp_ == 0 else self.tp_ / (self.tp_ + self.fp_)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def reset(self):
        """Clear counts and go back to zero"""
        self.tp_ = 0
        self.fp_ = 0
        self.fn_ = 0
        return self

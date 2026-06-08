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


class StreamingAUC:
    """Binary ROC AUC that stores scores as chunks arrive"""

    def __init__(self, pos_label=1):
        """Start with no labels or scores yet

        pos_label - label counted as the positive class
        """
        self.pos_label = pos_label
        self.y_true_ = []
        self.y_score_ = []
        self.total_ = 0

    def update(self, y_true, y_score):
        """Add one chunk of labels and scores

        y_true - true binary labels for this chunk
        y_score - score for the positive class, same shape as y_true

        Returns self so you can chain updates
        """
        y_true, y_score = _validate_binary_score_inputs(
            y_true, y_score, self.pos_label
        )
        self.y_true_.append(y_true)
        self.y_score_.append(y_score)
        self.total_ += y_true.size
        return self

    def result(self):
        """Get ROC AUC across all chunks seen so far"""
        if self.total_ == 0:
            raise ValueError("No samples have been added")

        y_true = np.concatenate(self.y_true_)
        y_score = np.concatenate(self.y_score_)
        return _binary_auc(y_true, y_score)

    def reset(self):
        """Clear stored labels and scores"""
        self.y_true_ = []
        self.y_score_ = []
        self.total_ = 0
        return self


class _RollingWindowBase:
    """Shared storage for metrics that only use the latest samples"""

    def __init__(self, window_size, pos_label=1):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.pos_label = pos_label
        self.y_true_ = np.array([])
        self.y_pred_ = np.array([])

    def _update_window(self, y_true, y_pred):
        y_true, y_pred = validate_metrics_array(y_true, y_pred)
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        self.y_true_ = np.concatenate([self.y_true_, y_true])[-self.window_size:]
        self.y_pred_ = np.concatenate([self.y_pred_, y_pred])[-self.window_size:]
        return self

    def _check_has_samples(self):
        if self.y_true_.size == 0:
            raise ValueError("No samples have been added")

    def reset(self):
        """Clear the rolling window"""
        self.y_true_ = np.array([])
        self.y_pred_ = np.array([])
        return self


class RollingWindowAccuracy(_RollingWindowBase):
    """Accuracy over the most recent window_size samples"""

    def update(self, y_true, y_pred):
        """Add one chunk and keep only the newest window_size samples"""
        return self._update_window(y_true, y_pred)

    def result(self):
        """Get accuracy for the current window"""
        self._check_has_samples()
        return float(np.mean(self.y_true_ == self.y_pred_))


class RollingWindowPrecision(_RollingWindowBase):
    """Precision over the most recent window_size samples"""

    def update(self, y_true, y_pred):
        """Add one chunk and keep only the newest window_size samples"""
        return self._update_window(y_true, y_pred)

    def result(self):
        """Get precision for the current window"""
        self._check_has_samples()
        tp, fp, _ = _binary_counts(self.y_true_, self.y_pred_, self.pos_label)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)


class RollingWindowRecall(_RollingWindowBase):
    """Recall over the most recent window_size samples"""

    def update(self, y_true, y_pred):
        """Add one chunk and keep only the newest window_size samples"""
        return self._update_window(y_true, y_pred)

    def result(self):
        """Get recall for the current window"""
        self._check_has_samples()
        tp, _, fn = _binary_counts(self.y_true_, self.y_pred_, self.pos_label)
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)


class RollingWindowF1(_RollingWindowBase):
    """F1 score over the most recent window_size samples"""

    def update(self, y_true, y_pred):
        """Add one chunk and keep only the newest window_size samples"""
        return self._update_window(y_true, y_pred)

    def result(self):
        """Get F1 for the current window"""
        self._check_has_samples()
        tp, fp, fn = _binary_counts(self.y_true_, self.y_pred_, self.pos_label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class RollingWindowAUC:
    """Binary ROC AUC over the most recent window_size scored samples"""

    def __init__(self, window_size, pos_label=1):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.pos_label = pos_label
        self.y_true_ = np.array([])
        self.y_score_ = np.array([], dtype=float)

    def update(self, y_true, y_score):
        """Add scored samples and keep only the newest window_size samples"""
        y_true, y_score = _validate_binary_score_inputs(
            y_true, y_score, self.pos_label
        )
        self.y_true_ = np.concatenate([self.y_true_, y_true])[-self.window_size:]
        self.y_score_ = np.concatenate([self.y_score_, y_score])[-self.window_size:]
        return self

    def result(self):
        """Get AUC for the current window"""
        if self.y_true_.size == 0:
            raise ValueError("No samples have been added")
        return _binary_auc(self.y_true_, self.y_score_)

    def reset(self):
        """Clear the rolling score window"""
        self.y_true_ = np.array([])
        self.y_score_ = np.array([], dtype=float)
        return self


def _validate_binary_score_inputs(y_true, y_score, pos_label):
    y_true = validate_array_like(y_true, name="y_true")
    y_score = validate_array_like(y_score, name="y_score")

    if y_true.size == 0 or y_score.size == 0:
        raise ValueError("y_true and y_score must be non-empty")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {y_true.shape} and {y_score.shape}"
        )

    y_binary = (y_true.ravel() == pos_label).astype(int)
    scores = y_score.ravel().astype(float)

    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must only contain finite values")

    return y_binary, scores


def _binary_counts(y_true, y_pred, pos_label):
    true_pos = y_true == pos_label
    pred_pos = y_pred == pos_label
    tp = int((true_pos & pred_pos).sum())
    fp = int((~true_pos & pred_pos).sum())
    fn = int((true_pos & ~pred_pos).sum())
    return tp, fp, fn


def _binary_auc(y_true, y_score):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative sample")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_ranks = np.arange(1, y_score.size + 1, dtype=float)

    _, starts, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    for start, count in zip(starts, counts):
        if count > 1:
            sorted_ranks[start:start + count] = sorted_ranks[start:start + count].mean()

    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

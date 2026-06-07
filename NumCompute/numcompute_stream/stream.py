"""Orchestration for training and scoring one chunk at a time"""

import numpy as np

from numcompute.utils import validate_array_like, validate_non_empty_array
from numcompute_stream.metrics import StreamingAccuracy


class StreamTrainer:
    """Wraps a pipeline plus metrics and keeps a log per chunk"""

    def __init__(self, pipeline, metrics=None):
        """Hook up a streaming pipeline and optional metric tracker

        pipeline - Pipeline with partial_fit and predict
        metrics - something with update(y_true, y_pred) and result(), defaults
        to StreamingAccuracy
        """
        self.pipeline = pipeline
        self.metrics = metrics if metrics is not None else StreamingAccuracy()
        self.log_ = []
        self._chunk_index = 0

    def fit_chunk(self, X, y):
        """Train on one chunk, predict it, update metrics, and log results

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns self so you can loop over chunks

        Raises ValueError if X/y empty or shapes don't match
        """
        # step 1: sanity check inputs
        X = validate_array_like(X, name="X").astype(float)
        y = validate_array_like(y, name="y")
        validate_non_empty_array(X, name="X")
        validate_non_empty_array(y, name="y")

        if X.ndim != 2:
            raise ValueError("StreamTrainer expects 2D input for X")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # step 2: train the pipeline on this chunk
        self.pipeline.partial_fit(X, y)

        # step 3: score the same chunk and update running metrics
        y_pred = self.pipeline.predict(X)
        chunk_accuracy = float(np.mean(y_pred == y))
        if hasattr(self.metrics, "update"):
            self.metrics.update(y, y_pred)

        # step 4: stash a log row for this chunk
        entry = {
            "chunk_index": self._chunk_index,
            "n_samples": int(X.shape[0]),
            "chunk_accuracy": chunk_accuracy,
            "memory_bytes": int(X.nbytes + y.nbytes),
        }
        if hasattr(self.metrics, "result"):
            entry["cumulative_accuracy"] = float(self.metrics.result())

        self.log_.append(entry)
        self._chunk_index += 1
        return self

    def score_chunk(self, X, y):
        """Predict on a chunk and return accuracy without training

        X - 2d array (n_samples, n_features)
        y - 1d array of class labels

        Returns chunk accuracy as a float between 0 and 1

        Raises ValueError if X/y empty or shapes don't match
        """
        # step 1: sanity check inputs
        X = validate_array_like(X, name="X").astype(float)
        y = validate_array_like(y, name="y")
        validate_non_empty_array(X, name="X")
        validate_non_empty_array(y, name="y")

        if X.ndim != 2:
            raise ValueError("StreamTrainer expects 2D input for X")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # step 2: predict and compare to labels
        y_pred = self.pipeline.predict(X)
        return float(np.mean(y_pred == y))

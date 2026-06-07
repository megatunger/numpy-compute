import unittest

import numpy as np

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer


class DummyClassifier:
    def __init__(self):
        self.classes_ = None
        self.majority_ = None
        self.n_seen_ = 0

    def partial_fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_ = np.bincount(y.astype(int)).argmax()
        self.n_seen_ += len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def fit(self, X, y=None):
        return self.partial_fit(X, y)


class TestStreamTrainer(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyClassifier()),
        ])
        self.chunk1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y1 = np.array([0, 0])
        self.chunk2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        self.y2 = np.array([1, 1])

    def test_fit_chunk_returns_self(self):
        trainer = StreamTrainer(self.pipeline)
        out = trainer.fit_chunk(self.chunk1, self.y1)
        self.assertIs(out, trainer)

    def test_fit_chunk_trains_pipeline(self):
        trainer = StreamTrainer(self.pipeline)
        trainer.fit_chunk(self.chunk1, self.y1)
        trainer.fit_chunk(self.chunk2, self.y2)

        self.assertEqual(self.pipeline.named_steps["model"].n_seen_, 4)

    def test_fit_chunk_appends_log(self):
        trainer = StreamTrainer(self.pipeline)
        trainer.fit_chunk(self.chunk1, self.y1)

        self.assertEqual(len(trainer.log_), 1)
        self.assertEqual(trainer.log_[0]["chunk_index"], 0)
        self.assertEqual(trainer.log_[0]["n_samples"], 2)
        self.assertIn("chunk_accuracy", trainer.log_[0])
        self.assertIn("cumulative_accuracy", trainer.log_[0])
        self.assertIn("memory_bytes", trainer.log_[0])

    def test_fit_chunk_updates_cumulative_metric(self):
        metric = StreamingAccuracy()
        trainer = StreamTrainer(self.pipeline, metrics=metric)
        trainer.fit_chunk(self.chunk1, self.y1)

        self.assertEqual(metric.result(), 1.0)
        self.assertEqual(trainer.log_[0]["cumulative_accuracy"], 1.0)

    def test_score_chunk_returns_chunk_accuracy(self):
        trainer = StreamTrainer(self.pipeline)
        trainer.fit_chunk(self.chunk1, self.y1)

        score = trainer.score_chunk(self.chunk2, self.y2)
        self.assertEqual(score, 0.0)

    def test_score_chunk_does_not_train(self):
        trainer = StreamTrainer(self.pipeline)
        trainer.fit_chunk(self.chunk1, self.y1)
        seen_before = self.pipeline.named_steps["model"].n_seen_

        trainer.score_chunk(self.chunk2, self.y2)

        self.assertEqual(self.pipeline.named_steps["model"].n_seen_, seen_before)

    def test_two_fit_chunks_grow_log(self):
        trainer = StreamTrainer(self.pipeline)
        trainer.fit_chunk(self.chunk1, self.y1)
        trainer.fit_chunk(self.chunk2, self.y2)

        self.assertEqual(len(trainer.log_), 2)
        self.assertEqual(trainer.log_[1]["chunk_index"], 1)

    def test_fit_chunk_empty_raises(self):
        trainer = StreamTrainer(self.pipeline)
        with self.assertRaises(ValueError):
            trainer.fit_chunk(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()

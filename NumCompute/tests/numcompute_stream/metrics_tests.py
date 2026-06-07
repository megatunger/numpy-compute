import unittest

import numpy as np

from numcompute.metrics import accuracy_score
from numcompute_stream.metrics import StreamingAccuracy


class TestStreamingAccuracy(unittest.TestCase):
    def setUp(self):
        self.y_true1 = np.array([1, 0, 1, 1])
        self.y_pred1 = np.array([1, 0, 0, 1])
        self.y_true2 = np.array([0, 1])
        self.y_pred2 = np.array([0, 1])

    def test_two_chunks_match_batch_accuracy(self):
        metric = StreamingAccuracy()
        metric.update(self.y_true1, self.y_pred1)
        metric.update(self.y_true2, self.y_pred2)

        y_true = np.concatenate([self.y_true1, self.y_true2])
        y_pred = np.concatenate([self.y_pred1, self.y_pred2])

        self.assertAlmostEqual(metric.result(), accuracy_score(y_true, y_pred))

    def test_single_chunk_matches_batch(self):
        metric = StreamingAccuracy()
        metric.update(self.y_true1, self.y_pred1)

        self.assertAlmostEqual(
            metric.result(), accuracy_score(self.y_true1, self.y_pred1)
        )

    def test_perfect_accuracy(self):
        metric = StreamingAccuracy()
        y = np.array([1, 2, 3])
        metric.update(y, y)

        self.assertEqual(metric.result(), 1.0)

    def test_zero_accuracy(self):
        metric = StreamingAccuracy()
        metric.update(np.array([1, 1, 1]), np.array([0, 0, 0]))

        self.assertEqual(metric.result(), 0.0)

    def test_empty_chunk_raises(self):
        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.update(np.array([]), np.array([]))

    def test_result_before_update_raises(self):
        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.result()

    def test_reset_clears_state(self):
        metric = StreamingAccuracy()
        metric.update(self.y_true1, self.y_pred1)
        metric.reset()

        self.assertEqual(metric.correct_, 0)
        self.assertEqual(metric.total_, 0)

    def test_update_returns_self(self):
        metric = StreamingAccuracy()
        out = metric.update(self.y_true1, self.y_pred1)
        self.assertIs(out, metric)


if __name__ == "__main__":
    unittest.main()

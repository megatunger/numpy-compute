import unittest

import numpy as np

from numcompute.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
)


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


class TestStreamingConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.y_true1 = np.array([0, 1, 1, 0])
        self.y_pred1 = np.array([0, 1, 0, 0])
        self.y_true2 = np.array([1, 0])
        self.y_pred2 = np.array([1, 1])

    def test_two_chunks_match_batch(self):
        metric = StreamingConfusionMatrix()
        metric.update(self.y_true1, self.y_pred1)
        metric.update(self.y_true2, self.y_pred2)

        y_true = np.concatenate([self.y_true1, self.y_true2])
        y_pred = np.concatenate([self.y_pred1, self.y_pred2])

        np.testing.assert_array_equal(
            metric.result(),
            confusion_matrix(y_true, y_pred, labels=[0, 1]),
        )

    def test_reset_clears_state(self):
        metric = StreamingConfusionMatrix()
        metric.update(self.y_true1, self.y_pred1)
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()

    def test_result_before_update_raises(self):
        metric = StreamingConfusionMatrix()
        with self.assertRaises(ValueError):
            metric.result()


class TestStreamingPrecision(unittest.TestCase):
    def setUp(self):
        self.y_true1 = np.array([1, 0, 0, 1])
        self.y_pred1 = np.array([1, 1, 0, 0])
        self.y_true2 = np.array([0, 1])
        self.y_pred2 = np.array([0, 0])

    def test_two_chunks_match_batch(self):
        metric = StreamingPrecision()
        metric.update(self.y_true1, self.y_pred1)
        metric.update(self.y_true2, self.y_pred2)

        y_true = np.concatenate([self.y_true1, self.y_true2])
        y_pred = np.concatenate([self.y_pred1, self.y_pred2])

        self.assertAlmostEqual(metric.result(), precision_score(y_true, y_pred))

    def test_reset_clears_state(self):
        metric = StreamingPrecision()
        metric.update(self.y_true1, self.y_pred1)
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()


class TestStreamingRecall(unittest.TestCase):
    def setUp(self):
        self.y_true1 = np.array([1, 1, 0, 0])
        self.y_pred1 = np.array([1, 0, 1, 0])
        self.y_true2 = np.array([0, 1])
        self.y_pred2 = np.array([0, 1])

    def test_two_chunks_match_batch(self):
        metric = StreamingRecall()
        metric.update(self.y_true1, self.y_pred1)
        metric.update(self.y_true2, self.y_pred2)

        y_true = np.concatenate([self.y_true1, self.y_true2])
        y_pred = np.concatenate([self.y_pred1, self.y_pred2])

        self.assertAlmostEqual(metric.result(), recall_score(y_true, y_pred))

    def test_reset_clears_state(self):
        metric = StreamingRecall()
        metric.update(self.y_true1, self.y_pred1)
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()


class TestStreamingF1(unittest.TestCase):
    def setUp(self):
        self.y_true1 = np.array([1, 1, 0, 0])
        self.y_pred1 = np.array([1, 0, 1, 0])
        self.y_true2 = np.array([0, 1])
        self.y_pred2 = np.array([0, 1])

    def test_two_chunks_match_batch(self):
        metric = StreamingF1()
        metric.update(self.y_true1, self.y_pred1)
        metric.update(self.y_true2, self.y_pred2)

        y_true = np.concatenate([self.y_true1, self.y_true2])
        y_pred = np.concatenate([self.y_pred1, self.y_pred2])

        self.assertAlmostEqual(metric.result(), f1_score(y_true, y_pred))

    def test_reset_clears_state(self):
        metric = StreamingF1()
        metric.update(self.y_true1, self.y_pred1)
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()


if __name__ == "__main__":
    unittest.main()

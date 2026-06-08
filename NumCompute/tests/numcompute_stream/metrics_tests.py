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
    RollingWindowAUC,
    RollingWindowAccuracy,
    RollingWindowF1,
    RollingWindowPrecision,
    RollingWindowRecall,
    StreamingAUC,
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


class TestStreamingAUC(unittest.TestCase):
    def test_two_chunks_match_expected_auc(self):
        metric = StreamingAUC()
        metric.update(np.array([0, 0]), np.array([0.1, 0.4]))
        metric.update(np.array([1, 1]), np.array([0.35, 0.8]))

        self.assertAlmostEqual(metric.result(), 0.75)

    def test_tied_scores_use_average_ranks(self):
        metric = StreamingAUC()
        metric.update(
            np.array([0, 1, 0, 1]),
            np.array([0.5, 0.5, 0.2, 0.8]),
        )

        self.assertAlmostEqual(metric.result(), 0.875)

    def test_result_before_update_raises(self):
        metric = StreamingAUC()
        with self.assertRaises(ValueError):
            metric.result()

    def test_single_class_raises(self):
        metric = StreamingAUC()
        metric.update(np.array([1, 1]), np.array([0.8, 0.9]))

        with self.assertRaises(ValueError):
            metric.result()

    def test_reset_clears_state(self):
        metric = StreamingAUC()
        metric.update(np.array([0, 1]), np.array([0.1, 0.9]))
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()


class TestRollingWindowMetrics(unittest.TestCase):
    def test_accuracy_uses_latest_samples_only(self):
        metric = RollingWindowAccuracy(window_size=3)
        metric.update(np.array([1, 0, 1]), np.array([1, 1, 1]))
        metric.update(np.array([0, 1]), np.array([0, 0]))

        self.assertAlmostEqual(metric.result(), 2 / 3)
        np.testing.assert_array_equal(metric.y_true_, np.array([1, 0, 1]))

    def test_precision_recall_f1_use_current_window(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 1])

        precision = RollingWindowPrecision(window_size=4)
        recall = RollingWindowRecall(window_size=4)
        f1 = RollingWindowF1(window_size=4)
        precision.update(y_true, y_pred)
        recall.update(y_true, y_pred)
        f1.update(y_true, y_pred)

        self.assertAlmostEqual(precision.result(), 2 / 3)
        self.assertAlmostEqual(recall.result(), 2 / 3)
        self.assertAlmostEqual(f1.result(), 2 / 3)

    def test_rolling_auc_uses_latest_scores_only(self):
        metric = RollingWindowAUC(window_size=4)
        metric.update(np.array([0, 1]), np.array([0.9, 0.1]))
        metric.update(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))

        self.assertAlmostEqual(metric.result(), 0.75)

    def test_invalid_window_size_raises(self):
        with self.assertRaises(ValueError):
            RollingWindowAccuracy(window_size=0)

    def test_result_before_update_raises(self):
        metric = RollingWindowAccuracy(window_size=3)
        with self.assertRaises(ValueError):
            metric.result()

    def test_reset_clears_window(self):
        metric = RollingWindowAccuracy(window_size=3)
        metric.update(np.array([1, 0]), np.array([1, 0]))
        metric.reset()

        with self.assertRaises(ValueError):
            metric.result()


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


class TestVisualise(unittest.TestCase):
    def test_plot_metric_over_time_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metric.png")
            print(f"Saving to {path}")
            plot_metric_over_time([0.5, 0.7, 0.9], "Accuracy", "Accuracy", save_path=path)
            self.assertTrue(os.path.isfile(path))

    def test_compare_models_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compare.png")
            print(f"Saving to {path}")
            compare_models([0.5, 0.6], [0.4, 0.8], ["Tree", "Bagging"], save_path=path)
            self.assertTrue(os.path.isfile(path))

    def test_plot_predictions_vs_ground_truth_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pred.png")
            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 0, 0])
            plot_predictions_vs_ground_truth(y_true, y_pred, save_path=path)
            self.assertTrue(os.path.isfile(path))

    def test_compare_models_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compare_models([0.5, 0.6], [0.4], ["A", "B"])

    def test_plot_predictions_empty_raises(self):
        with self.assertRaises(ValueError):
            plot_predictions_vs_ground_truth(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()

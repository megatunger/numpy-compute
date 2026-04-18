import unittest
import numpy as np

from numcompute.metrics import accuracy, precision, recall, f1, mse, confusion_matrix

class AccuracyTest(unittest.TestCase):
    def test_perfect_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

    def test_no_match(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(accuracy(y_true, y_pred), 0.0)

    def test_partial_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 0, 3, 0])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.5)

    def test_single_element(self):
        y_true = np.array([1])
        y_pred = np.array([1])
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

    def test_single_element_incorrect(self):
        y_true = np.array([1])
        y_pred = np.array([0])
        self.assertEqual(accuracy(y_true, y_pred), 0.0)

    def test_empty_arrays(self):
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)

    def test_shape_mismatch(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)

    def test_multidimensional_arrays(self):
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 0], [3, 0]])
        # 2 correct out of 4
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.5)

    def test_boolean_arrays(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 1/3)

    def test_float_values_exact_match(self):
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

    def test_float_values_precision_issue(self):
        y_true = np.array([0.1 + 0.2])
        y_pred = np.array([0.3])
        # Likely False due to floating point precision
        self.assertEqual(accuracy(y_true, y_pred), 0.0)

    def test_with_nan_values(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, np.nan, 0.0])
        # np.nan != np.nan, so only 1 match
        self.assertAlmostEqual(accuracy(y_true, y_pred), 1/3)

    def test_with_inf_values(self):
        y_true = np.array([1.0, np.inf, -np.inf])
        y_pred = np.array([1.0, np.inf, np.inf])
        # 2 matches
        self.assertAlmostEqual(accuracy(y_true, y_pred), 2/3)

    def test_string_labels(self):
        y_true = np.array(["cat", "dog", "bird"])
        y_pred = np.array(["cat", "dog", "fish"])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 2/3)

    def test_mixed_types(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.0, 2.0, 0.0])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 2/3)

    def test_large_array(self):
        y_true = np.ones(100000)
        y_pred = np.ones(100000)
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

    def test_broadcasting_not_allowed(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1])
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)


class PrecisionTest(unittest.TestCase):

    def test_perfect_precision(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(precision(y_true, y_pred), 1.0)

    def test_no_true_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(precision(y_true, y_pred), 0.0)

    def test_no_predicted_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(precision(y_true, y_pred), 0.0)

    def test_partial_precision(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0])
        # TP=1, FP=1 → 1/2
        self.assertAlmostEqual(precision(y_true, y_pred), 0.5)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(precision(y_true, y_pred, pos_label=2), 0.5)

    def test_boolean_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(precision(y_true, y_pred, True), 0.5)


class RecallTest(unittest.TestCase):

    def test_perfect_recall(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(recall(y_true, y_pred), 1.0)

    def test_no_true_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(recall(y_true, y_pred), 0.0)

    def test_no_actual_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        self.assertEqual(recall(y_true, y_pred), 0.0)

    def test_partial_recall(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        # TP=1, FN=1 → 1/2
        self.assertAlmostEqual(recall(y_true, y_pred), 0.5)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(recall(y_true, y_pred, pos_label=2), 0.5)

    def test_multidimensional_input(self):
        y_true = np.array([[1, 0], [1, 0]])
        y_pred = np.array([[1, 1], [0, 0]])
        self.assertAlmostEqual(recall(y_true, y_pred), 0.5)


class F1Test(unittest.TestCase):

    def test_perfect_f1(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(f1(y_true, y_pred), 1.0)

    def test_zero_f1(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(f1(y_true, y_pred), 0.0)

    def test_partial_f1(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        # precision = 0.5, recall = 0.5 → f1 = 0.5
        self.assertAlmostEqual(f1(y_true, y_pred), 0.5)

    def test_precision_zero_case(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        self.assertEqual(f1(y_true, y_pred), 0.0)

    def test_recall_zero_case(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(f1(y_true, y_pred), 0.0)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(f1(y_true, y_pred, pos_label=2), 0.5)

    def test_boolean_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(f1(y_true, y_pred, True), 0.5)


class ConfusionMatrixTest(unittest.TestCase):

    def test_binary_confusion_matrix(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        expected = np.array([[1, 1],
                             [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_multiclass_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 2, 1, 1])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        expected = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(cm, expected)

    def test_auto_label_detection(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 1])

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape[0], cm.shape[1])

    def test_shape_mismatch(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        with self.assertRaises(ValueError):
            confusion_matrix(y_true, y_pred)

    def test_empty_input(self):
        y_true = np.array([])
        y_pred = np.array([])

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.size, 0)


class MSETest(unittest.TestCase):

    def test_zero_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertEqual(mse(y_true, y_pred), 0.0)

    def test_basic_case(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        # (1^2 + 0 + 1^2)/3 = 2/3
        self.assertAlmostEqual(mse(y_true, y_pred), 2/3)

    def test_negative_values(self):
        y_true = np.array([-1, -2])
        y_pred = np.array([1, 2])
        self.assertEqual(mse(y_true, y_pred), 8.0)

    def test_float_precision(self):
        y_true = np.array([0.1, 0.2])
        y_pred = np.array([0.1, 0.3])
        self.assertAlmostEqual(mse(y_true, y_pred), 0.005)

    def test_multidimensional(self):
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 3], [2, 4]])
        self.assertAlmostEqual(mse(y_true, y_pred), 0.5)

    def test_shape_mismatch(self):
        y_true = np.array([1, 2])
        y_pred = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            mse(y_true, y_pred)

    def test_empty_input(self):
        y_true = np.array([])
        y_pred = np.array([])

        with self.assertRaises(ValueError):
            mse(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()

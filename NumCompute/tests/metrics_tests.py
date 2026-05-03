import unittest
import numpy as np

from numcompute.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix

class AccuracyTest(unittest.TestCase):
    def test_perfect_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_no_match(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(accuracy_score(y_true, y_pred), 0.0)

    def test_partial_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 0, 3, 0])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.5)

    def test_single_element(self):
        y_true = np.array([1])
        y_pred = np.array([1])
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_single_element_incorrect(self):
        y_true = np.array([1])
        y_pred = np.array([0])
        self.assertEqual(accuracy_score(y_true, y_pred), 0.0)

    def test_empty_arrays(self):
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)

    def test_shape_mismatch(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)

    def test_multidimensional_arrays(self):
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 0], [3, 0]])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.5)

    def test_boolean_arrays(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 1/3)

    def test_float_values_exact_match(self):
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_float_values_precision_issue(self):
        y_true = np.array([0.1 + 0.2])
        y_pred = np.array([0.3])
        self.assertEqual(accuracy_score(y_true, y_pred), 0.0)

    def test_with_nan_values(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, np.nan, 0.0])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 1/3)

    def test_with_inf_values(self):
        y_true = np.array([1.0, np.inf, -np.inf])
        y_pred = np.array([1.0, np.inf, np.inf])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 2/3)

    def test_string_labels(self):
        y_true = np.array(["cat", "dog", "bird"])
        y_pred = np.array(["cat", "dog", "fish"])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 2/3)

    def test_mixed_types(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.0, 2.0, 0.0])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 2/3)

    def test_large_array(self):
        y_true = np.ones(100000)
        y_pred = np.ones(100000)
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_broadcasting_not_allowed(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1])
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)


class PrecisionTest(unittest.TestCase):

    def test_perfect_precision_score(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(precision_score(y_true, y_pred), 1.0)

    def test_no_true_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(precision_score(y_true, y_pred), 0.0)

    def test_no_predicted_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(precision_score(y_true, y_pred), 0.0)

    def test_partial_precision_score(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(precision_score(y_true, y_pred), 0.5)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(precision_score(y_true, y_pred, pos_label=2), 0.5)

    def test_boolean_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(precision_score(y_true, y_pred, True), 0.5)


class RecallTest(unittest.TestCase):

    def test_perfect_recall_score(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(recall_score(y_true, y_pred), 1.0)

    def test_no_true_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(recall_score(y_true, y_pred), 0.0)

    def test_no_actual_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        self.assertEqual(recall_score(y_true, y_pred), 0.0)

    def test_partial_recall_score(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(recall_score(y_true, y_pred), 0.5)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(recall_score(y_true, y_pred, pos_label=2), 0.5)
        
    def test_multidimensional_input(self):
        y_true = np.array([[1, 0], [1, 0]])
        y_pred = np.array([[1, 1], [0, 0]])
        self.assertAlmostEqual(recall_score(y_true, y_pred), 0.5)
    
    def test_boolean_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(precision_score(y_true, y_pred, True), 0.5)


class F1Test(unittest.TestCase):

    def test_perfect_f1_score(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        self.assertEqual(f1_score(y_true, y_pred), 1.0)

    def test_zero_f1_score(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(f1_score(y_true, y_pred), 0.0)

    def test_partial_f1_score(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.5)

    def test_precision_zero_case(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        self.assertEqual(f1_score(y_true, y_pred), 0.0)

    def test_recall_zero_case(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        self.assertEqual(f1_score(y_true, y_pred), 0.0)

    def test_non_binary_label(self):
        y_true = np.array([2, 2, 1])
        y_pred = np.array([2, 1, 2])
        self.assertAlmostEqual(f1_score(y_true, y_pred, pos_label=2), 2/4)

    def test_boolean_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, True, False])
        self.assertAlmostEqual(f1_score(y_true, y_pred, True), 2/4)


class ConfusionMatrixTest(unittest.TestCase):
    # TODO: implement to accept this test
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

        with self.assertRaises(ValueError):
            confusion_matrix(y_true, y_pred)


class MSETest(unittest.TestCase):

    def test_zero_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertEqual(mean_squared_error(y_true, y_pred), 0.0)

    def test_basic_case(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 2/3)

    def test_negative_values(self):
        y_true = np.array([-1, -2])
        y_pred = np.array([1, 2])
        self.assertEqual(mean_squared_error(y_true, y_pred), 10.0)

    def test_float_precision_score(self):
        y_true = np.array([0.1, 0.2])
        y_pred = np.array([0.1, 0.3])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.005)

    def test_multidimensional(self):
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 3], [2, 4]])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.5)

    def test_shape_mismatch(self):
        y_true = np.array([1, 2])
        y_pred = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)

    def test_empty_input(self):
        y_true = np.array([])
        y_pred = np.array([])

        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()

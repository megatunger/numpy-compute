import unittest

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_init_sets_hyperparameters(self):
        tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=4,
            criterion="entropy",
            max_features=2,
        )

        self.assertEqual(tree.max_depth, 3)
        self.assertEqual(tree.min_samples_split, 4)
        self.assertEqual(tree.criterion, "entropy")
        self.assertEqual(tree.max_features, 2)
        self.assertIsNone(tree.tree_)
        self.assertIsNone(tree.classes_)

    def test_partial_fit_returns_self(self):
        tree = DecisionTreeClassifier()
        out = tree.partial_fit(self.X, self.y)
        self.assertIs(out, tree)

    def test_single_class_constant_predictions(self):
        tree = DecisionTreeClassifier()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1, 1, 1])

        tree.partial_fit(X, y)
        y_pred = tree.predict(X)

        np.testing.assert_array_equal(y_pred, [1, 1, 1])

    def test_max_depth_respected(self):
        tree = DecisionTreeClassifier(max_depth=1, min_samples_split=2)
        tree.partial_fit(self.X, self.y)

        def max_depth(node, depth=0):
            if node is None or node.left is None:
                return depth
            return max(max_depth(node.left, depth + 1), max_depth(node.right, depth + 1))

        self.assertLessEqual(max_depth(tree.tree_), tree.max_depth)

    def test_three_partial_fits_grow_tree(self):
        tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

        chunk1 = self.X[:2], self.y[:2]
        chunk2 = self.X[2:4], self.y[2:4]
        chunk3 = self.X[4:], self.y[4:]

        tree.partial_fit(*chunk1)
        tree_after_one = tree.tree_
        tree.partial_fit(*chunk2)
        tree.partial_fit(*chunk3)

        self.assertIsNotNone(tree.tree_)
        self.assertGreaterEqual(tree.tree_.n_samples, len(self.X))

    def test_predict_before_fit_raises(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.predict(self.X)

    def test_partial_fit_empty_raises(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.partial_fit(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from numcompute_stream.ensemble import BaggingClassifier


class TestBaggingClassifier(unittest.TestCase):
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

    def test_init_creates_n_estimators(self):
        bag = BaggingClassifier(n_estimators=5, max_depth=2, random_state=0)

        self.assertEqual(len(bag.estimators_), 5)
        self.assertEqual(bag.estimators_[0].max_depth, 2)
        self.assertIsNone(bag.classes_)

    def test_partial_fit_returns_self(self):
        bag = BaggingClassifier(n_estimators=3, random_state=0)
        out = bag.partial_fit(self.X, self.y)
        self.assertIs(out, bag)

    def test_all_estimators_get_fitted(self):
        bag = BaggingClassifier(n_estimators=4, random_state=0)
        bag.partial_fit(self.X, self.y)

        for tree in bag.estimators_:
            self.assertIsNotNone(tree.tree_)
            self.assertGreaterEqual(tree.tree_.n_samples, 1)

    def test_predict_returns_labels_from_training_classes(self):
        bag = BaggingClassifier(n_estimators=5, random_state=0)
        bag.partial_fit(self.X, self.y)
        y_pred = bag.predict(self.X)

        self.assertEqual(y_pred.shape, (len(self.X),))
        for label in y_pred:
            self.assertIn(label, self.y)

    def test_two_partial_fits_keep_all_trees_updated(self):
        bag = BaggingClassifier(n_estimators=3, random_state=1)
        bag.partial_fit(self.X[:3], self.y[:3])
        counts_after_one = [tree.tree_.n_samples for tree in bag.estimators_]
        bag.partial_fit(self.X[3:], self.y[3:])
        counts_after_two = [tree.tree_.n_samples for tree in bag.estimators_]

        for before, after in zip(counts_after_one, counts_after_two):
            self.assertGreater(after, before)

    def test_majority_vote_breaks_ties_to_smaller_label(self):
        bag = BaggingClassifier(n_estimators=1)
        votes = np.array([[0, 1, 1], [1, 0, 1]])
        y_pred = bag._majority_vote(votes)

        np.testing.assert_array_equal(y_pred, [0, 0, 1])

    def test_predict_before_fit_raises(self):
        bag = BaggingClassifier(n_estimators=3)
        with self.assertRaises(ValueError):
            bag.predict(self.X)

    def test_partial_fit_empty_raises(self):
        bag = BaggingClassifier(n_estimators=3)
        with self.assertRaises(ValueError):
            bag.partial_fit(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()

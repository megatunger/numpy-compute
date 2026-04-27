import unittest
import numpy as np
from numcompute.loops.rank_loop import rank_loop, percentile_loop

class RankTests(unittest.TestCase):
    def test_ordinal_no_ties(self):
        a = np.array([3.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(rank_loop(a, method='ordinal'), np.array([3, 1, 2])))

    def test_ordinal_with_ties(self):
        a = np.array([1.0, 2.0, 1.0])
        self.assertTrue(np.array_equal(rank_loop(a, method='ordinal'), np.array([1, 3, 2])))

    def test_dense_no_ties(self):
        a = np.array([3.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(rank_loop(a, method='dense'), np.array([3, 1, 2])))

    def test_dense_with_ties(self):
        a = np.array([1.0, 2.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(rank_loop(a, method='dense'), np.array([1, 2, 2, 3])))

    def test_average_no_ties(self):
        a = np.array([3.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(rank_loop(a), np.array([3.0, 1.0, 2.0])))

    def test_average_with_ties(self):
        a = np.array([1.0, 2.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(rank_loop(a), np.array([1.0, 2.5, 2.5, 4.0])))

    def test_average_all_tied(self):
        a = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.array_equal(rank_loop(a), np.array([2.0, 2.0, 2.0])))

    def test_invalid_method(self):
        a = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            rank_loop(a, method='definitely real method')

    def test_shape(self):
        a = np.array([3.0, 1.0, 2.0])
        for method in ['ordinal', 'dense', 'average']:
            self.assertEqual(rank_loop(a, method=method).shape, a.shape)

class PercentileTests(unittest.TestCase):
    def test_median(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 50), 3.0)

    def test_minimum(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 0), 1.0)

    def test_maximum(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 100), 5.0)

    def test_linear(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 25, interpolation='linear'), 2.0)

    def test_lower(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 30, interpolation='lower'), 2.0)

    def test_higher(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 30, interpolation='higher'), 3.0)

    def test_midpoint(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(percentile_loop(a, 30, interpolation='midpoint'), 2.5)

    def test_multiple(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile_loop(a, [0, 50, 100])
        np.testing.assert_array_almost_equal(result, np.array([1.0, 3.0, 5.0]))

    def test_invalid_method(self):
        a = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            percentile_loop(a, 50, interpolation='definitely real method')

if __name__ == '__main__':
    unittest.main()

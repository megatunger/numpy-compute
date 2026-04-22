import unittest
import numpy as np

from numcompute.stats import mean, median, std, min, max, quantile, histogram

class TestMean(unittest.TestCase):

    def test_basic(self):
        x = np.array([1, 2, 3])
        self.assertEqual(mean(x), 2.0)

    def test_float(self):
        x = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(mean(x), 0.2)

    def test_negative(self):
        x = np.array([-1, -2, -3])
        self.assertEqual(mean(x), -2.0)

    def test_multidimensional(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(mean(x), 2.5)

    def test_axis(self):
        x = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(mean(x, axis=0), [2, 3])
        np.testing.assert_array_equal(mean(x, axis=1), [1.5, 3.5])

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            mean(x)

    def test_with_nan(self):
        x = np.array([1.0, 2.0, np.nan])
        self.assertAlmostEqual(mean(x, nan_handling=0), 1.0)

    def test_large_values(self):
        x = np.array([1e10, 1e10])
        self.assertEqual(mean(x), 1e10)


class TestMedian(unittest.TestCase):

    def test_odd_length(self):
        x = np.array([3, 1, 2])
        self.assertEqual(median(x), 2)

    def test_even_length(self):
        x = np.array([1, 2, 3, 4])
        self.assertEqual(median(x), 2.5)

    def test_unsorted(self):
        x = np.array([10, 1, 5])
        self.assertEqual(median(x), 5)

    def test_multidimensional(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(median(x), 2.5)

    def test_axis(self):
        x = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(median(x, axis=0), [2, 3])

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            mean(x)

    def test_with_nan(self):
        x = np.array([1, np.nan])
        self.assertAlmostEqual(median(x, nan_handling=0), 0.5)


class TestStd(unittest.TestCase):

    def test_basic(self):
        x = np.array([1, 2, 3])
        self.assertAlmostEqual(std(x), np.sqrt(2/3))

    def test_zero_variance(self):
        x = np.array([5, 5, 5])
        self.assertEqual(std(x), 0.0)

    def test_negative_values(self):
        x = np.array([-1, -2, -3])
        self.assertAlmostEqual(std(x), np.sqrt(2/3))

    def test_ddof(self):
        x = np.array([1, 2, 3])
        self.assertAlmostEqual(std(x, ddof=1), 1.0)

    def test_multidimensional(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertAlmostEqual(std(x), std(x.flatten()))

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            mean(x)

    def test_with_nan(self):
        x = np.array([1.0, np.nan])
        self.assertAlmostEqual(std(x, nan_handling=0.0), 0.5)

class TestMinMax(unittest.TestCase):

    def test_basic(self):
        x = np.array([1, 2, 3])
        self.assertEqual(min(x), 1)
        self.assertEqual(max(x), 3)

    def test_negative(self):
        x = np.array([-5, -1, -3])
        self.assertEqual(min(x), -5)
        self.assertEqual(max(x), -1)

    def test_multidimensional(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(min(x), 1)
        self.assertEqual(max(x), 4)

    def test_axis(self):
        x = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(min(x, axis=0), [1, 2])
        np.testing.assert_array_equal(max(x, axis=1), [2, 4])

    def test_with_nan(self):
        x = np.array([1.0, np.nan])
        self.assertEqual(min(x, nan_handling=0), 0.0)
        self.assertEqual(max(x, nan_handling=0), 1.0)

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            min(x)
        with self.assertRaises(ValueError):
            max(x)


class TestHistogram(unittest.TestCase):

    def test_basic(self):
        x = np.array([1, 2, 3, 4])
        hist, bins = histogram(x, bins=2)
        self.assertEqual(hist.sum(), len(x))
        self.assertEqual(len(hist), 2)

    def test_custom_bins(self):
        x = np.array([0, 1, 2, 3])
        hist, bins = histogram(x, bins=[0, 2, 4])
        np.testing.assert_array_equal(hist, [2, 2])

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            histogram(x)

    def test_with_negative(self):
        x = np.array([-2, -1, 0, 1])
        hist, _ = histogram(x, bins=2)
        self.assertEqual(hist.sum(), len(x))

    def test_density(self):
        x = np.array([1, 2, 3, 4])
        hist, bins = histogram(x, bins=2, density=True)
        # integral approx = 1
        self.assertAlmostEqual(np.sum(hist * np.diff(bins)), 1.0, places=5)

class TestQuantile(unittest.TestCase):

    def test_basic(self):
        x = np.array([1, 2, 3, 4])
        self.assertEqual(quantile(x, 0.5), 2.5)

    def test_multiple_quantiles(self):
        x = np.array([1, 2, 3, 4])
        q = quantile(x, [0.25, 0.5, 0.75])
        np.testing.assert_array_equal(q, [1.75, 2.5, 3.25])

    def test_unsorted(self):
        x = np.array([4, 1, 3, 2])
        self.assertEqual(quantile(x, 0.5), 2.5)

    def test_multidimensional(self):
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(quantile(x, 0.5), 2.5)

    def test_axis(self):
        x = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(quantile(x, 0.5, axis=0), [2., 3.])

    def test_edge_quantiles(self):
        x = np.array([1, 2, 3])
        self.assertEqual(quantile(x, 0.0), 1)
        self.assertEqual(quantile(x, 1.0), 3)

    def test_invalid_quantile(self):
        x = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            quantile(x, -0.1)

    def test_empty(self):
        x = np.array([])
        with self.assertRaises(ValueError):
            quantile(x, 0.5)

    def test_with_nan(self):
        x = np.array([1.0, np.nan])
        self.assertEqual(quantile(x, 0.5, nan_handling=0), 0.5)


if __name__ == "__main__":
    unittest.main()
    # x = np.array([[1, 2], [3, 4]])
    # print(quantile(x, 0.5, axis=0))
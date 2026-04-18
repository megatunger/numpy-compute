import unittest
import numpy as np
from numcompute.utils import euclidean_distance, manhattan_distance, chebyshev_distance

class EuclideanDistanceTests(unittest.TestCase):

    def test_euclidean_distance_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 5.196)

    def test_euclidean_distance_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 12.450)

    def test_euclidean_distance_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        # using np.isclose due to floating point error
        self.assertEqual(round(euclidean_distance(a, b)), 0)

    def test_euclidean_distance_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 5.716)

    def test_euclidean_distance_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(euclidean_distance(a, b), 3), 10.0)

class ManhattanDistanceTests(unittest.TestCase):

    def test_manhattan_distance_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 9.0)

    def test_manhattan_distance_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 21)

    def test_manhattan_distance_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        # using np.isclose due to floating point error
        self.assertEqual(round(manhattan_distance(a, b)), 0)

    def test_manhattan_distance_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 9.9)

    def test_manhattan_distance_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(manhattan_distance(a, b), 3), 100.0)

class ChebyshevDistanceTests(unittest.TestCase):

    def test_chebyshev_distance_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 3.0)

    def test_chebyshev_distance_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 9.0)

    def test_chebyshev_distance_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        # using np.isclose due to floating point error
        self.assertEqual(round(chebyshev_distance(a, b)), 0)

    def test_chebyshev_distance_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 3.3)

    def test_chebyshev_distance_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(chebyshev_distance(a, b), 3), 1.0)

if __name__ == '__main__':
    unittest.main()

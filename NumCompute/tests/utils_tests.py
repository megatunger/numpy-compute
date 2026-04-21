import unittest
import numpy as np
from numcompute.utils import (
    euclidean_distance, 
    manhattan_distance, 
    chebyshev_distance, 
    cosine_similarity,
    logsumexp,
    tanh,
    relu,
    leaky_relu,
    sigmoid,
    softmax
)

class EuclideanDistanceTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 5.196)
        # self.assertAlmostEqual(euclidean_distance(a, b), 5.19615)

    def test_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 12.45)

    def test_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertEqual(round(euclidean_distance(a, b)), 0)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(euclidean_distance(a, b), 3), 5.716)

    def test_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(euclidean_distance(a, b), 3), 10.0)

class ManhattanDistanceTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 9.0)

    def test_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 21)

    def test_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertEqual(round(manhattan_distance(a, b)), 0)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(manhattan_distance(a, b), 3), 9.9)

    def test_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(manhattan_distance(a, b), 3), 100.0)

class ChebyshevDistanceTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 3.0)

    def test_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 9.0)

    def test_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertEqual(round(chebyshev_distance(a, b)), 0)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([4.4, 5.5, 6.6])
        self.assertEqual(round(chebyshev_distance(a, b), 3), 3.3)

    def test_dims(self):
        a = np.zeros(100)
        b = np.ones(100)
        self.assertEqual(round(chebyshev_distance(a, b), 3), 1.0)

class CosineSimilarityTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(round(cosine_similarity(a, b), 3), 0.975)

    def test_negative(self):
        a = np.array([1, 2, 3])
        b = np.array([-4, -5, -6])
        self.assertEqual(round(cosine_similarity(a, b), 3), -0.975)

    def test_same(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertEqual(round(cosine_similarity(a, b)), 1)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        b = np.array([7.7, 8.8, 9.9])
        self.assertEqual(round(cosine_similarity(a, b), 3), 0.959)

    def test_dims(self):
        a = np.full(100, 2)
        b = np.ones(100)
        self.assertEqual(round(chebyshev_distance(a, b), 3), 1.0)

    def test_zeros(self):
        a = np.zeros(3)
        b = np.ones(3)
        with self.assertRaises(ValueError):
            cosine_similarity(a, b)

class LogsumexpTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        self.assertEqual(round(logsumexp(a), 3), 3.408)

    def test_one_value(self):
        a = np.array([1])
        self.assertEqual(round(logsumexp(a), 3), 1.0)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        self.assertEqual(round(logsumexp(a), 3), 3.667)

    def test_zeros(self):
        a = np.zeros(3)
        self.assertEqual(round(logsumexp(a), 3), 1.099)

    def test_overflow_large(self):
        a = np.array([100000, 100002, 100003])
        self.assertEqual(round(logsumexp(a), 3), 100003.349)

    def test_overflow_small(self):
        a = np.array([-100000, -100002, -100003])
        self.assertEqual(round(logsumexp(a), 3), -99999.830)

    def test_dims(self):
        a = np.full(100, 2)
        self.assertEqual(round(logsumexp(a), 3), 6.605)

class TanhTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(np.round(tanh(a), 3), np.array([0.762, 0.964, 0.995])))

    def test_one_value(self):
        a = np.array([1])
        self.assertTrue(np.array_equal(np.round(tanh(a), 3), np.array([0.762])))

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3])
        self.assertTrue(np.array_equal(np.round(tanh(a), 3), np.array([0.800, 0.976, 0.997])))

    def test_zero(self):
        a = np.zeros(1)
        self.assertTrue(np.array_equal(np.round(tanh(a), 3), np.array([0.0])))

    def test_symmetry(self):
        a = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(np.round(tanh(-a), 3), np.round(-tanh(a), 3)))

    def test_range(self):
        a = np.random.rand(3)
        result = tanh(a)
        self.assertTrue(np.all(result > -1) and np.all(result < 1))
    
    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(tanh(a).shape, a.shape)

class ReluTests(unittest.TestCase):

    def test_zeros(self):
        a = np.zeros(3)
        self.assertTrue(np.array_equal(relu(a), a))

    def test_positive(self):
        a = np.array([1.1, 2.2, 3.3])
        self.assertTrue(np.array_equal(relu(a), a))

    def test_negative(self):
        a = np.array([-1.1, -2.2, -3.3])
        self.assertTrue(np.array_equal(relu(a), np.zeros(3)))

    def test_mixed(self):
        a = np.array([-2.3, 5.4, 6.6, 0.0, -9.7])
        self.assertTrue(np.array_equal(relu(a), np.array([0.0, 5.4, 6.6, 0.0, 0.0])))

    def test_range(self):
        a = np.random.rand(3)
        self.assertTrue(np.all(relu(a) > 0))
    
    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(relu(a).shape, a.shape)

class LeakyReluTests(unittest.TestCase):

    def test_zeros(self):
        a = np.zeros(3)
        self.assertTrue(np.array_equal(leaky_relu(a), a))

    def test_positive(self):
        a = np.array([1.1, 2.2, 3.3])
        self.assertTrue(np.array_equal(leaky_relu(a), a))

    def test_negative(self):
        a = np.array([-1.1, -2.2, -3.3])
        # using allclose because floating point precision
        self.assertTrue(np.allclose(leaky_relu(a), np.array([-0.011, -0.022, -0.033])))

    def test_mixed(self):
        a = np.array([-2.3, 5.4, 6.6, 0.0, -9.7])
        # using allclose because floating point precision
        self.assertTrue(np.allclose(leaky_relu(a), np.array([-0.023, 5.4, 6.6, 0.0, -0.097])))
    
    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(relu(a).shape, a.shape)

class SigmoidTests(unittest.TestCase):
    
    def test_zero(self):
        a = np.zeros(1)
        self.assertTrue(np.array_equal(sigmoid(a), np.array([0.5])))

    def test_symmetry(self):
        a = np.random.rand(3)
        self.assertTrue(np.allclose(sigmoid(-a), 1 - sigmoid(a)))

    def test_range(self):
        a = np.random.rand(3)
        result = sigmoid(a)
        self.assertTrue(np.all(result > 0) and np.all(result < 1))

    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(sigmoid(a).shape, a.shape)

class SoftmaxTests(unittest.TestCase):

    def test_range(self):
        a = np.random.rand(3)
        result = softmax(a)
        self.assertTrue(np.all(result > 0) and np.all(result < 1))

    def test_sum(self):
        a = np.random.rand(3)
        result = softmax(a)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_shift(self):
        a = np.random.rand(3)
        self.assertTrue(np.allclose(softmax(a), softmax(a + 10)))

    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(sigmoid(a).shape, a.shape)

if __name__ == '__main__':
    unittest.main()

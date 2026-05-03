import unittest

import numpy as np

from numcompute.optim import grad, jacobian


class GradTests(unittest.TestCase):
    def test_central_difference(self):
        def f(X):
            return X[:, 0] ** 2 + X[:, 1] ** 3

        result = grad(f, np.array([3.0, 4.0]), method="central")
        expected = np.array([6.0, 48.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_difference(self):
        def f(X):
            return X[:, 0] ** 2 + X[:, 1] ** 3

        result = grad(f, np.array([3.0, 4.0]), h=1e-6, method="forward")
        expected = np.array([6.0, 48.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_invalid_method(self):
        def f(X):
            return X[:, 0]

        with self.assertRaises(ValueError):
            grad(f, np.array([1.0, 2.0]), method="invalid")

    def test_invalid_input_dimension(self):
        def f(X):
            return X[:, 0]

        with self.assertRaises(ValueError):
            grad(f, np.array([[1.0, 2.0]]))

    def test_invalid_function_output_shape(self):
        def f(X):
            return np.ones((X.shape[0], 2))

        with self.assertRaises(ValueError):
            grad(f, np.array([1.0, 2.0]))


class JacobianTests(unittest.TestCase):
    def test_central_difference(self):
        def F(X):
            return np.column_stack((X[:, 0] ** 2 + X[:, 1], X[:, 0] * X[:, 1]))

        result = jacobian(F, np.array([3.0, 4.0]), method="central")
        expected = np.array([[6.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_forward_difference(self):
        def F(X):
            return np.column_stack((X[:, 0] ** 2 + X[:, 1], X[:, 0] * X[:, 1]))

        result = jacobian(F, np.array([3.0, 4.0]), h=1e-6, method="forward")
        expected = np.array([[6.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_single_output_shape(self):
        def F(X):
            return (X[:, 0] ** 2 + X[:, 1])[:, np.newaxis]

        result = jacobian(F, np.array([3.0, 4.0]))
        self.assertEqual(result.shape, (1, 2))

    def test_invalid_function_output_shape(self):
        def F(X):
            return X[:, 0]

        with self.assertRaises(ValueError):
            jacobian(F, np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()

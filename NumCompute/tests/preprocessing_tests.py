import unittest

import numpy as np

from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.X_num = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_fit_requires_2d(self):
        scaler = StandardScaler()
        with self.assertRaisesRegex(ValueError, "StandardScaler expects 2D input."):
            scaler.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_returns_self_and_sets_statistics(self):
        scaler = StandardScaler()
        out = scaler.fit(self.X_num)
        self.assertIs(out, scaler)
        np.testing.assert_allclose(scaler.mean_, np.array([2.0, 3.0]))
        np.testing.assert_allclose(scaler.scale_, np.array([1.0, 1.0]))

    def test_transform_requires_fit(self):
        scaler = StandardScaler()
        with self.assertRaisesRegex(
            ValueError, "StandardScaler is not fitted. Call fit\\(\\) first."
        ):
            scaler.transform(self.X_num)

    def test_transform_returns_standardized_values(self):
        scaler = StandardScaler().fit(self.X_num)
        transformed = scaler.transform(self.X_num)
        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(transformed, expected)

    def test_fit_transform_matches_fit_then_transform(self):
        scaler = StandardScaler()
        fit_then_transform = scaler.fit(self.X_num).transform(self.X_num)

        scaler_2 = StandardScaler()
        fit_transform = scaler_2.fit_transform(self.X_num)

        np.testing.assert_allclose(fit_transform, fit_then_transform)


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.X_num = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_fit_requires_2d(self):
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.fit(np.array([1.0, 2.0, 3.0]))

    def test_transform_requires_fit(self):
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.transform(self.X_num)

    def test_fit_computes_min_and_max(self):
        scaler = MinMaxScaler()
        scaler.fit(self.X_num)
        np.testing.assert_array_equal(scaler.data_min_, [1.0, 2.0])
        np.testing.assert_array_equal(scaler.data_max_, [3.0, 4.0])

    def test_transform_scales_to_zero_one(self):
        scaler = MinMaxScaler()
        scaler.fit(self.X_num)
        transformed = scaler.transform(self.X_num)
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(transformed, expected)

    def test_fit_transform_matches_fit_then_transform(self):
        scaler = MinMaxScaler()
        fit_then_transform = scaler.fit(self.X_num).transform(self.X_num)
        fit_transform = MinMaxScaler().fit_transform(self.X_num)
        np.testing.assert_allclose(fit_transform, fit_then_transform)


# class TestOneHotEncoder(unittest.TestCase):
#     def setUp(self):
#         self.X_cat = np.array([["red", "S"], ["blue", "M"]], dtype=object)

#     def test_fit_requires_2d(self):
#         enc = OneHotEncoder()
#         with self.assertRaises(ValueError):
#             enc.fit(np.array(["red", "blue"], dtype=object))

#     def test_fit_not_implemented_yet(self):
#         enc = OneHotEncoder()
#         with self.assertRaises(NotImplementedError):
#             enc.fit(self.X_cat)

#     def test_transform_requires_fit(self):
#         enc = OneHotEncoder()
#         with self.assertRaises(ValueError):
#             enc.transform(self.X_cat)


# class TestSimpleImputer(unittest.TestCase):
#     def setUp(self):
#         self.X_num = np.array([[1.0, 2.0], [3.0, 4.0]])

#     def test_fit_returns_self(self):
#         imp = SimpleImputer(fill_value=0.0)
#         out = imp.fit(self.X_num)
#         self.assertIs(out, imp)

#     def test_fit_requires_2d(self):
#         imp = SimpleImputer()
#         with self.assertRaises(ValueError):
#             imp.fit(np.array([1.0, np.nan, 3.0]))

#     def test_transform_not_implemented_yet(self):
#         imp = SimpleImputer(fill_value=0.0).fit(self.X_num)
#         with self.assertRaises(NotImplementedError):
#             imp.transform(self.X_num)


# class TestPreprocessingConstructorValidation(unittest.TestCase):
#     def test_invalid_constructor_arguments(self):
#         with self.assertRaises(ValueError):
#             MinMaxScaler(feature_range=(1.0, 1.0))
#         with self.assertRaises(ValueError):
#             OneHotEncoder(handle_unknown="unknown")


if __name__ == "__main__":
    unittest.main()

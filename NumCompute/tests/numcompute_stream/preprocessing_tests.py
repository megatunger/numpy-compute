import unittest

import numpy as np

from numcompute.preprocessing import OneHotEncoder as BatchOneHotEncoder
from numcompute.preprocessing import SimpleImputer as BatchSimpleImputer
from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler


class TestStreamingStandardScaler(unittest.TestCase):
    def setUp(self):
        self.chunk1 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        self.chunk2 = np.array([[4.0, 40.0], [5.0, 50.0]])

    def test_two_partial_fits_match_batch_fit(self):
        scaler = StandardScaler()
        scaler.partial_fit(self.chunk1)
        scaler.partial_fit(self.chunk2)

        batch = BatchStandardScaler().fit(np.vstack([self.chunk1, self.chunk2]))

        np.testing.assert_allclose(scaler.mean_, batch.mean_)
        np.testing.assert_allclose(scaler.scale_, batch.scale_)

    def test_partial_fit_then_transform_matches_batch(self):
        scaler = StandardScaler()
        scaler.partial_fit(self.chunk1)
        scaler.partial_fit(self.chunk2)

        X = np.vstack([self.chunk1, self.chunk2])
        batch = BatchStandardScaler().fit(X)

        np.testing.assert_allclose(scaler.transform(X), batch.transform(X))

    def test_partial_fit_empty_raises(self):
        scaler = StandardScaler()
        with self.assertRaises(ValueError):
            scaler.partial_fit(np.array([]))

    def test_partial_fit_requires_2d(self):
        scaler = StandardScaler()
        with self.assertRaisesRegex(ValueError, "StandardScaler expects 2D input."):
            scaler.partial_fit(np.array([1.0, 2.0, 3.0]))

    def test_partial_fit_returns_self(self):
        scaler = StandardScaler()
        out = scaler.partial_fit(self.chunk1)
        self.assertIs(out, scaler)


class TestStreamingSimpleImputer(unittest.TestCase):
    def setUp(self):
        self.chunk1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        self.chunk2 = np.array([[5.0, 6.0], [np.nan, 8.0]])

    def test_two_partial_fits_match_batch_fit(self):
        imputer = SimpleImputer()
        imputer.partial_fit(self.chunk1)
        imputer.partial_fit(self.chunk2)

        batch = BatchSimpleImputer().fit(np.vstack([self.chunk1, self.chunk2]))

        np.testing.assert_allclose(imputer.statistics_, batch.statistics_)

    def test_partial_fit_then_transform_matches_batch(self):
        imputer = SimpleImputer()
        imputer.partial_fit(self.chunk1)
        imputer.partial_fit(self.chunk2)

        X = np.vstack([self.chunk1, self.chunk2])
        batch = BatchSimpleImputer().fit(X)

        np.testing.assert_allclose(imputer.transform(X), batch.transform(X))

    def test_partial_fit_empty_raises(self):
        imputer = SimpleImputer()
        with self.assertRaises(ValueError):
            imputer.partial_fit(np.array([]))

    def test_partial_fit_requires_2d(self):
        imputer = SimpleImputer()
        with self.assertRaisesRegex(ValueError, "SimpleImputer expects 2D input."):
            imputer.partial_fit(np.array([1.0, np.nan, 3.0]))

    def test_partial_fit_returns_self(self):
        imputer = SimpleImputer()
        out = imputer.partial_fit(self.chunk1)
        self.assertIs(out, imputer)


class TestStreamingOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.chunk1 = np.array([["red", "S"], ["blue", "M"]], dtype=object)
        self.chunk2 = np.array([["green", "L"]], dtype=object)

    def test_two_partial_fits_match_batch_fit(self):
        encoder = OneHotEncoder()
        encoder.partial_fit(self.chunk1)
        encoder.partial_fit(self.chunk2)

        batch = BatchOneHotEncoder().fit(
            np.vstack([self.chunk1, self.chunk2])
        )

        self.assertEqual(len(encoder.categories_), len(batch.categories_))
        for got, expected in zip(encoder.categories_, batch.categories_):
            np.testing.assert_array_equal(got, expected)

    def test_partial_fit_then_transform_matches_batch(self):
        encoder = OneHotEncoder()
        encoder.partial_fit(self.chunk1)
        encoder.partial_fit(self.chunk2)

        X = np.vstack([self.chunk1, self.chunk2])
        batch = BatchOneHotEncoder().fit(X)

        np.testing.assert_array_equal(encoder.transform(X), batch.transform(X))

    def test_partial_fit_grows_categories_for_new_values(self):
        encoder = OneHotEncoder()
        encoder.partial_fit(self.chunk1)
        encoder.partial_fit(self.chunk2)

        self.assertIn("green", encoder.categories_[0])
        self.assertIn("L", encoder.categories_[1])

    def test_handle_unknown_ignore_after_partial_fit(self):
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.partial_fit(self.chunk1)

        unknown = np.array([["purple", "S"]], dtype=object)
        transformed = encoder.transform(unknown)

        self.assertEqual(transformed.shape[1], sum(len(c) for c in encoder.categories_))
        np.testing.assert_array_equal(transformed, np.zeros((1, transformed.shape[1])))

    def test_partial_fit_empty_raises(self):
        encoder = OneHotEncoder()
        with self.assertRaises(ValueError):
            encoder.partial_fit(np.array([]))

    def test_partial_fit_returns_self(self):
        encoder = OneHotEncoder()
        out = encoder.partial_fit(self.chunk1)
        self.assertIs(out, encoder)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute_stream.preprocessing import StandardScaler


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


if __name__ == "__main__":
    unittest.main()

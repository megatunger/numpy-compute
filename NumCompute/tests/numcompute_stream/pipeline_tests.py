import unittest

import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler


class DummyClassifier:
    """Tiny stand-in model until DecisionTreeClassifier exists"""

    def __init__(self):
        self.classes_ = None
        self.majority_ = None
        self.n_seen_ = 0

    def partial_fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_ = np.bincount(y.astype(int)).argmax()
        self.n_seen_ += len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def fit(self, X, y=None):
        return self.partial_fit(X, y)


class TestStreamingPipeline(unittest.TestCase):
    def setUp(self):
        self.chunk1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y1 = np.array([0, 0])
        self.chunk2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        self.y2 = np.array([1, 1])

    def test_partial_fit_two_chunks_runs(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyClassifier()),
        ])

        pipe.partial_fit(self.chunk1, self.y1)
        pipe.partial_fit(self.chunk2, self.y2)

        self.assertIsNotNone(pipe.named_steps["scale"].mean_)
        self.assertEqual(pipe.named_steps["model"].n_seen_, 4)

    def test_partial_fit_updates_model_predictions(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyClassifier()),
        ])

        pipe.partial_fit(self.chunk1, self.y1)
        y_pred_after_one = pipe.predict(self.chunk2)

        pipe.partial_fit(self.chunk2, self.y2)
        y_pred_after_two = pipe.predict(self.chunk2)

        np.testing.assert_array_equal(y_pred_after_one, [0, 0])
        np.testing.assert_array_equal(y_pred_after_two, [1, 1])

    def test_partial_fit_returns_self(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyClassifier()),
        ])

        out = pipe.partial_fit(self.chunk1, self.y1)
        self.assertIs(out, pipe)

    def test_scaler_stats_match_after_two_partial_fits(self):
        pipe = Pipeline([("scale", StandardScaler())])

        pipe.partial_fit(self.chunk1, self.y1)
        pipe.partial_fit(self.chunk2, self.y2)

        batch_scaler = StandardScaler()
        batch_scaler.partial_fit(self.chunk1)
        batch_scaler.partial_fit(self.chunk2)

        np.testing.assert_allclose(
            pipe.named_steps["scale"].mean_, batch_scaler.mean_
        )
        np.testing.assert_allclose(
            pipe.named_steps["scale"].scale_, batch_scaler.scale_
        )

    def test_missing_partial_fit_raises(self):
        class BadStep:
            def transform(self, X):
                return X

        pipe = Pipeline([("bad", BadStep()), ("model", DummyClassifier())])

        with self.assertRaisesRegex(TypeError, "partial_fit"):
            pipe.partial_fit(self.chunk1, self.y1)


if __name__ == "__main__":
    unittest.main()

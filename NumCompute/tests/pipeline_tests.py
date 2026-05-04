import unittest

import numpy as np

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler


class DummyTransformer:
    def fit(self, X, y=None):
        self.was_fitted = True
        return self

    def transform(self, X):
        return X + 1


class DummyEstimator:
    def fit(self, X, y=None):
        self.was_fitted = True
        self.mean_target = np.mean(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_target)


class CountingTransformer:
    def __init__(self):
        self.transform_calls = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.transform_calls += 1
        return X + 1


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y = np.array([10.0, 20.0])

    def test_pipeline_requires_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_named_steps_are_stored(self):
        transformer = DummyTransformer()
        pipe = Pipeline([("add_one", transformer)])

        self.assertIs(pipe.named_steps["add_one"], transformer)

    def test_duplicate_step_names_raise_error(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("same", DummyTransformer()),
                ("same", DummyTransformer()),
            ])

    def test_fit_transform_chains_preprocessors(self):
        pipe = Pipeline([
            ("add_one_1", DummyTransformer()),
            ("add_one_2", DummyTransformer()),
        ])

        transformed = pipe.fit_transform(self.X)
        expected = self.X + 2

        np.testing.assert_array_equal(transformed, expected)

    def test_fit_transform_transforms_each_step_once(self):
        first = CountingTransformer()
        second = CountingTransformer()
        pipe = Pipeline([
            ("first", first),
            ("second", second),
        ])

        pipe.fit_transform(self.X)

        self.assertEqual(first.transform_calls, 1)
        self.assertEqual(second.transform_calls, 1)

    def test_pipeline_works_with_existing_preprocessors(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 10.0],
        ])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("encode", OneHotEncoder()),
        ])

        transformed = pipe.fit_transform(X)

        self.assertEqual(transformed.shape, (3, 5))

    def test_pipeline_integrates_imputer_and_scaler(self):
        X = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
        ])

        transformed = pipe.fit_transform(X)
        expected = np.array([
            [-1.22474487, 0.0],
            [0.0, -1.22474487],
            [1.22474487, 1.22474487],
        ])

        np.testing.assert_allclose(transformed, expected)

    def test_pipeline_integrates_scaler_and_encoder(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 10.0],
        ])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("encode", OneHotEncoder()),
        ])

        transformed = pipe.fit_transform(X)
        expected = np.array([
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ])

        np.testing.assert_array_equal(transformed, expected)

    def test_fit_returns_self(self):
        pipe = Pipeline([
            ("add_one", DummyTransformer()),
            ("model", DummyEstimator()),
        ])

        result = pipe.fit(self.X, self.y)

        self.assertIs(result, pipe)

    def test_predict_uses_final_estimator(self):
        pipe = Pipeline([
            ("add_one", DummyTransformer()),
            ("model", DummyEstimator()),
        ])

        pipe.fit(self.X, self.y)
        predictions = pipe.predict(self.X)
        expected = np.array([15.0, 15.0])

        np.testing.assert_array_equal(predictions, expected)

    def test_missing_required_method_raises_error(self):
        pipe = Pipeline([("bad_step", object())])

        with self.assertRaises(TypeError):
            pipe.transform(self.X)


if __name__ == "__main__":
    unittest.main()

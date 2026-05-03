import math

import numpy as np

from numcompute.utils import (
    validate_array_like,
    validate_non_empty_array,
    validate_options,
)


class StandardScalerLoop:
    def __init__(self):
        """Loop-based StandardScaler for benchmarking against NumPy."""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("StandardScaler expects 2D input.")

        rows, cols = X.shape
        means = []
        scales = []

        for col in range(cols):
            total = 0.0
            for row in range(rows):
                total += X[row, col]
            mean = total / rows
            means.append(mean)

            squared_diff_total = 0.0
            for row in range(rows):
                diff = X[row, col] - mean
                squared_diff_total += diff * diff

            scale = math.sqrt(squared_diff_total / rows)
            scales.append(1.0 if scale == 0.0 else scale)

        self.mean_ = np.array(means, dtype=float)
        self.scale_ = np.array(scales, dtype=float)
        return self

    def transform(self, X):
        X = validate_array_like(X, name="X").astype(float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")

        rows, cols = X.shape
        X_scaled = np.empty_like(X, dtype=float)

        for row in range(rows):
            for col in range(cols):
                X_scaled[row, col] = (X[row, col] - self.mean_[col]) / self.scale_[col]

        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MinMaxScalerLoop:
    def __init__(self, feature_range=(0.0, 1.0)):
        """Loop-based MinMaxScaler for benchmarking against NumPy."""
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 2D input.")

        rows, cols = X.shape
        data_min = []
        data_max = []
        data_range = []

        for col in range(cols):
            col_min = X[0, col]
            col_max = X[0, col]

            for row in range(1, rows):
                value = X[row, col]
                if value < col_min:
                    col_min = value
                if value > col_max:
                    col_max = value

            data_min.append(col_min)
            data_max.append(col_max)
            data_range.append(col_max - col_min)

        self.data_min_ = np.array(data_min, dtype=float)
        self.data_max_ = np.array(data_max, dtype=float)
        self.data_range_ = np.array(data_range, dtype=float)
        return self

    def transform(self, X):
        X = validate_array_like(X, name="X").astype(float)
        if self.data_min_ is None or self.data_range_ is None:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")

        rows, cols = X.shape
        X_scaled = np.empty_like(X, dtype=float)

        for row in range(rows):
            for col in range(cols):
                X_scaled[row, col] = (X[row, col] - self.data_min_[col]) / self.data_range_[col]

        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OneHotEncoderLoop:
    def __init__(self, handle_unknown="error"):
        """Loop-based OneHotEncoder for benchmarking against NumPy."""
        validate_options(
            handle_unknown, ("error", "ignore"), x_name="handle_unknown"
        )
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = validate_array_like(X, name="X")
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")

        rows, cols = X.shape
        categories = []

        for col in range(cols):
            values = []
            for row in range(rows):
                value = X[row, col]
                if value not in values:
                    values.append(value)
            categories.append(np.array(sorted(values)))

        self.categories_ = categories
        self.n_features_in_ = cols
        return self

    def transform(self, X):
        X = validate_array_like(X, name="X")
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted. Call fit() first.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input has different number of features than fit data.")

        rows = X.shape[0]
        total_categories = sum(len(categories) for categories in self.categories_)
        encoded = np.zeros((rows, total_categories), dtype=float)

        output_col_start = 0
        for col, categories in enumerate(self.categories_):
            for row in range(rows):
                value = X[row, col]
                found = False

                for offset, category in enumerate(categories):
                    if value == category:
                        encoded[row, output_col_start + offset] = 1.0
                        found = True
                        break

                if not found and self.handle_unknown == "error":
                    raise ValueError(f"Unknown category in column {col}: {[value]}")

            output_col_start += len(categories)

        return encoded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class SimpleImputerLoop:
    def __init__(self, strategy="mean"):
        """Loop-based SimpleImputer for benchmarking against NumPy."""
        validate_options(strategy, ("mean",), x_name="strategy")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")

        rows, cols = X.shape
        statistics = []

        for col in range(cols):
            total = 0.0
            count = 0

            for row in range(rows):
                value = X[row, col]
                if not math.isnan(value):
                    total += value
                    count += 1

            statistics.append(total / count if count else np.nan)

        self.statistics_ = np.array(statistics, dtype=float)
        return self

    def transform(self, X):
        X = validate_array_like(X, name="X").astype(float)
        if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")

        rows, cols = X.shape
        X_filled = np.empty_like(X, dtype=float)

        for row in range(rows):
            for col in range(cols):
                value = X[row, col]
                X_filled[row, col] = self.statistics_[col] if math.isnan(value) else value

        return X_filled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

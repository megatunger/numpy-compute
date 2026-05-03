import numpy as np

from numcompute.utils import (
    validate_array_like,
    validate_non_empty_array,
    validate_options,
)


class StandardScaler:
    def __init__(self):
        """Create a scaler that standardizes each feature."""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute mean and standard deviation from input data.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            self.

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("StandardScaler expects 2D input.")
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # If STD is 0, set scale to 1 to avoid divide 0
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """Scale input data using stats learned in fit.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).

        Returns:
            Scaled array with shape (n_samples, n_features).

        Raises:
            ValueError: If the scaler is not fitted.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
        
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled

    def fit_transform(self, X, y=None):
        """Fit the scaler and return scaled data.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            Scaled array with shape (n_samples, n_features).

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        """Create a scaler that rescales features to a range.

        Parameters:
            feature_range: Tuple like (min, max) for output scale.

        Raises:
            ValueError: If feature_range min is greater than or equal to max.
        """
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        """Compute per-feature min and max from input data.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            self.

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 2D input.")
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, X):
        """Rescale input data using min and range from fit.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).

        Returns:
            Rescaled array with shape (n_samples, n_features).

        Raises:
            ValueError: If the scaler is not fitted.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        if self.data_min_ is None or self.data_range_ is None:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
        X_scaled = (X - self.data_min_) / self.data_range_
        return X_scaled

    def fit_transform(self, X, y=None):
        """Fit the scaler and return scaled data.

        Parameters:
            X: 2D input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            Rescaled array with shape (n_samples, n_features).

        Raises:
            ValueError: If X is empty or not 2D, or if the scaler is not fitted.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    def __init__(self, handle_unknown="error"):
        """Create an encoder for categorical columns.

        Parameters:
            handle_unknown: "error" raises on unknown categories. "ignore"
                encodes unknown categories as all zeros.

        Raises:
            ValueError: If handle_unknown is not "error" or "ignore".
        """
        validate_options(
            handle_unknown, ("error", "ignore"), x_name="handle_unknown"
        )
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Store unique categories for each input column.

        Parameters:
            X: 2D categorical input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            self.

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features log n_samples), space O(k), where k
            is the total number of stored categories.
        """
        X = validate_array_like(X, name="X")
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")
        n_cols = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(n_cols)]
        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        """Convert categorical columns to one-hot encoded columns.

        Parameters:
            X: 2D categorical input data with shape (n_samples, n_features).

        Returns:
            Encoded array with shape (n_samples, n_encoded_features).

        Shapes:
            n_encoded_features is the total number of fitted categories.

        Raises:
            ValueError: If X is not 2D, the encoder is not fitted, the feature
                count differs from fit, or unknown categories are found when
                handle_unknown is "error".

        Complexity:
            Time O(n_samples * n_encoded_features), space
            O(n_samples * n_encoded_features).
        """
        X = validate_array_like(X, name="X")
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted. Call fit() first.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input has different number of features than fit data.")
        encoded_columns = []
        
        for i in range(X.shape[1]):
            cats = self.categories_[i]
            col_data = X[:, [i]]
            unknown_mask = ~np.isin(X[:, i], cats)
            if np.any(unknown_mask) and self.handle_unknown == "error":
                unknown_vals = np.unique(X[unknown_mask, i])
                raise ValueError(
                    f"Unknown category in column {i}: {unknown_vals.tolist()}"
                )
            
            one_hot_col = (col_data == cats).astype(float)
            encoded_columns.append(one_hot_col)
        
        return np.hstack(encoded_columns)

    def fit_transform(self, X, y=None):
        """Fit the encoder and return one-hot encoded data.

        Parameters:
            X: 2D categorical input data with shape (n_samples, n_features).
            y: Not used, kept for API consistency.

        Returns:
            Encoded array with shape (n_samples, n_encoded_features).

        Shapes:
            n_encoded_features is the total number of fitted categories.

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features log n_samples +
            n_samples * n_encoded_features), space
            O(n_samples * n_encoded_features).
        """
        return self.fit(X, y).transform(X)


class SimpleImputer:
    def __init__(self, strategy='mean'):
        """Create an imputer that fills missing values.

        Parameters:
            strategy: Method used to fill missing values. Only "mean" is
                supported.

        Raises:
            ValueError: If strategy is not "mean".
        """
        validate_options(strategy, ("mean",), x_name="strategy")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        """Compute fill values for each feature.

        Parameters:
            X: 2D input data with shape (n_samples, n_features). May contain
                NaN.
            y: Not used, kept for API consistency.

        Returns:
            self.

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        X = validate_non_empty_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        """Replace NaN values using statistics from fit.

        Parameters:
            X: 2D input data with shape (n_samples, n_features). May contain
                NaN.

        Returns:
            Filled array with shape (n_samples, n_features).

        Raises:
            ValueError: If the imputer is not fitted or X is not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        X = validate_array_like(X, name="X").astype(float)
        if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        X_filled = np.where(np.isnan(X), self.statistics_, X)
        return X_filled

    def fit_transform(self, X, y=None):
        """Fit the imputer and return filled data.

        Parameters:
            X: 2D input data with shape (n_samples, n_features). May contain
                NaN.
            y: Not used, kept for API consistency.

        Returns:
            Filled array with shape (n_samples, n_features).

        Raises:
            ValueError: If X is empty or not 2D.

        Complexity:
            Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        return self.fit(X, y).transform(X)
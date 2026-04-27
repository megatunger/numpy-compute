import numpy as np

class StandardScaler:
    def __init__(self):
        """Create a scaler that standardizes each feature

        Parameters:
            None
        """
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute mean and standard deviation from input data

        Parameters:
            X: 2D input data
            y: Not used, kept for API consistency
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("StandardScaler expects 2D input.")
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # If STD is 0, set scale to 1 to avoid divide 0
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """Scale input data using stats learned in fit

        Parameters:
            X: 2D input data to scale
        """
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
        
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled

    def fit_transform(self, X, y=None):
        """Fit the scaler and return scaled data

        Parameters:
            X: 2D input data
            y: Not used, kept for API consistency
        """
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        """Create a scaler that rescales features to a range

        Parameters:
            feature_range: Tuple like (min, max) for output scale
        """
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None
        self.data_max_ = None

    def fit(self, X, y=None):
        """Compute per-feature min and max from input data

        Parameters:
            X: 2D input data
            y: Not used, kept for API consistency
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 2D input.")
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, X):
        """Rescale input data using min and range from fit

        Parameters:
            X: 2D input data to rescale
        """
        X = np.asarray(X, dtype=float)
        if self.data_min_ is None or self.data_range_ is None:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
        X_scaled = (X - self.data_min_) / self.data_range_
        return X_scaled

    def fit_transform(self, X, y=None):
        """Fit the scaler and return scaled data

        Parameters:
            X: 2D input data
            y: Not used, kept for API consistency
        """
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    def __init__(self, handle_unknown="error"):
        """Create an encoder for categorical columns

        Parameters:
            handle_unknown: What to do with unknown categories
        """
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Store unique categories for each input column

        Parameters:
            X: 2D categorical input data
            y: Not used, kept for API consistency
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")
        n_cols = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(n_cols)]
        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        """Convert categorical columns to one-hot encoded columns

        Parameters:
            X: 2D categorical input data to encode
        """
        X = np.asarray(X)
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
        """Fit the encoder and return one-hot encoded data

        Parameters:
            X: 2D categorical input data
            y: Not used, kept for API consistency
        """
        return self.fit(X, y).transform(X)


class SimpleImputer:
    def __init__(self, strategy = 'mean'):
        """Create an imputer that fills missing values

        Parameters:
            strategy: Method used to fill missing values
        """
        if strategy not in ('mean'):
            raise NotImplementedError("This strategy has not been supported yet")
        self.statistics_ = None

    def fit(self, X, y=None):
        """Compute fill values for each feature

        Parameters:
            X: 2D input data, can contain NaN
            y: Not used, kept for API consistency
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        """Replace NaN values using statistics from fit

        Parameters:
            X: 2D input data with possible NaN values
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        X_filled = np.copy(X)
        X_filled = np.where(np.isnan(X), self.statistics_, X)
        return X_filled

    def fit_transform(self, X, y=None):
        """Fit the imputer and return filled data

        Parameters:
            X: 2D input data, can contain NaN
            y: Not used, kept for API consistency
        """
        return self.fit(X, y).transform(X)
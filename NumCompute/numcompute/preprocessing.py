import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("StandardScaler expects 2D input.")
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # If STD is 0, set scale to 1 to avoid divide 0
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
        
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects 2D input.")
        raise NotImplementedError("Implement data_min_ and data_range_ computation.")

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.data_min_ is None or self.data_range_ is None:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
        raise NotImplementedError("Implement min-max scaling transformation.")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects 2D input.")
        raise NotImplementedError("Implement category discovery per feature.")

    def transform(self, X):
        X = np.asarray(X)
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted. Call fit() first.")
        raise NotImplementedError("Implement one-hot encoding transformation.")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class SimpleImputer:
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SimpleImputer expects 2D input.")
        raise NotImplementedError("Implement NaN replacement with fill_value.")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
"""Simple pipeline utilities for chaining preprocessing and model steps."""

from typing import Any, Protocol


class Transformer(Protocol):
    """Protocol for preprocessing classes."""

    def fit(self, X, y=None):
        """Fit the transformer using input data."""
        pass

    def transform(self, X):
        """Transform input data after fitting."""
        pass


class Estimator(Protocol):
    """Protocol for model classes."""

    def fit(self, X, y=None):
        """Fit the model using input data and optional targets."""
        pass

    def predict(self, X):
        """Predict target values from input data."""
        pass


class Pipeline:
    """Create a pipeline from named steps.

    Parameters:
        steps: List of (name, object) pairs
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline needs at least one step.")

        self.steps = list(steps)
        self.named_steps = {}

        for step_name, step_obj in self.steps:
            if step_name in self.named_steps:
                raise ValueError(f"Duplicate step name found: {step_name}")

            self.named_steps[step_name] = step_obj

    def fit(self, X, y=None):
        """Fit each step in order.

        Parameters:
            X: Input data
            y: Optional target data
        """
        data = X

        for step_name, step_obj in self.steps[:-1]:
            self._require_method(step_obj, "fit", step_name)
            self._require_method(step_obj, "transform", step_name)

            step_obj.fit(data, y)
            data = step_obj.transform(data)

        final_name, final_step = self.steps[-1]
        self._require_method(final_step, "fit", final_name)

        final_step.fit(data, y)

        return self

    def transform(self, X):
        """Transform data through every step.

        Parameters:
            X: Input data to transform
        """
        data = X

        for step_name, step_obj in self.steps:
            self._require_method(step_obj, "transform", step_name)
            data = step_obj.transform(data)

        return data

    def fit_transform(self, X, y=None):
        """Fit the pipeline and return transformed data.

        Parameters:
            X: Input data
            y: Optional target data
        """
        self.fit(X, y)

        transformed_data = self.transform(X)
        return transformed_data

    def predict(self, X):
        """Transform input data and predict with the final step.

        Parameters:
            X: Input data to predict
        """
        data = X

        for step_name, step_obj in self.steps[:-1]:
            self._require_method(step_obj, "transform", step_name)
            data = step_obj.transform(data)

        model_name, model = self.steps[-1]
        self._require_method(model, "predict", model_name)

        predictions = model.predict(data)
        return predictions

    def _require_method(self, step: Any, method_name: str, step_name: str):
        """Check that a step has a required method."""
        has_needed_method = hasattr(step, method_name)

        if not has_needed_method:
            raise TypeError(
                f"Step '{step_name}' must have a '{method_name}' method."
            )

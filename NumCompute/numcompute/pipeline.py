"""Simple pipeline utilities for chaining preprocessing and model steps."""

from typing import Any, Protocol


class Transformer(Protocol):
    """Protocol for preprocessing classes."""

    def fit(self, X, y=None):
        """Fit the transformer using input data.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).
            y: Optional target data, usually shape (n_samples,).

        Returns:
            A fitted transformer.
        """
        pass

    def transform(self, X):
        """Transform input data after fitting.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).

        Returns:
            Transformed data.
        """
        pass


class Estimator(Protocol):
    """Protocol for model classes."""

    def fit(self, X, y=None):
        """Fit the model using input data and optional targets.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).
            y: Optional target data, usually shape (n_samples,).

        Returns:
            A fitted estimator.
        """
        pass

    def predict(self, X):
        """Predict target values from input data.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).

        Returns:
            Predicted values, usually shape (n_samples,).
        """
        pass


class Pipeline:
    """Create a pipeline from named steps.

    Parameters:
        steps: List of (name, object) pairs.

    Raises:
        ValueError: If steps is empty or has duplicate names.
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
            X: Input data, usually shape (n_samples, n_features).
            y: Optional target data, usually shape (n_samples,).

        Returns:
            self.

        Shapes:
            Each non-final step may change X. The final step receives the last
            transformed shape.

        Raises:
            TypeError: If a step is missing fit or transform.
            Any exception raised by a step.

        Complexity:
            Time is the sum of step fit and transform costs. Space is the
            largest intermediate array plus fitted step storage.
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
            X: Input data, usually shape (n_samples, n_features).

        Returns:
            Transformed data from the final step.

        Shapes:
            Each step may change the shape.

        Raises:
            TypeError: If a step is missing transform.
            Any exception raised by a step.

        Complexity:
            Time is the sum of step transform costs. Space is the largest
            intermediate array.
        """
        data = X

        for step_name, step_obj in self.steps:
            self._require_method(step_obj, "transform", step_name)
            data = step_obj.transform(data)

        return data

    def fit_transform(self, X, y=None):
        """Fit the pipeline and return transformed data.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).
            y: Optional target data, usually shape (n_samples,).

        Returns:
            Transformed data from the final step.

        Shapes:
            Each step may change the shape.

        Raises:
            TypeError: If a step is missing fit or transform.
            Any exception raised by a step.

        Complexity:
            Time is fit plus transform cost. Space is the largest intermediate
            array plus fitted step storage.
        """
        data = X

        for step_name, step_obj in self.steps:
            self._require_method(step_obj, "fit", step_name)
            self._require_method(step_obj, "transform", step_name)

            step_obj.fit(data, y)
            data = step_obj.transform(data)

        return data

    def predict(self, X):
        """Transform input data and predict with the final step.

        Parameters:
            X: Input data, usually shape (n_samples, n_features).

        Returns:
            Predictions from the final step.

        Shapes:
            Predictions are usually shape (n_samples,), but depend on the final
            step.

        Raises:
            TypeError: If a preprocessing step is missing transform or the final
                step is missing predict.
            Any exception raised by a step.

        Complexity:
            Time is preprocessing transform cost plus final predict cost. Space
            is the largest intermediate array plus predictions.
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
        """Check that a step has a required method.

        Parameters:
            step: Step object to check.
            method_name: Required method name.
            step_name: Name used in the error message.

        Raises:
            TypeError: If step does not have method_name.
        """
        has_needed_method = hasattr(step, method_name)

        if not has_needed_method:
            raise TypeError(
                f"Step '{step_name}' must have a '{method_name}' method."
            )

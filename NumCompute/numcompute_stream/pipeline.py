"""Pipeline that chains streaming partial_fit steps"""

from numcompute.pipeline import Pipeline as BatchPipeline


class Pipeline(BatchPipeline):
    """Pipeline you can train one chunk at a time"""

    def partial_fit(self, X, y=None):
        """Train every step on one chunk, transformers then model

        X - input data for this chunk
        y - targets for this chunk (passed to each step for API consistency)

        Returns self so you can loop over chunks

        Raises TypeError if a step is missing partial_fit or transform
        """
        data = X

        # step 1: feed each preprocessor chunk, then pass transformed data on
        for step_name, step_obj in self.steps[:-1]:
            self._require_method(step_obj, "partial_fit", step_name)
            self._require_method(step_obj, "transform", step_name)
            step_obj.partial_fit(data, y)
            data = step_obj.transform(data)

        # step 2: final step is the model — partial_fit only, no transform
        final_name, final_step = self.steps[-1]
        self._require_method(final_step, "partial_fit", final_name)
        final_step.partial_fit(data, y)

        return self

import numpy as np

from numcompute.utils import validate_vector, validate_array_like, validate_options


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of a scalar function f at point x.

    f must accept a batch with shape (batch_size, n_features) and return one
    scalar per row.

    Parameters:
        f: Function to differentiate.
        x: Point where the gradient is estimated, shape (n_features,).
        h: Step size for finite differences.
        method: "central" or "forward".

    Returns:
        Gradient array with shape (n_features,).

    Shapes:
        f is called with shape (n_features, n_features) for perturbed points.
        For forward differences, f is also called with shape (1, n_features).

    Raises:
        ValueError: If method is invalid, x is not a vector, or f does not return
            one scalar per input row.

    Complexity:
        Time O(cost(f)) for the batched finite-difference call. Space
        O(n_features^2) for the perturbation matrix.
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    validate_vector(x)
    n = x.size

    # Each row changes exactly one coordinate, so f evaluates all axes at once.
    perturbations = h * np.eye(n)
    f_plus = np.asarray(f(x + perturbations), dtype=float).ravel()
    if f_plus.size != n:
        raise ValueError("f must return one scalar value per input row.")

    if method == "central":
        f_minus = np.asarray(f(x - perturbations), dtype=float).ravel()
        if f_minus.size != n:
            raise ValueError("f must return one scalar value per input row.")
        return (f_plus - f_minus) / (2 * h)

    # Forward differences reuse the single value at the unperturbed point.
    f_at_base = np.asarray(f(x[np.newaxis, :]), dtype=float).ravel()
    if f_at_base.size != 1:
        raise ValueError("f must return one scalar value per input row.")
    return (f_plus - f_at_base[0]) / h


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate the Jacobian matrix of a vector-valued function F at point x.

    F must accept a batch with shape (batch_size, n_features) and return shape
    (batch_size, n_outputs).

    Parameters:
        F: Function to differentiate.
        x: Point where the Jacobian is estimated, shape (n_features,).
        h: Step size for finite differences.
        method: "central" or "forward".

    Returns:
        Jacobian array with shape (n_outputs, n_features).

    Shapes:
        F is first called with shape (1, n_features). Each output component is
        then passed through grad.

    Raises:
        ValueError: If method is invalid, x is not a vector, or F returns an
            array with the wrong shape.

    Complexity:
        Time O(n_outputs * cost(F)) through grad. Space
        O(n_outputs * n_features + n_features^2).
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    validate_vector(x)
    f_at_x = np.asarray(F(x[np.newaxis, :]), dtype=float)
    if f_at_x.ndim != 2 or f_at_x.shape[0] != 1:
        raise ValueError("F must return a 2D array with one row per input point.")
    m = f_at_x.shape[1]

    J = np.zeros((m, x.size))

    # A Jacobian row is the gradient of one output component of F.
    for output_index in range(m):
        def component_function(points, output_index=output_index):
            values = np.asarray(F(points), dtype=float)
            if values.ndim != 2 or values.shape[0] != points.shape[0]:
                raise ValueError("F must return shape (batch_size, m).")
            return values[:, output_index]

        J[output_index, :] = grad(component_function, x, h=h, method=method)

    return J

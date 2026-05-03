import numpy as np

from numcompute.utils import _validate_vector, validate_array_like, validate_options


def grad_loop(f, x, h=1e-5, method="central"):
    """Estimate a gradient using loop-based finite differences.

    f must accept a 2D batch of shape (batch_size, n) and return one scalar
    per input row.
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    _validate_vector(x)

    gradient = np.zeros_like(x)
    f_at_base = None
    if method == "forward":
        f_at_base = np.asarray(f(x[np.newaxis, :]), dtype=float).ravel()
        if f_at_base.size != 1:
            raise ValueError("f must return one scalar value per input row.")

    for coord_index in range(x.size):
        x_plus_h = np.copy(x)
        x_plus_h[coord_index] += h
        f_plus = np.asarray(f(x_plus_h[np.newaxis, :]), dtype=float).ravel()
        if f_plus.size != 1:
            raise ValueError("f must return one scalar value per input row.")

        if method == "central":
            x_minus_h = np.copy(x)
            x_minus_h[coord_index] -= h
            f_minus = np.asarray(f(x_minus_h[np.newaxis, :]), dtype=float).ravel()
            if f_minus.size != 1:
                raise ValueError("f must return one scalar value per input row.")
            gradient[coord_index] = (f_plus[0] - f_minus[0]) / (2 * h)
        else:
            gradient[coord_index] = (f_plus[0] - f_at_base[0]) / h

    return gradient


def jacobian_loop(F, x, h=1e-5, method="central"):
    """Estimate a Jacobian by reusing grad_loop for each output component."""
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    _validate_vector(x)

    f_at_x = np.asarray(F(x[np.newaxis, :]), dtype=float)
    if f_at_x.ndim != 2 or f_at_x.shape[0] != 1:
        raise ValueError("F must return a 2D array with one row per input point.")

    m = f_at_x.shape[1]
    J = np.zeros((m, x.size))

    for output_index in range(m):
        def component_function(points, output_index=output_index):
            values = np.asarray(F(points), dtype=float)
            if values.ndim != 2 or values.shape[0] != points.shape[0]:
                raise ValueError("F must return shape (batch_size, m).")
            return values[:, output_index]

        J[output_index, :] = grad_loop(component_function, x, h=h, method=method)

    return J

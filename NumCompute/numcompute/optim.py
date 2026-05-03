import numpy as np

from numcompute.utils import _validate_vector, validate_array_like, validate_options


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of a scalar function f at point x

    Given batched f: R^(batch_size x n) -> R^(batch_size), return a 1D
    array whose entry i approximates the partial derivative ∂f/∂x_i at x.

    Parameters:
        f: Callable taking a 2D array-like of shape (batch_size, n) and
           returning one scalar per input row
        x: 1D array-like, point at which to estimate the gradient
        h: Small step size used for finite differences
        method: 'central' for (f(x+h) - f(x-h)) / (2h),
                'forward' for (f(x+h) - f(x))   / h
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    _validate_vector(x)
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
    """Estimate the Jacobian matrix of a vector-valued function F at point x

    Given batched F: R^(batch_size x n) -> R^(batch_size x m), return a
    2D array J of shape (m, n) where J[i, j] approximates the partial
    derivative of F_i with respect to x_j.

    Parameters:
        F: Callable taking a 2D array-like of shape (batch_size, n) and
           returning a 2D array-like of shape (batch_size, m)
        x: 1D array-like of length n, point at which to estimate the Jacobian
        h: Small step size used for finite differences
        method: 'central' or 'forward' (same meaning as in `grad`)
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = validate_array_like(x, name="input").astype(float)
    _validate_vector(x)
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

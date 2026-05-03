import numpy as np

from numcompute.utils import validate_options


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of a scalar function f at point x

    Given f: R^n -> R, return a 1D array of the same shape as x whose entry i
    approximates the partial derivative ∂f/∂x_i at x.

    Parameters:
        f: Callable taking a 1D array-like and returning a scalar
        x: 1D array-like, point at which to estimate the gradient
        h: Small step size used for finite differences
        method: 'central' for (f(x+h) - f(x-h)) / (2h),
                'forward' for (f(x+h) - f(x))   / h
    """
    validate_options(method, ("central", "forward"), x_name="method")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("grad expects a 1D array x.")
    # Estimated ∂f/∂x_i for each coordinate i (finite differences along each axis).
    gradient = np.zeros_like(x)
    # Forward stencil uses one baseline value f(x); central compares ±h without needing it.
    f_at_base = f(x) if method == "forward" else None
    for coord_index in range(x.size):
        x_plus_h = np.copy(x)
        x_plus_h[coord_index] += h
        if method == "central":
            x_minus_h = np.copy(x)
            x_minus_h[coord_index] -= h
            gradient[coord_index] = (f(x_plus_h) - f(x_minus_h)) / (2 * h)
        else:
            gradient[coord_index] = (f(x_plus_h) - f_at_base) / h
    return gradient


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate the Jacobian matrix of a vector-valued function F at point x

    Given F: R^n -> R^m, return a 2D array J of shape (m, n) where
    J[i, j] approximates the partial derivative of F_i with respect to x_j.

    Parameters:
        F: Callable taking a 1D array-like of length n and returning
           a 1D array-like of length m
        x: 1D array-like of length n, point at which to estimate the Jacobian
        h: Small step size used for finite differences
        method: 'central' or 'forward' (same meaning as in `grad`)
    """
    # Step 1: validate `method` (same rule as in grad).

    # Step 2: convert x to a 1D float ndarray.

    # Step 3: probe the output dimension `m`:
    #         call F once at x (or at a copy of x) and look at the
    #         length of the returned array. This tells you how many
    #         rows the Jacobian will have.
    #
    #         Hint: np.asarray(F(x), dtype=float).ravel() gives a clean 1D vector.

    # Step 4: allocate J as a zero matrix of shape (m, n).

    # Step 5: loop over each input coordinate j in range(n):
    #           - perturb only x[j] by +h (and -h for central) on a COPY of x
    #           - evaluate F at the perturbed point(s)
    #           - compute the finite-difference column vector:
    #               * forward: (F(x + h e_j) - F(x))         / h
    #               * central: (F(x + h e_j) - F(x - h e_j)) / (2h)
    #           - store this vector into the j-th COLUMN of J,  i.e. J[:, j].

    # Step 6: return J.
    #
    # Bonus thinking question (not required to code):
    #   How could you re-use `grad` to build `jacobian`? What would
    #   you call `grad` on, and what is the trade-off vs. the loop above?
    raise NotImplementedError("jacobian: implement me using finite differences")

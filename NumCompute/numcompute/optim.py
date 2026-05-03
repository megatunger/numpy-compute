import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of a scalar function f at point x

    Given f: R^n -> R, return an array g of the same shape as x where
    g[i] approximates the partial derivative of f with respect to x[i].

    Parameters:
        f: Callable taking a 1D array-like and returning a scalar
        x: 1D array-like, point at which to estimate the gradient
        h: Small step size used for finite differences
        method: 'central' for (f(x+h) - f(x-h)) / (2h),
                'forward' for (f(x+h) - f(x))   / h
    """
    # Step 1: validate `method` is one of {'central', 'forward'}.
    #         Raise ValueError otherwise (look at how preprocessing.py
    #         validates `handle_unknown` for inspiration).

    # Step 2: convert x to a 1D float ndarray with np.asarray(..., dtype=float).
    #         Optionally check x.ndim == 1 and raise ValueError if not.

    # Step 3: allocate an output array `g` with the same shape as x,
    #         filled with zeros (np.zeros_like is handy here).

    # Step 4: loop over each coordinate i in range(x.size):
    #           - build x_plus  = a copy of x with x_plus[i]  += h
    #           - build x_minus = a copy of x with x_minus[i] -= h   (only if central)
    #           - compute the finite-difference quotient using f(...)
    #           - store the result into g[i]
    #
    #         Hint: np.copy(x) gives you an independent copy you can mutate.
    #         Hint: do NOT mutate x itself.

    # Step 5: return g.
    raise NotImplementedError("grad: implement me using finite differences")


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

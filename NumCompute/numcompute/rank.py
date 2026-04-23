import numpy as np
from typing import Union
from numcompute.utils import _validate_vector

def rank(a: np.ndarray, method: str='average') -> np.ndarray:
    """
    Assign numerical ranks to elements in a vector, with three supported
    ranking strategies:
    - 'ordinal' : ties are broken by order of appearance (first seen =
      lower rank).
    - 'dense'   : ties share the lowest rank in their group, and the
      next distinct value receives the immediately following rank (no gaps).
    - 'average' : ties share the average of the ranks they
      would have occupied (default).
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
    method : str, optional
        Tie-breaking strategy. Can be 'average', 'dense', or
        'ordinal'. Default is 'average'.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a, containing the rank of each element,
        with the smallest value receiving rank 1.
 
    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1.
        If method is not 'average', 'dense', or 'ordinal'.
 
    Complexity
    ----------
    Time  : O(nlogn) because of the sort call (which uses mergesort).
    Space : O(n) for the intermediate arrays.
    """
    _validate_vector(a)

    # ranks ties based on order of appearance
    if method == 'ordinal':
        order = np.argsort(a, kind='stable')
        return np.argsort(order, kind='stable') + 1

    a_sorted, ranks, counts = np.unique(a, return_inverse=True, return_counts=True)

    if method == 'dense':
        return ranks + 1

    if method == 'average':
        # get the starting rank of each ties
        # the first tie always start at 1 so we add 1 at the start
        # the following ones = the prev + the next count
        start_ranks = np.cumsum(np.concatenate(([1], counts[:-1])))
        # average ranking = start rank + last rank / 2
        average_ranks = start_ranks + (counts - 1) / 2.0
        # apply the dense rank to get the correct order
        return average_ranks[ranks]

    raise ValueError("Unknown ranking method received! Valid methods are 'average', 'dense', or 'ordinal'.")

def percentile(a: np.ndarray, q: Union[float, list[float]], interpolation: str='linear') -> Union[np.ndarray, np.floating]:
    """
    Compute the q-th percentile(s) of a vector (through numpy's np.percentile), with 
    four supported interpolation strategies:
    - 'linear'   : weighted average: i + (j - i) - fraction (default).
    - 'lower'    : always take the lower value i.
    - 'higher'   : always take the higher value j.
    - 'midpoint' : average of i and j.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
    q : float or list of floats
        Percentile(s) to compute, in the range [0, 100].
    interpolation : str, optional
        Method used when the target percentile falls between two adjacent
        data points i and j (where i < j). Can be 'linear', 'lower', 'higher',
        or 'midpoint'. Default is 'linear'.
 
    Returns
    -------
    np.floating
        When q is a scalar: a single float percentile value.
    np.ndarray
        When q is a list of floats: a vector with shape (len(q),)
        containing the corresponding percentile float values.
 
    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1.
        If interpolation is not 'linear', 'lower', 'higher', or 'midpoint'.
        If any value in q is outside of the range [0, 100].
 
    Complexity
    ----------
    Time  : O(nlogn) because numpy's np.percentile sorts the array internally.
    Space : O(n) for the sorted copy.
    """
    _validate_vector(a)
    
    methods = ['linear', 'lower', 'higher', 'midpoint']
    if interpolation not in methods:
        raise ValueError("Unknown interpolation method received! Valid methods are 'linear', 'lower', 'higher', or 'midpoint'.")
    if isinstance(q, list):
        temp = np.asarray(q)
        if np.any(temp < 0) or np.any(temp > 100):
            raise ValueError(f"Invalid percentiles received!. Got {q}")
    else:
        if q < 0 or q > 100:
            raise ValueError(f"Invalid percentiles received!. Got {q}")

    return np.percentile(a, q, method=interpolation)

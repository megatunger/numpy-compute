# Author: DUY ANH
# - Mean, median, standard deviation, min, max
# - Histogram
# - Quantiles (with NaN handling)
# - Axis-wise stats with clear dimension/shape behaviour

import numpy as np


def mean(arr: np.ndarray) -> np.ndarray:
    """Compute the arithmetic mean.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        mean (np.ndarray)
    """

    return arr.mean()


def median(arr: np.ndarray) -> np.ndarray:
    """Compute the median.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        median (np.ndarray)
    """

    return arr.median()


def std(arr: np.ndarray) -> np.ndarray:
    """Compute the standard deviation.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        std (np.ndarray)
    """

    return arr.std()


def min(arr: np.ndarray) -> np.ndarray:
    """Return the minimum.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        min (np.ndarray)
    """

    return arr.min()


def max(arr: np.ndarray) -> np.ndarray:
    """Return the maximum.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        max (np.ndarray)
    """

    return arr.max()


def histogram(
    arr: np.ndarray,
    bins: int = 10,
    range: tuple = None
) -> np.ndarray:
    """Compute the histogram of a dataset.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        histogram (np.ndarray)
    """

    return np.histogram(arr, bins=bins, range=range)[0]

def quantile(
    arr: np.ndarray,
    q: int
) -> np.ndarray:
    """Compute the histogram of a dataset.

    Args:
        arr (np.ndarray): array of any shape
    Returns:
        histogram (np.ndarray)
    """

    return np.quantile(arr, q)

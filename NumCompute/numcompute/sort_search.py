import numpy as np

### SORTING

def stable_sort(a: np.ndarray) -> np.ndarray:
    return np.sort(a, kind='stable')

def multi_key_sort(a: np.ndarray, columns: list[int]) -> np.ndarray:
    """
    Example:
        columns = [0, 1] sorts by column 0 first, then column 1.
    """
    indices = np.lexsort([a[:, col] for col in reversed(columns)])
    return a[indices]

import numpy as np
from typing import Union

### SORTING

def stable_sort_loop(a: np.ndarray) -> np.ndarray:
    result = a.tolist()
    n = len(result)

    for i in range(1, n):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key

    return np.array(result)

def _compare_multi_key(row_a: list, row_b: list, columns: list[int]) -> bool:
    """
    Helper for multi_key_sort_loop
    """
    for col in columns:
        if row_a[col] < row_b[col]:
            return False
        elif row_a[col] > row_b[col]:
            return True
    return False

def multi_key_sort_loop(a: np.ndarray, columns: list[int]) -> np.ndarray:
    result = a.tolist()
    n = len(result)

    for i in range(1, n):
        key = result[i]
        j = i - 1
        while j >= 0 and _compare_multi_key(result[j], key, columns) > 0:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key

    return np.array(result)

def topk_loop(a: np.ndarray, k: int, largest: bool = True, return_indices: bool = True) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:

    topk_pairs = list(enumerate(a.tolist()))

    topk_pairs.sort(key=lambda x: x[1], reverse=largest)
    values = np.array([p[1] for p in topk_pairs[:k]])

    if not return_indices:
        return values

    indices = np.array([p[0] for p in topk_pairs[:k]])
    return values, indices
def binary_search_loop(a: np.ndarray, x: Union[int, float]) -> tuple[int, bool]:
    result = a.tolist()
    left = 0
    right = len(result)

    while left < right:
        mid = (left + right) // 2
        if result[mid] < x:
            left = mid + 1
        else:
            right = mid

    found = left < len(result) and result[left] == x
    return left, found

def select(a: list, left: int, right: int, k: int) -> Union[int, float]:
    if left == right:
        return a[left]

    pivot = a[np.random.randint(left, right)]
    low, mid, high = left, left, right
    while mid <= high:
        if a[mid] < pivot:
            a[low], a[mid] = a[mid], a[low]
            low += 1
            mid += 1
        elif a[mid] == pivot:
            mid += 1
        else:
            a[mid], a[high] = a[high], a[mid]
            high -= 1
    if low <= k <= mid - 1:
        return a[k]
    elif k < low:
        return select(a, left, low - 1, k)
    else:
        return select(a, mid, right, k)

def quickselect_loop(a: np.ndarray, k: int) -> Union[int, float]:
    if k < 0 or k > a.size - 1:
        raise ValueError(f"k value must be in [0, {a.size - 1}], got: {k}")

    a_temp = a.tolist()  
    return select(a_temp, 0, len(a) - 1, k)

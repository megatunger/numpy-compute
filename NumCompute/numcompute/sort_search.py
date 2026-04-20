import numpy as np
from typing import Union

### SORTING

def stable_sort(a: np.ndarray) -> np.ndarray:
    return np.sort(a, kind='stable')

def multi_key_sort(a: np.ndarray, columns: list[int]) -> np.ndarray:
    """
    Example:
        columns = [0, 1] sorts by column 0 first, then column 1.
    """
    # revsered since last key is primary key in lexsort
    indices = np.lexsort([a[:, col] for col in reversed(columns)])
    return a[indices]


### TOP K

def topk(a: np.ndarray, k: int, largest: bool = True, return_indices: bool = True) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    # argparition is O(n)

    indices = []
    order = []
    # argpartition gets the top k values' indices, but unsorted
    # argsort gets the correct sorted order of the indices above
    if largest:
        indices = np.argpartition(a, -k)[-k:]
        order = np.argsort(a[indices])[::-1]
    else:
        indices = np.argpartition(a, k)[:k]
        order = np.argsort(a[indices])

    # apply the sorted order to the indices to get the correct indices and values
    top_k_indices = indices[order]
    top_k = a[top_k_indices]

    if return_indices:
        return (top_k, top_k_indices)
    return top_k


### SEARCH
def binary_search(a: np.ndarray, x: int | float) -> tuple[int, bool]:
    """
    Currently inserts to the left, if the same value is already in the array.
    """
    # searchsorted uses binary search
    index = np.searchsorted(a, x)
    return (int(index), index >= 0 and index < len(a) and a[index] == x)

# data = np.array([[3, 2], [1, 5], [1, 3], [2, 4]])
# print(multi_key_sort(data, [0, 1]))

### QUICKSELECT

def select(a: list, left: int, right: int, k: int) -> int | float:
    if left == right:
        return a[left]

    pivot = a[np.random.randint(left, right)]
    temp = left
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

def quickselect(a: np.ndarray, k: int) -> int | float:
    """
    """
    a = a.tolist()  

    return select(a, 0, len(a) - 1, k)


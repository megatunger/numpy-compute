import numpy as np

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

def topk(a: np.ndarray, k: int, largest: bool = True, return_indices: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:

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

data = np.array([[3, 2], [1, 5], [1, 3], [2, 4]])
print(multi_key_sort(data, [0, 1]))

### QUICKSELECT

def select(a: list, left: int, right: int, k: int) -> int | float:
    if left == right:
        return a[left]

    pivot = a[right]
    temp = left
    for i in range(left, right):
        if a[i] <= pivot:
            a[i], a[temp] = a[temp], a[i]
            temp += 1
    a[temp], a[right] = a[right], a[temp]

    if k == temp:
        return a[temp]
    elif k < temp:
        return select(a, left, temp - 1, k)
    else:
        return select(a, temp + 1, right, k)

def quickselect(a: np.ndarray, k: int) -> int | float:
    """
    """
    a = a.tolist()  

    return select(a, 0, len(a) - 1, k)


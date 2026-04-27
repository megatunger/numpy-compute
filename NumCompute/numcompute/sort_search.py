import numpy as np
from typing import Union
from numcompute.utils import _validate_vector

### SORTING

def stable_sort(a: np.ndarray) -> np.ndarray:
    """
    Sort a vector in ascending order, preserving the relative order of
    equal elements (stable sort). Uses NumPy's mergesort under the hood.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a.

    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1.
 
    Complexity
    ----------
    Time  : O(nlogn) because numpy's np.sort uses merge sort.
    Space : O(n) because mergesort requires auxiliary memory proportional to input size.
    """
    _validate_vector(a)
    return np.sort(a, kind='stable')

def multi_key_sort(a: np.ndarray, columns: list[int]) -> np.ndarray:
    """
    Sort a 2D array by multiple columns, with earlier entries in columns
    taking higher priority.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
    columns : list[int]
        Ordered list of column indices to sort by. The first element is the
        primary sort key, the second is the secondary key, and so on.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a with rows sorted according to the
        specified column priority.
 
    Raises
    ------
    ValueError
        If any value in columns is out of bounds for the number of columns in a.
 
    Complexity
    ----------
    Time  : O(nlogn * len(columns)).
    Space : O(n) for the intermediate array indices.
    """
    temp = np.asarray(columns)
    if np.any((temp < 0) | (temp >= a.shape[1])):
        raise ValueError(f"All column indices must be in [0, {a.shape[1] - 1}], got: {columns}")
    # revsered since last key is primary key in lexsort
    indices = np.lexsort([a[:, col] for col in reversed(columns)])
    return a[indices]


### TOP K

def topk(a: np.ndarray, k: int, largest: bool = True, return_indices: bool = True) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Retrieve the top-k elements (and potentially indices) of a 1-D array, sorted 
    in descending order or ascending order.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
    k : int
        Number of elements to return.
    largest : bool, optional
        - True (default), return the k largest elements in descending order. 
        - False, return the k smallest elements in ascending order.
    return_indices : bool, optional
        - True (default), return a tuple of (values, indices).
        - False, return only the values.
 
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        When return_indices=True: a tuple (values, indices)
        where both have shape (k,). 
        - Values contain the top k elements
        sorted in the correct orderi
        - Indices store these values' indices with respect to 
        the original array
    np.ndarray
        When return_indices=False: a values array of shape (k,).
 
    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1, or k is less than 1 or greater than the length of a.
 
    Complexity
    ----------
    Time  : O(n + klogk) because O(n) partition + O(klogk) sort of k elements.
    Space : O(k) for values and indices.
    """
    _validate_vector(a)
    if k < 1 or k > a.size:
        raise ValueError(f"k value must be in [1, {a.size}], got: {k}")

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

def binary_search(a: np.ndarray, x: Union[int, float]) -> tuple[int, bool]:
    """
    Search for a value x in a sorted vector using binary search (through numpy's
    np.searchsorted, which implements binary search internally).

    A check to ensure a is sorted could have been added, but methods like np.all()
    can make the time complexity O(n), and so was not added.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,). MUST be sorted for correct behaviour.
    x : int or float
        The value to search for.
 
    Returns
    -------
    tuple[int, bool]
        A tuple (index, found) where:
        - index is the position at which x is found, or
          the position where it would be inserted to maintain sorted order.
          When duplicates of x exist, the leftmost valid position is
          returned.
         - found is True if x is found in a, False otherwise.
 
    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1.
 
    Complexity
    ----------
    Time  : O(logn) because binary search
    Space : O(1) because only index is used as an auxilliary value
    """
    _validate_vector(a)
    # searchsorted uses binary search
    index = np.searchsorted(a, x)
    return (int(index), index >= 0 and index < len(a) and a[index] == x)

### QUICKSELECT

def quickselect(a: np.ndarray, k: int) -> Union[int, float]:
    """
    Finds the element that would appear at index k in the
    sorted version of a.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
    k : int
        Target rank index.
 
    Returns
    -------
    int or float
        The value at index k in the sorted array.
 
    Raises
    ------
    ValueError
        If the input has a dimension not equal to 1, or k is less than 0 or greater than the length of a - 1.
 
    Complexity
    ----------
    Time  : O(n) average because numpy's argpartition.
    Space : O(n) because the copy of a (worst case of quick select is also only O(n)).
    """
    _validate_vector(a)
    if k < 0 or k > a.size - 1:
        raise ValueError(f"k value must be in [0, {a.size - 1}], got: {k}")

    a_temp = a.copy()  
    return a[np.argpartition(a, k)[k]]

import numpy as np
from typing import Union

### INPUT VALIDATION

def validate_array_like(a, name="input"):
    """Validate if an object is convertible to np.ndarray"""
    try:
        a = np.asanyarray(a)
    except ValueError as e:
        raise ValueError(
            f"Unable to convert {name} to numpy ndarray, got error: {e}"
        )
    else:
        return a
    
def validate_non_empty_array(a, name="input"):
    """Validate if array is non-empty"""
    if not (a.size > 0):
        raise ValueError(f"Expect {name} to be non-empty array")

    return a

def validate_metrics_array(y_true, y_pred):
    """Validate metrics input"""
    y_true = validate_array_like(y_true, name="targets")
    y_pred = validate_array_like(y_pred, name="predictions")
    y_true = validate_non_empty_array(y_true, name="targets")
    y_pred = validate_non_empty_array(y_pred, name="predictions")
    
    if not (y_true.shape == y_pred.shape):
        raise ValueError(
            f"Expect predictions and targets to have same shape, got {y_true.shape} and {y_pred.shape}"
        )

    try:
        y_pred = y_pred.astype(y_true.dtype)
    except ValueError:
        raise ValueError(
            f"Expect predictions and targets to have compatible dtype, got {y_true.dtype} and {y_pred.dtype}"
        )
        
    return y_true, y_pred

def validate_options(x, options, x_name="input"):
    """Validate if input in options"""
    if x not in options:
        raise ValueError(f"{x_name} must be one of the following options: {options}")
    
    return x, options

def _validate_vector(a: np.ndarray) -> None:
    """
    Validates that input a is a vector, and doesn't contain any NaN values.
 
    Parameters
    ----------
    a: np.ndarray
        Input vector of shape (n,).
 
    Raises
    ------
    ValueError
        If input does not have a dimension equal to 1.
        If either inputs contain a NaN value.
    """
    if a.ndim != 1:
        raise ValueError(f"The input vectors is not 1 dimension! Input had {a.ndim} dimension.")
    # only floating point arrays can contain Nan
    if np.issubdtype(a.dtype, np.floating):
        if np.isnan(a).any():
            raise ValueError(f"The input vector contains NaN values!")

def _validate_vectors(a: np.ndarray, b: np.ndarray) -> None:
    """
    Validates that inputs `a` and `b` are vectors, have the same shape, and don't contain
    any NaN values.
 
    Parameters
    ----------
    a: np.ndarray
        First input — vector of shape (n,).
    b: np.ndarray
        Second input — vector of shape (n,).
 
    Raises
    ------
    ValueError
        If either inputs have a dimension not equal to 1, or if both inputs don't share the same shape.
        If either inputs contain a NaN value.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"At least one of the input vectors is not 1 dimension! Inputs had {a.ndim} and {b.ndim} dimensions.")
    if a.shape != b.shape:
        raise ValueError(f"The inputs must have matching shapes! Inputs were shapes {a.shape} and {b.shape}.")
    # only floating point arrays can contain Nan
    if np.issubdtype(a.dtype, np.floating):
        if np.isnan(a).any() or np.isnan(b).any():
            raise ValueError(f"The inputs contain NaN values!")

### DISTANCES

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Euclidean distance between two input vectors.
 
    Parameters
    ----------
    a : np.ndarray
        First input — vector of shape (n,).
    b : np.ndarray
        Second input — vector of shape (n,).
 
    Returns
    -------
    float
        The calculated Euclidean distance between the two input vectors.
 
    Raises
    ------
    ValueError
        If either inputs have a dimension not equal to 1, or if both inputs don't share the same shape.
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.linalg.norm
    Space : O(n) for the intermediate a - b.
    """    
    _validate_vectors(a, b)
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Euclidean distance between two input vectors.
 
    Parameters
    ----------
    a : np.ndarray
        First input — vector of shape (n,).
    b : np.ndarray
        Second input — vector of shape (n,).
 
    Returns
    -------
    float
        The calculated Manhattan distance between the two input vectors.
 
    Raises
    ------
    ValueError
        If either inputs have a dimension not equal to 1, or if both inputs don't share the same shape.
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.sum and np.abs are both O(n)
    Space : O(n) for the intermediate a - b.
    """    
    _validate_vectors(a, b)
    return np.sum(np.abs(a - b))

def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Chebyshev distance between two input vectors.
 
    Parameters
    ----------
    a : np.ndarray
        First input — vector of shape (n,).
    b : np.ndarray
        Second input — vector of shape (n,).
 
    Returns
    -------
    float
        The calculated Chebyshev distance between the two input vectors.
 
    Raises
    ------
    ValueError
        If either inputs have a dimension not equal to 1, or if both inputs don't share the same shape.
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.sum and np.max are both O(n)
    Space : O(n) for the intermediate a - b.
    """    
    _validate_vectors(a, b)
    return np.max(np.abs(a - b))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Cosine similarity between two non-zero input vectors.
 
    Parameters
    ----------
    a : np.ndarray
        First input — vector of shape (n,).
    b : np.ndarray
        Second input — vector of shape (n,).
 
    Returns
    -------
    float
        The calculated Cosine similarity between the two input vectors.
 
    Raises
    ------
    ValueError
        If either inputs have a dimension not equal to 1, if both inputs don't share the same shape,
        or if either inputs is a zero vector.
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.linalg.norm and np.dot are both O(n) (for vectors).
    Space : O(n) for the intermediate normalised arrays norm_a, norm_b.
    """    
    _validate_vectors(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("At least one of the input vectors is a zero vector! Cosine similarity is undefined for the zero vector.")
    return np.dot(a, b) / (norm_a * norm_b)

# TODO: distance calculation for matrices as well?

def logsumexp(a: np.ndarray, axis: Union[int, None] = None) -> Union[float, np.ndarray]:    
    """
    Computes the numerically stable log of the sum of exponentials.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
    axis : int or None, optional
        Axis along which to compute. 
            - None (default) reduces over all elements
            - 0 reduces along rows 
            - 1 reduces along columns  

    Returns
    -------
    float or np.ndarray
        - axis=None : scalar float.
        - axis=0    : array of shape (d,).
        - axis=1    : array of shape (m,).
 
    Complexity
    ----------
    Time  : O(n) because numpy's functions used here are O(n) at worst.
    Space : O(n) for the intermediate arrays a_max, and a - a_max.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(a_max) + np.log(np.sum(np.exp(a - a_max), axis=axis))

### ACTIVATION FUNCTIONS

def tanh(a: np.ndarray) -> np.ndarray:
    """
    Applies the hyperbolic tangent activation.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a, with values in the interval (-1, 1).
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.tanh is O(n).
    Space : O(n) for the output array.
    """
    return np.tanh(a)

def relu(a: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit activation. Any negative values are zeroed, while positive values
    remain the same.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a, with values in the interval [0, +inf).
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.maximum is O(n).
    Space : O(n) for the output array.
    """
    return np.maximum(0, a)

def leaky_relu(a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Applies the Leaky Rectified Linear Unit activation. Any negative values are multiplied with a specified 
    alpha value, while positive values remain the same.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
    alpha : float, optional
        Small positive value that negativa values scales with. Default is 0.01.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a.
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.where is O(n).
    Space : O(n) for the output array.
    """
    return np.where(a > 0, a, alpha * a)

def sigmoid(a: np.ndarray) -> np.ndarray:
    """
    Applies the numerically stable Sigmoid activation. For negative inputs, the formula is
    changed to exp(a) / (1 + exp(a)), in order to avoid underflow.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a, with values in the interval (0, 1).
 
    Complexity
    ----------
    Time  : O(n) because numpy's np.where is O(n).
    Space : O(n) for the output array.
    """
    return np.where(a >= 0, 
                    1 / (1 + np.exp(-a)), 
                    np.exp(a) / (1 + np.exp(a)))

def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Applies the numerically stable Softmax activation along a specified axis.
 
    Parameters
    ----------
    a : np.ndarray
        Input - numpy array of any shape.
    axis : int, optional
        Axis along which softmax is computed.
            - -1 (default) chooses the last axis, which is row-wise
            - 0 applies column-wise
 
    Returns
    -------
    np.ndarray
        Numpy array with the same shape as a, with values in the interval (0, 1) and sums up to 1.
 
    Complexity
    ----------
    Time  : O(n) because numpy's functions used here are O(n) at worst.
    Space : O(n) for the intermediate arrays.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=axis, keepdims=True)


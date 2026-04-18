import numpy as np

### DISTANCES

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b))

def logsumexp(a: np.ndarray) -> float:
    # stable version of formula for numerical stability (avoid underflow/overflow issue)
    # source: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))

### ACTIVATION FUNCTIONS

def tanh(a: np.ndarray) -> np.ndarray:
    return np.tanh(a)

def relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

def leaky_relu(a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(a > 0, a, alpha * a)

def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))

def softmax(a: np.ndarray) -> np.ndarray:
    # subtract max for numerical stability
    exp_a = np.exp(a - np.max(a))  
    return exp_a / exp_a.sum()

def linear(a: np.ndarray) -> np.ndarray:
    return a

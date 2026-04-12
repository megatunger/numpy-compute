# Author: DUY ANH

import numpy as np

def accuracy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Accuracy classification score.

    Args:
        y_true (np.ndarray): one-hot array of shape [C,...], where C is the number of classes
        y_pred (np.ndarray): one-hot array of shape [C,...], where C is the number of classes, must be the same shape as y_true

    Returns:
        accuracy (np.ndarray): 1d-array of shape [C]: accuracy for each class
    """
    assert y_true.shape == y_pred.shape

    C = y_true.shape[0]
    y_true = y_true.reshape(C, -1)
    y_pred = y_pred.reshape(C, -1)
    N = y_true.shape[-1]
    acc = (y_true == y_pred).astype(np.float32) / N

    return float(acc)

def precision(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Precision classification score.

    Args:
        y_true (np.ndarray): one-hot array of shape [C,...], where C is the number of classes
        y_pred (np.ndarray): one-hot array of shape [C,...], where C is the number of classes, must be the same shape as y_true

    Returns:
        precision (np.ndarray): 1d-array of shape [C]: precision (TP/TP + FN) for each class
    """
    assert y_true.shape == y_pred.shape

    C = y_true.shape[0]
    y_true = y_true.reshape(C, -1)
    y_pred = y_pred.reshape(C, -1)

    TP = (y_true * y_pred).sum(axis=-1)
    FN = (y_true * (1 - y_pred)).sum(axis=-1)
    return float(TP / (TP + FN))

def recall(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Recall classification score.

    Args:
        y_true (np.ndarray): one-hot array of shape [C,...], where C is the number of classes
        y_pred (np.ndarray): one-hot array of shape [C,...], where C is the number of classes, must be the same shape as y_true

    Returns:
        recall (np.ndarray): 1d-array of shape [C]: recall (TP/TP + FP) for each class
    """
    assert y_true.shape == y_pred.shape

    C = y_true.shape[0]
    y_true = y_true.reshape(C, -1)
    y_pred = y_pred.reshape(C, -1)

    TP = (y_true * y_pred).sum(axis=-1)
    FP = ((1 - y_true) * y_pred).sum(axis=-1)
    return float(TP / (TP + FP))

def f1(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """F1 classification score.

    Args:
        y_true (np.ndarray): one-hot array of shape [C,...], where C is the number of classes
        y_pred (np.ndarray): one-hot array of shape [C,...], where C is the number of classes, must be the same shape as y_true

    Returns:
        F1-score (np.ndarray): 1d-array of shape [C]: F1-score for each class
    """
    assert y_true.shape == y_pred.shape

    C = y_true.shape[0]
    y_true = y_true.reshape(C, -1)
    y_pred = y_pred.reshape(C, -1)

    _precision = precision(y_true, y_pred)
    _recall =  recall(y_true, y_pred)

    f1_score = 2 * (_recall * _precision) / (_recall + _precision)
    return float(f1_score)

def confusion_matrix(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    """F1 classification score.

    Args:
        y_true (np.ndarray): one-hot array of shape [C,...], where C is the number of classes
        y_pred (np.ndarray): one-hot array of shape [C,...], where C is the number of classes, must be the same shape as y_true

    Returns:
        confusion matrix (np.ndarray): 2d-array of shape [C, C], where element [i, j] denote the number of samples of class i being classified as j
    """
    assert y_true.shape == y_pred.shape

    C = y_true.shape[0]
    y_true = y_true.reshape(C, -1)
    y_pred = y_pred.reshape(C, -1)

    return y_true @ y_pred.transpose()

def mse(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Mean-Squared Error regression score.

    Args:
        y_true (np.ndarray): array of any shape
        y_pred (np.ndarray): array of any shape, must be the same shape as y_true

    Returns:
        mse (np.ndarray): mean-squared error between y_true and y_pred
    """
    assert y_true.shape == y_pred.shape

    return float(((y_true - y_pred).astype(np.float32) ** 2).sum())

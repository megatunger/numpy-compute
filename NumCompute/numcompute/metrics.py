# Author: DUY ANH

import numpy as np

def accuracy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1 
    assert y_true.shape == y_pred.shape and y_true.dtype == y_pred.dtype

    acc = (y_true == y_pred).astype(np.float32)

    return float(acc.mean())

def precision(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Support beyond :term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    precision for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and precision for both classes are computed, then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    precision for all `labels` are either returned or averaged depending on the
    `average` parameter. Use `labels` specify the set of labels to calculate precision
    for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values. Sparse matrix is only supported when
        targets are of :term:`multilabel` type.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. Sparse matrix is only
        supported when targets are of :term:`multilabel` type.
    Returns
    -------
    precision : float (if average is not None) or array of float of shape (n_unique_labels,)
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.
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

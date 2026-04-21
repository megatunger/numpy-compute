# Author: DUY ANH
# - Classification:
#   - `accuracy`
#   - `precision`
#   - `recall`
#   - `f1`
#   - `confusion_matrix`
# - Regression:
#   - `mse(y_true, y_pred)`
# - **Bonus**: `roc_curve` and `auc` for binary classification

import numpy as np

from numcompute.utils import validate_metrics_array, validate_array_like

def accuracy_score(y_true : np.ndarray, y_pred : np.ndarray) -> float:
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
    y_true, y_pred = validate_metrics_array(y_true, y_pred)

    acc = (y_true == y_pred).astype(np.float32)

    return float(acc.mean())

def precision_score(y_true : np.ndarray, y_pred : np.ndarray, pos_label=1) -> float:
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
    
    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.
        
    Returns
    -------
    precision : float (if average is not None) or array of float of shape (n_unique_labels,)
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)
    
    y_true = np.where(y_true == pos_label, 1, 0)
    y_pred = np.where(y_pred == pos_label, 1, 0)

    TP = (y_true * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()

    if TP + FN == 0:
        # TODO: implement different behaviors when dividing by zero
        return 0.0

    return float(TP / (TP + FN))

def recall_score(y_true : np.ndarray, y_pred : np.ndarray, pos_label=1) -> float:
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Support beyond :term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    recall for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and recall for both classes are computed then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    recall for all `labels` are either returned or averaged depending on the `average`
    parameter. Use `labels` specify the set of labels to calculate recall for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values. Sparse matrix is only supported when
        targets are of :term:`multilabel` type.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. Sparse matrix is only
        supported when targets are of :term:`multilabel` type.

    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.
    
    Returns
    -------
    recall : float (if average is not None) or array of float of shape \
             (n_unique_labels,)
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)

    y_true = np.where(y_true == pos_label, 1, 0)
    y_pred = np.where(y_pred == pos_label, 1, 0)
    
    TP = (y_true * y_pred).sum()
    FP = ((1 - y_true) * y_pred).sum()

    if TP + FP == 0:
        # TODO: implement different behaviors when dividing by zero
        return 0.0

    return float(TP / (TP + FP))

def f1_score(y_true : np.ndarray, y_pred : np.ndarray, pos_label=1) -> float:
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is:

    .. math::
        \\text{F1} = \\frac{2 * \\text{TP}}{2 * \\text{TP} + \\text{FP} + \\text{FN}}

    Where :math:`\\text{TP}` is the number of true positives, :math:`\\text{FN}` is the
    number of false negatives, and :math:`\\text{FP}` is the number of false positives.
    F1 is by default
    calculated as 0.0 when there are no true positives, false negatives, or
    false positives.

    Support beyond :term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    F1 score for `pos_label`. If `average` is not `'binary'`, `pos_label` is ignored
    and F1 score for both classes are computed, then averaged or both returned (when
    `average=None`). Similarly, for :term:`multiclass` and :term:`multilabel` targets,
    F1 score for all `labels` are either returned or averaged depending on the
    `average` parameter. Use `labels` specify the set of labels to calculate F1 score
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
        
    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)

    _precision = precision_score(y_true, y_pred, pos_label=pos_label)
    _recall =  recall_score(y_true, y_pred, pos_label=pos_label)

    if _recall + _precision == 0:
        # TODO: implement different behaviors when dividing by zero
        return 0.0

    _f1_score = 2 * (_recall * _precision) / (_recall + _precision)
    return float(_f1_score)

def confusion_matrix(y_true : np.ndarray, y_pred : np.ndarray, labels=None) -> np.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
        
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)
    labels = validate_array_like(labels)
    
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    if labels is None:
        true_classes = np.unique(y_true)
        pred_classes = np.unique(y_pred)
        labels = np.union1d(true_classes, pred_classes)
        
    labels = labels.ravel()

    C = labels.size
    N = y_true.shape[0]

    y_true = y_true[np.newaxis].repeat(C, axis=0)
    y_pred = y_pred[np.newaxis].repeat(C, axis=0)
    expanded_classes = labels[:, np.newaxis].repeat(N, axis=1)

    y_true = (y_true == expanded_classes).astype(np.float32)
    y_pred = (y_pred == expanded_classes).astype(np.float32)

    return y_true @ y_pred.transpose()

def mean_squared_error(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """Mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float or array of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)

    error = (y_true - y_pred).astype(np.float32)
    return float((error ** 2).mean())

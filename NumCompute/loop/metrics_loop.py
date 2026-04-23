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

def accuracy_score_loop(y_true, y_pred) -> float:    
    total, correct = 0, 0
    for a, b in zip(y_true, y_pred):
        total += 1
        correct += (a == b)

    return float(correct / total)

def precision_score_loop(y_true, y_pred, pos_label=1) -> float:
    TP, FN = 0, 0
    
    for a, b in zip(y_true, y_pred):
        TP += (a == pos_label and b == pos_label)
        FN += (a == pos_label and b != pos_label)
        
    if TP + FN == 0:
        return 0.0

    return float(TP / (TP + FN))

def recall_score_loop(y_true, y_pred, pos_label=1) -> float:
    
    TP, FP = 0, 0
    for a, b in zip(y_true, y_pred):
        TP += (a == pos_label and b == pos_label)
        FN += (a != pos_label and b == pos_label)
        
    if TP + FP == 0:
        return 0.0

    return float(TP / (TP + FP))

def f1_score_loop(y_true, y_pred, pos_label=1) -> float:
    _precision = precision_score_loop(y_true, y_pred, pos_label=pos_label)
    _recall =  recall_score_loop(y_true, y_pred, pos_label=pos_label)

    if _recall + _precision == 0:
        # TODO: implement different behaviors when dividing by zero
        return 0.0

    _f1_score = 2 * (_recall * _precision) / (_recall + _precision)
    return float(_f1_score)

def confusion_matrix_loop(y_true, y_pred, labels=None) -> np.ndarray:
    
    if labels is None:
        labels = set(y_true).update(set(y_pred))
    
    labels = list(labels)
    cm = [[0] * len(labels)] * len(labels)
    
    for gt, pd in zip(y_true, y_pred):
        cm[labels.index(gt)][labels.index(pd)] += 1

    return cm

def mean_squared_error_loop(y_true, y_pred) -> float:
    
    n, sum_squared_error = 0, 0
    
    for a, b in zip(y_true, y_pred):
        n += 1
        sum_squared_error += (a - b) ** 2
    
    return float(sum_squared_error / n)

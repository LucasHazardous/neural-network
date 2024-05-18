import numpy as np


def mean_squared_error(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        return np.nan
    return (1/y_true.shape[0]) * np.sum((y_true - y_pred) ** 2)


def log_loss(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        return np.nan
    return (1 / y_true.shape[0]) * np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

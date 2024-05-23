import numpy as np


def measure_accuracy(true: np.ndarray, pred: np.ndarray, feature_num: int) -> float:
    one_hot_vector_equals = np.sum(true == pred, axis=1)
    result = np.count_nonzero(one_hot_vector_equals[one_hot_vector_equals == feature_num]) / true.shape[0]
    return result

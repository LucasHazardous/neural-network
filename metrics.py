import numpy as np


def measure_accuracy(true, pred, feature_num):
    one_hot_vector_equals = np.sum(true == pred, axis=1)
    result = np.count_nonzero(one_hot_vector_equals[one_hot_vector_equals == feature_num]) / true.shape[0]
    return result

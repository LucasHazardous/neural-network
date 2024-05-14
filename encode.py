import numpy as np


def one_hot_encode(y):
    result = np.zeros((y.size, y.max()+1))
    result[np.arange(y.size), y] = 1
    return result

import numpy as np


def add_polynomial_features(x: np.ndarray, max_degree: int) -> np.ndarray:
    result = np.zeros((x.shape[0], x.shape[1] * max_degree))
    result[:, :x.shape[1]] = x

    for degree in range(2, max_degree + 1):
        for feature in range(x.shape[1]):
            polynomial_feature = x[:, feature] ** degree
            result[:, x.shape[1] * (degree-1) + feature] = polynomial_feature
    return result

import numpy as np


def mean_and_scale_normalization(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)

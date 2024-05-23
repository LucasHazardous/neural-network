import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1/(1+np.exp(-x))

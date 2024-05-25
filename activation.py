from enum import Enum

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def relu_gradient(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    return np.exp(z) / np.sum(np.exp(z))


class ActivationEnum(Enum):
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"


def choose_activation_function_with_derivative(activation: ActivationEnum):
    match activation:
        case ActivationEnum.ReLU:
            return relu, relu_gradient
        case ActivationEnum.Sigmoid:
            return sigmoid, sigmoid_gradient

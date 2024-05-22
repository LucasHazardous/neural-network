import numpy as np

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def precise_predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float, regularization: float = 0):
        pass

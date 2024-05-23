import numpy as np
from activation import sigmoid
from model.base_model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, feature_num: int, label_num: int):
        self.label_num = label_num
        self.weights = np.random.uniform(low=-1, high=1, size=(label_num, feature_num + 1))

    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float, regularization: float = 0):
        extend_x = np.hstack((np.ones((x.shape[0], 1)), x))
        samples_count = x.shape[0]

        for i in range(self.label_num):
            one_vs_all_y = np.apply_along_axis(lambda label: 1 if label[i] == 1 else 0, axis=1, arr=y)

            h = sigmoid(extend_x @ self.weights[i])

            gradient = (1 / samples_count) * (
                np.transpose(extend_x) @ (-one_vs_all_y * np.log(h) - (1 - one_vs_all_y) * np.log(1 - h))
            )

            regularization_update = regularization / samples_count * self.weights[i]
            regularization_update[0] = 0
            self.weights[i] -= learning_rate * (gradient + regularization_update)

    def predict(self, x: np.ndarray) -> np.ndarray:
        extend_x = np.hstack((np.ones(1), x))
        h = sigmoid(self.weights @ extend_x)
        return h

    def precise_predict(self, x: np.ndarray) -> np.ndarray:
        h = self.predict(x)
        chosen = np.argmax(h)

        result = np.zeros(self.label_num)
        result[chosen] = 1
        return result

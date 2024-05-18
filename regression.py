import numpy as np
from activation import sigmoid


class LogisticRegression:
    def __init__(self, feature_num, label_num):
        self.label_num = label_num
        self.weights = np.random.uniform(low=-1, high=1, size=(label_num, feature_num + 1))

    def train(self, x, y, learning_rate):
        extend_x = np.hstack((np.ones((x.shape[0], 1)), x))
        samples_count = x.shape[0]

        for i in range(self.label_num):
            one_vs_all_y = np.apply_along_axis(lambda label: 1 if label[i] == 1 else 0, axis=1, arr=y)

            h = sigmoid(extend_x @ self.weights[i])

            gradient = (1 / samples_count) * (
                np.transpose(extend_x) @ (-one_vs_all_y * np.log(h) - (1 - one_vs_all_y) * np.log(1 - h))
            )

            self.weights[i] -= gradient * learning_rate

    def predict(self, x):
        extend_x = np.hstack((np.ones(1), x))
        h = sigmoid(self.weights @ extend_x)
        chosen = np.argmax(h)

        result = np.zeros(self.label_num)
        result[chosen] = 1
        return result

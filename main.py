import numpy as np

from metrics import measure_accuracy
from model.base_model import BaseModel
from model.neural_network import NeuralNetwork
from model.regression import LogisticRegression
from error import log_loss

from dataset.iris.iris_example import IrisExampleDataset

np.random.seed(42)

POLYNOMIAL_DEGREE = 3

iris_dataset = IrisExampleDataset(POLYNOMIAL_DEGREE)

X, y, test_x, test_y = iris_dataset.train_test_split(.2)


def logistic_regression_test():
    logistic_regression = LogisticRegression(
        IrisExampleDataset.IRIS_FEATURE_NUM * POLYNOMIAL_DEGREE,
        IrisExampleDataset.IRIS_LABEL_NUM
    )

    for i in range(10):
        logistic_regression.train(X, y, 0.05, 0.8)

    return measure_loss_and_accuracy_test(logistic_regression)


def neural_network_test():
    nn = NeuralNetwork(
        IrisExampleDataset.IRIS_FEATURE_NUM * POLYNOMIAL_DEGREE,
        [50],
        IrisExampleDataset.IRIS_LABEL_NUM
    )

    for i in range(10):
        nn.train(X, y, 1, 1)

    return measure_loss_and_accuracy_test(nn)


def measure_loss_and_accuracy_test(model: BaseModel):
    predicted = np.apply_along_axis(model.predict, 1, X)
    train_loss = log_loss(y, predicted)

    predicted_test = np.apply_along_axis(model.predict, 1, test_x)
    test_loss = log_loss(test_y, predicted_test)

    precise_predicted_test = np.apply_along_axis(model.precise_predict, 1, test_x)
    test_accuracy = measure_accuracy(test_y, precise_predicted_test, IrisExampleDataset.IRIS_LABEL_NUM)

    return train_loss, test_loss, test_accuracy


print(logistic_regression_test())
print(neural_network_test())

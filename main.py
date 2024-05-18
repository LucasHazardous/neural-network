import numpy as np

from encode import one_hot_encode
from regression import LogisticRegression
from error import log_loss
from features import generate_polynomial_features

np.random.seed(42)

arr = np.loadtxt("dataset/Iris.csv", delimiter=",", skiprows=1, dtype=str)
arr = arr[:, 1:]

mapper = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

vectorized_label_convert = np.vectorize(lambda label: mapper.index(label))

arr[:, 4] = vectorized_label_convert(arr[:, 4])

np.random.shuffle(arr)

X = arr[:, :4].astype(dtype=np.float16)
y = arr[:, 4].astype(dtype=np.uint8)

POLYNOMIAL_DEGREE = 3

X = (X - X.mean(axis=0)) / X.std(axis=0)
X = generate_polynomial_features(X, POLYNOMIAL_DEGREE)

y = one_hot_encode(y)

test_size = 30

test_x = X[:test_size]
test_y = y[:test_size]

X = X[test_size:]
y = y[test_size:]

logistic_regression = LogisticRegression(4 * POLYNOMIAL_DEGREE, 3)

for i in range(10):
    logistic_regression.train(X, y, 0.05, 0.8)

predicted = np.apply_along_axis(logistic_regression.predict, 1, X)

print("train loss", log_loss(y, predicted))

predicted_test = np.apply_along_axis(logistic_regression.predict, 1, test_x)

print("test loss", log_loss(test_y, predicted_test))

precise_predicted_test = np.apply_along_axis(logistic_regression.precise_predict, 1, test_x)

r = np.sum(test_y == precise_predicted_test, axis=1)
print("test accuracy", np.count_nonzero(r[r == 3]) / test_y.shape[0])

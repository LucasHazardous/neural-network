import numpy as np

from encode import one_hot_encode
from regression import LogisticRegression
from error import mean_squared_error

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

X = (X - X.mean()) / X.std()

y = one_hot_encode(y)

test_size = 30

test_x = X[:test_size]
test_y = y[:test_size]

X = X[test_size:]
y = y[test_size:]

logistic_regression = LogisticRegression(4, 3)

logistic_regression.train(X, y, 0.1)

predicted = np.apply_along_axis(logistic_regression.predict, 1, test_x)


print(mean_squared_error(test_y, predicted))

r = np.sum(test_y == predicted, axis=1)
print("accuracy", np.count_nonzero(r[r == 3]) / test_y.shape[0])

import numpy as np

from normalization import mean_and_scale_normalization
from encode import one_hot_encode
from features import add_polynomial_features

MAPPER = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]


class IrisExampleDataset(object):
    IRIS_FEATURE_NUM = 4
    IRIS_LABEL_NUM = 3

    def __init__(self, polynomial_degree=1):
        self.data = np.loadtxt("dataset/iris/Iris.csv", delimiter=",", skiprows=1, dtype=str)
        np.random.shuffle(self.data)

        self.__remove_id_column()
        self.__map_labels_to_numbers()
        self.X, self.y = self.__feature_label_split()

        self.X = mean_and_scale_normalization(self.X)
        self.X = add_polynomial_features(self.X, polynomial_degree)

        self.y = one_hot_encode(self.y)

    def __remove_id_column(self):
        self.data = self.data[:, 1:]

    def __map_labels_to_numbers(self):
        vectorized_label_convert = np.vectorize(MAPPER.index)

        self.data[:, 4] = vectorized_label_convert(self.data[:, 4])

    def __feature_label_split(self):
        x = self.data[:, :4].astype(dtype=np.float16)
        y = self.data[:, 4].astype(dtype=np.uint8)
        return x, y

    def train_test_split(self, test_size=0.2):
        test_size = int(self.data.shape[0] * test_size)

        test_x = self.X[:test_size]
        test_y = self.y[:test_size]

        x = self.X[test_size:]
        y = self.y[test_size:]

        return x, y, test_x, test_y

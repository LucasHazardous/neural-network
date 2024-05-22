import numpy as np
from activation import sigmoid


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, feature_layer_size, hidden_layers_sizes, result_layer_size):
        self.weights = []
        self.result_layer_size = result_layer_size
        layer_sizes = [feature_layer_size, *hidden_layers_sizes, result_layer_size]
        for i in range(1, len(layer_sizes)):
            weight = np.random.uniform(low=-1, high=1, size=(layer_sizes[i - 1] + 1, layer_sizes[i]))
            self.weights.append(weight)

    def train(self, x, y, learning_rate, regularization=0):
        sample_count = x.shape[0]

        gradients = [np.zeros(weight.shape) for weight in self.weights]

        for sample, label in zip(x, y):
            back_pass_result = self.backward_pass(sample, label, sample_count)
            for layer in range(len(gradients)):
                gradients[layer] += back_pass_result[layer]

        for layer in range(len(gradients)):
            weight_regularization = regularization / sample_count * self.weights[layer]
            weight_regularization[:, 0] = 0

            self.weights[layer] -= learning_rate * gradients[layer] + weight_regularization

    def backward_pass(self, sample, label, sample_count):
        activations, connections = self.forward_pass(sample)
        final_delta = activations[-1] - label
        final_deltas = [final_delta]

        final_grad = np.outer(final_delta / sample_count, activations[-2]).T
        final_grads = [final_grad]

        for i in range(len(self.weights) - 1, 0, -1):
            activation, connection, weight = activations[i - 1], connections[i - 1], self.weights[i]
            delta = weight[1:, :] @ final_deltas[0] * sigmoid_gradient(connection)
            grad = np.outer(delta / sample_count, activation)

            final_grads.insert(0, grad.T)
            final_deltas.insert(0, delta)

        return final_grads

    def forward_pass(self, x):
        activations = [np.hstack((np.ones(1), x))]
        connections = []
        for weight in self.weights:
            connection = activations[-1] @ weight
            connections.append(connection)
            activation = sigmoid(np.hstack((np.ones(1), connection)))
            activations.append(activation)
        activations[-1] = activations[-1][1:]
        return activations, connections

    def predict(self, x):
        result = x
        for weight in self.weights:
            result = np.hstack((np.ones(1), result))
            result = sigmoid(result @ weight)
        return result

    def precise_predict(self, x):
        h = self.predict(x)
        chosen = np.argmax(h)

        result = np.zeros(self.result_layer_size)
        result[chosen] = 1
        return result

import numpy as np

class ANN:
    def __init__(self, n_features, n_hidden=10, n_outputs=3, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w_hidden = np.random.randn(n_features, n_hidden)
        self.b_hidden = np.zeros(n_hidden)
        self.w_output = np.random.randn(n_hidden, n_outputs)
        self.b_output = np.zeros(n_outputs)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        y_encoded = np.zeros((y.size, y.max() + 1))
        y_encoded[np.arange(y.size), y] = 1

        for _ in range(self.n_iters):
            # Forward pass
            hidden_input = np.dot(X, self.w_hidden) + self.b_hidden
            hidden_output = self._sigmoid(hidden_input)
            output_layer_input = np.dot(hidden_output, self.w_output) + self.b_output
            predicted_output = self._sigmoid(output_layer_input)

            # Backpropagation
            output_error = y_encoded - predicted_output
            output_delta = output_error * self._sigmoid_derivative(predicted_output)
            hidden_error = output_delta.dot(self.w_output.T)
            hidden_delta = hidden_error * self._sigmoid_derivative(hidden_output)

            # Weight and bias updates
            self.w_output += hidden_output.T.dot(output_delta) * self.lr
            self.b_output += np.sum(output_delta, axis=0) * self.lr
            self.w_hidden += X.T.dot(hidden_delta) * self.lr
            self.b_hidden += np.sum(hidden_delta, axis=0) * self.lr

    def predict(self, X):
        hidden_output = self._sigmoid(np.dot(X, self.w_hidden) + self.b_hidden)
        output_layer = self._sigmoid(np.dot(hidden_output, self.w_output) + self.b_output)
        return np.argmax(output_layer, axis=1)

    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)

from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import optimizers
import losses
import layers


class Model:
    def __init__(self):
        self._layers: list[layers.Layer] = []
        self.loss: losses.Loss = None
        self.optimizer: optimizers.Optimizer = None

    def build(self, loss: losses.Loss, optimizer: optimizers.Optimizer):
        if loss is not None:
            self.loss = loss
        else:
            raise ValueError("Loss cannot be none.")
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            raise ValueError("Optimizer cannot be none.")

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, epochs: int = 5,
            batch_size: int = 32, val_data: npt.ArrayLike = None):
        x_val, y_val = val_data
        n = len(X)
        self._prepare_variables(X[0])
        for epoch in range(epochs):
            train_loss = 0.0
            val_acc = 0
            for batch in range(0, n, batch_size):
                if n - batch < batch_size:
                    continue
                train_loss = (self._train_network(
                    X[batch:batch + batch_size], y[batch:batch + batch_size]))
            v_loss, val_acc = (self.evaluate(x_val, y_val))
            print(
                f"Epoch {epoch+1}: Train Loss: {np.mean(train_loss)} Val Loss: {np.mean(v_loss)} \
                Val_Acc: {val_acc}/{len(y_val)}")

    def predict(self, X) -> Union[int, float]:
        x = np.transpose(X)

        for layer in self._layers[1:]:
            x, _ = layer(x)

        return x

    def evaluate(self, x_test, y_test) -> Tuple[float, int]:
        # y_pred = np.stack([np.argmax(self.predict(x)) for x in x_test])
        y_pred = np.stack([self.predict(x) for x in x_test])
        # y_test = np.stack([np.argmax(y) for y in y_test])
        loss = np.sum(self.loss(y_pred, y_test)) / len(y_test)
        y = [(np.argmax(y_p), np.argmax(y_t))
             for (y_p, y_t) in zip(y_pred, y_test)]
        # y = list(zip(y_pred, y_test))
        num_correct = sum([int(a == b) for a, b in y])
        return loss, num_correct

    def add_layer(self, layer: layers.Layer):
        self._layers.append(layer)

    def _train_network(self, x: np.ndarray, y: np.ndarray) -> float:

        gradient_w, gradient_b = self._backprop(x, y)

        for i, (gw, gb) in enumerate(zip(gradient_w, gradient_b)):
            # Add 1 because the gradients don't include the input layer.
            i += 1
            weights = self.optimizer.apply_gradients(
                gw, self._layers[i].get_weights())
            biases = self.optimizer.apply_gradients(
                gb, self._layers[i].get_biases())
            self._layers[i]._set_weights(weights)
            self._layers[i]._set_biases(biases)
        loss, _ = self.evaluate(x, y)
        return loss

    def _backprop(self, x, y):
        gradient_b = [np.zeros(len(b.get_biases()))
                      for b in self._layers[1:]]
        gradient_w = [np.zeros(w.get_weights().shape)
                      for w in self._layers[1:]]

        layer_activations = []
        layer_outputs = []
        activation = np.transpose(x)
        layer_activations.append(activation)

        for i, (b, w) in enumerate(zip(self._biases, self._weights)):
            activation, layer_output = self._layers[i + 1](activation)
            layer_activations.append(activation)
            layer_outputs.append(layer_output)
        activation_function = None
        delta = self.loss._delta(layer_outputs[-1],
                                 np.transpose(layer_activations[-1]), y, activation_function)

        # Calculate gradients for final layer.
        # Delta is (10, 32). This reduces it to 10.
        gradient_b[-1] = np.sum(delta, axis=1)
        gradient_w[-1] = np.matmul(delta, np.transpose(layer_activations[-2]))

        for i in range(2, len(self._layers)):
            activation_function = self._layers[-i].get_activation()
            a_prime = activation_function.prime(layer_outputs[-i])
            weights = np.transpose(self._layers[-i + 1].get_weights())
            delta = np.matmul(weights, delta)
            delta = delta * a_prime
            gradient_b[-i] = np.sum(delta, axis=1)
            gradient_w[-i] = np.matmul(
                delta, np.transpose(layer_activations[-i - 1]))

        return gradient_w, gradient_b

    def _prepare_variables(self, x: npt.ArrayLike):
        self._weights = []
        self._biases = []
        prev_shape = x.shape

        for i in range(1, len(self._layers)):
            self._layers[i].build(prev_shape)
            self._weights.append(self._layers[i].get_weights())
            self._biases.append(self._layers[i].get_biases())
            prev_shape = self._layers[i].get_output_shape()


def one_hot(data, categories, axis):
    data_out = np.zeros(shape=(data.shape[axis], len(categories)))
    data_out[np.arange(data.shape[axis]), data] = 1
    return data_out

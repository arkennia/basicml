import activations
from typing import Union
import numpy as np

import numpy.typing as npt
import optimizers
import losses
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
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
        n = X.shape[0]
        val_loss = []
        train_loss = []
        self._prepare_variables(X[0])
        for epoch in range(epochs):
            for batch in range(0, n, batch_size):
                train_loss.append(self._train_network(
                    X[batch:batch + batch_size], y[batch:batch + batch_size]))
                val_loss.append(self.evaluate(x_val, y_val))

    def predict(self, X) -> Union[int, float]:
        pass

    def evaluate(self, x_test, y_test) -> float:
        y_pred = [np.argmax(self.predict(x)) for x in x_test]
        loss = np.sum(self.loss(y_pred, y_test))
        return loss

    def add_layer(self, layer: layers.Layer):
        self._layers.append(layer)

    def _train_network(self, x: np.ndarray, y: np.ndarray) -> float:

        delta_gradient_w, delta_gradient_b = self._backprop(x, y)

    def _backprop(self, x, y):
        gradient_b = [np.zeros(len(b.get_biases()))
                      for b in self._layers[1:]]
        gradient_w = [np.zeros(w.get_weights().shape)
                      for w in self._layers[1:]]

        layer_activations = []
        layer_outputs = []
        activation = np.transpose(x)

        for i, (b, w) in enumerate(zip(self._biases, self._weights)):
            activation, layer_output = self._layers[i + 1](activation)
            layer_activations.append(activation)
            layer_outputs.append(layer_output)
        actvation_function = self._layers[-1].get_activation()
        if actvation_function is None:
            delta = self.loss._delta(
                activations[-1], y) * self._layers[-1].get_biases()
        else:
            delta = self.loss._delta(layer_outputs[-1],
                                     layer_activations[-1], y, actvation_function)
        gradient_b[-1] = np.sum(delta, axis=1)
        gradient_w[-1] = np.matmul(delta, np.transpose(layer_activations[-2]))

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


x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, shuffle=True)

model = Model()
model.build(losses.MSE(), optimizers.SGD(1e-3))

model.add_layer(layers.Input())
model.add_layer(layers.Dense(30, activations.Sigmoid()))
model.add_layer(layers.Dense(10, activations.Sigmoid()))

model.fit(x_train, y_train, val_data=(x_test, y_test))

"""Densely connected layer."""

from typing import Tuple, Union
import activations
from layers import Layer
from activations import Activation
import numpy.typing as npt
import numpy as np


class Dense(Layer):
    """Implements a Densely connected neural network layer. Derives from `Layer`."""

    def __init__(self, units: int, activation: Activation = None, trainable: bool = True):
        """
        Init a Dense layer.

        Parameters:
            units: The number of neurons for the layer.
            activation: The activation function to use.
            trainable: Whether or not it should update weights.
        """
        super().__init__(trainable=trainable)
        self.units = units
        self.activation = activation

    def build(self, input_shape: tuple):
        """
        Initialize the correct shapes. Create the bias and weight matricies.

        Parameters:
        input_shape: a tuple of the dimensions.
        """
        self.input_shape = input_shape
        self.output_shape = (self.units, input_shape[0])
        # self.weights = np.random.randn(
        #     self.output_shape[0], self.output_shape[1])
        # self.biases = np.random.randn(self.units)
        self.weights = np.random.default_rng(0).random(
            (self.output_shape[0], self.output_shape[1])) / np.sqrt(self.output_shape[1])
        self.biases = np.random.default_rng(0).random(self.units)
        self.__built = True

    def __call__(self, inputs) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Do a feedforward pass of the layer.

        If activation is `None`, the activation is linear (np.dot(weights, inputs) + b).

        Paramters:

        inputs: An array-like of inputs.

        Return:
        The activation of the layer.
        """
        if not self.__built:
            self.build(inputs.shape)
        # a = np.dot(self.weights, inputs) + self.biases
        a = np.matmul(self.weights, inputs)
        b = self.biases
        if a.ndim > 1:
            b = np.repeat(b[:, np.newaxis], a.shape[-1], axis=1)
        a = np.add(a, b)
        z = a  # pre-activation function output
        if self.activation is not None:
            a = self.activation(a)
        return a, z

    def get_biases(self) -> npt.ArrayLike:
        """Return a copy of the biases of the layer."""
        return self.biases

    def get_weights(self) -> npt.ArrayLike:
        """Return a copy of the biases of the layer."""
        return self.weights

    def get_output_shape(self) -> Tuple:
        """Return the output shape of the layer."""
        return self.output_shape

    def get_activation(self) -> Union[activations.Activation, None]:
        """Return a copy of the layer's activation function."""
        return self.activation

    def _set_weights(self, weights: npt.ArrayLike):
        if self.weights.shape == weights.shape:
            self.weights = weights
        else:
            raise ValueError(
                "Dense: Tried to set weights with different dimensions.")

    def _set_biases(self, biases: npt.ArrayLike):
        if self.biases.shape == biases.shape:
            self.biases = biases
        else:
            raise ValueError(
                "Dense: Tried to set biases with different dimensions.")

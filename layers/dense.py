from layers import Layer
from activations import Activation
import numpy.typing as npt
import numpy as np


class Dense(Layer):
    def __init__(self, units: int, activation: Activation = None, trainable: bool = True) -> None:
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
        self.output_shape = (self.units, input_shape[-1])
        self.weights = np.random.randn(self.output_shape)
        self.biases = np.random.randn(self.units)
        self.__built = True

    def call(self, inputs) -> npt.ArrayLike:
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
        a = np.dot(self.weights, inputs) + self.biases
        if self.activation is not None:
            a = self.activation(a)
        return a

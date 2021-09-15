"""Input layer."""

import activations
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from layers import Layer


class Input(Layer):
    """
    The input layer of the neural network. Derives from `Layer`.

    The purpose of this layer is to make sure the input is correctly shaped.
    """

    def __init__(self) -> None:
        """Init the layer."""
        super().__init__(trainable=False)

    def build(self, input_shape: tuple):
        """
        Build the layer with the given input shape.

        Parameters:
            input_shape: The shape to expect the inputs in.
        """
        self.weights = np.empty(
            input_shape)  # The input layer will not have any weights.
        self.__built = True
        self.shape = input_shape

    def __call__(self, inputs: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Take the inputs, and make sure they are in the right shape and type.

        Parameters:
            inputs: A numpy array containing rows of inputs.

        Returns:
            The inputs.
        """
        if not self.__built:
            self.build(inputs.shape)
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Inputs should be a numpy array/matrix.")
        return inputs, []

    def get_output_shape(self):
        """Return the output shape of the layer."""
        pass

    def get_weights(self) -> npt.ArrayLike:
        """Return the weights of the layer."""
        return np.empty(self.shape)

    def get_activation(self) -> Union[activations.Activation, None]:
        """Return the activation function of the layer."""
        return super().get_activation()

    def get_biases(self) -> npt.ArrayLike:
        """Return the biases of the layer."""
        return np.empty(self.shape)

    def _set_weights(self, weights: npt.ArrayLike):
        """Set the weights of the layer."""
        pass

    def _set_biases(self, biases: npt.ArrayLike):
        """Set the biases of the layer."""
        pass

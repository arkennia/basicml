import activations
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from layers import Layer


class Input(Layer):
    def __init__(self) -> None:
        super().__init__(trainable=False)

    def build(self, input_shape):
        self.weights = np.empty()  # The input layer will not have any weights.
        self.__built = True
        self.shape = input_shape

    def __call__(self, inputs) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        if not self.__built:
            self.build(inputs.shape)
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        return inputs, inputs

    def get_output_shape(self):
        """Get the output shape of the layer."""
        pass

    def get_weights(self) -> npt.ArrayLike:
        """Get the weights of the layer."""
        return np.empty()

    def get_activation(self) -> Union[activations.Activation, None]:
        return super().get_activation()

    def get_biases(self) -> npt.ArrayLike:
        """Get the biases of the layer."""
        return np.empty()

    def _set_weights(self, weights: npt.ArrayLike):
        """Set the weights of the layer."""
        pass

    def _set_biases(self, biases: npt.ArrayLike):
        """Set the biases of the layer."""
        pass

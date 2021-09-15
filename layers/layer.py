"""Layer base class."""

from abc import ABC, abstractmethod
import activations
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt


class Layer(ABC):
    """
    The base class that all layers derive from.

    All layers must implement the following functions:
        build:
        get_output_shape
        get_weights
        get_biases
        get_activation
        _set_weights
        _set_biases
    """

    def __init__(self, trainable=True) -> None:
        """Init the layer."""
        super().__init__()
        self.trainable = trainable
        self.weights = None
        self.__built = False

    @abstractmethod
    def build(self, input_shape):
        """
        Initialize the weights for the layer based on input size and any other intialization that needs to be done.

        This cannot be done until the input shape is known
        """
        self.weights = np.zeros(shape=input_shape, dtype='float32')
        self.__built = True

    @abstractmethod
    def get_output_shape(self) -> Tuple:
        """Return the output shape of the layer."""
        pass

    @abstractmethod
    def get_weights(self) -> npt.ArrayLike:
        """Return the weights of the layer."""
        return np.empty()

    @abstractmethod
    def get_biases(self) -> npt.ArrayLike:
        """Return the biases of the layer."""
        return np.empty()

    @abstractmethod
    def get_activation(self) -> Union[activations.Activation, None]:
        """Return the activation of the layer."""
        return None

    @abstractmethod
    def _set_weights(self, weights: npt.ArrayLike):
        """Set the weights of the layer."""
        pass

    @abstractmethod
    def _set_biases(self, biases: npt.ArrayLike):
        """Set the biases of the layer."""
        pass

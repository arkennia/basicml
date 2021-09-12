from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Layer(ABC):
    def __init__(self, trainable=True) -> None:
        super().__init__()
        self.trainable = trainable
        self.weights = np.empty()
        self.__built = False

    @abstractmethod
    def build(self, input_shape):
        """
        Initialize the weights for the layer based on input size and any other intialization that needs to be done
        after knowing the input size.
        """
        self.weights = np.zeros(shape=input_shape, dtype='float32')
        self.__built = True

    @abstractmethod
    def call(self, inputs, ** kwargs) -> Any:
        """
        Call the layer on the given inputs and return some output.
        """
        pass

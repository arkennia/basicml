import numpy as np
import numpy.typing as npt
from layers import Layer


class Input(Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        self.weights = np.empty()  # The input layer will not have any weights.
        self.__built = True
        self.shape = input_shape

    def call(self, inputs) -> npt.ArrayLike:
        if not self.__built:
            self.build(inputs.shape)
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        return inputs

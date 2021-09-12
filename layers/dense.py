from layers import Layer
import numpy.typing as npt
import numpy as np


class Dense(Layer):
    def __init__(self, units, trainable=True) -> None:
        super().__init__(trainable=trainable)
        self.units = units

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (self.units, input_shape[-1])
        self.weights = np.random.randn(self.output_shape)
        self.biases = np.random.randn(self.units)
        self.__built = True

    def call(self, inputs) -> npt.ArrayLike:
        a = inputs
        for b, w in zip(self.weights, self.biases):
            a = np.dot(w, a) + b
        return a

"""Sigmoid activation."""

from activations.activation import Activation
import numpy as np
import numpy.typing as npt


class Sigmoid(Activation):
    """Implements the Sigmoid activation function. Derives from `Activation`."""

    def __call__(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute sigmoid(z).

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The activation output.
        """
        return self._apply(z)

    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the output of the derivative of the sigmoid function on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The output of the derivative of sigmoid on z.
        """
        return self._apply(z) * (1 - self._apply(z))

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """Compute the activation of the function on z."""
        exp = np.exp(-z)
        return 1.0 / (1.0 + exp)
